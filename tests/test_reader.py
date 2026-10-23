from pathlib import Path
import json
import os
import pymupdf
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select,func,delete
from sqlalchemy.engine import make_url
from paper.api import create_app
from paper.db import Paper,Chunk,QueryLog
from paper.parser import parse_pdf,ordered_blocks
from paper.answers import validate_selection,answer

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def client(tmp_path):
    if os.getenv('TEST_DATABASE_URL'):
        assert make_url(os.environ['TEST_DATABASE_URL']).database.endswith('_test'), 'Use a dedicated *_test database'
    app=create_app(os.getenv('TEST_DATABASE_URL',f'sqlite:///{tmp_path}/db.sqlite'),tmp_path/'data')
    if os.getenv('TEST_DATABASE_URL'):
        with app.state.sessions.begin() as s:
            s.execute(delete(QueryLog));s.execute(delete(Chunk));s.execute(delete(Paper))
    with TestClient(app) as client:yield client

def upload(client,name='cedar'):
    response=client.post('/api/papers',files={'file':(name+'.pdf',(ROOT/'fixtures'/f'{name}.pdf').read_bytes(),'application/pdf')})
    assert response.status_code==201,response.text
    return response.json()

def test_pdf_page_geometry_and_column_order():
    doc=parse_pdf(ROOT/'fixtures/cedar.pdf')
    chunks=doc['chunks'];text='\n'.join(c['content'] for c in chunks)
    assert doc['pages']==2
    assert text.index('2 Method')<text.index('3 Datasets')
    assert all(c['page']==2 for c in chunks if '76.1' in c['content'])
    assert all(len(c['bbox'])==4 and c['bbox'][2]>c['bbox'][0] for c in chunks)
    assert 'SYNTHETIC TEST FIXTURE' not in text  # repeated page header removed
    assert doc['title']=='Cedar: Evidence Retrieval Study'

def test_section_context_retrieval(client):
    p=upload(client)
    r=client.post('/api/search',json={'question':'What are the limitations?','paper_ids':[p['id']],'mode':'hybrid'}).json()
    assert any(h['page']==2 and 'English' in h['text'] for h in r['hits'])

def test_keyword_search_survives_embedding_failure(client,monkeypatch):
    p=upload(client)
    def fail(*args,**kwargs):raise RuntimeError('Embedding service unavailable')
    monkeypatch.setattr('paper.retrieval.embed',fail)
    response=client.post('/api/search',json={'question':'CedarQA dataset','paper_ids':[p['id']],'mode':'keyword'})
    assert response.status_code==200
    assert any('CedarQA' in h['text'] for h in response.json()['hits'])

def test_upload_deduplicates_and_persists(client):
    first=upload(client);second=upload(client)
    assert first['id']==second['id'] and second['deduplicated']
    assert client.get('/api/papers/'+first['id']+'/file').content.startswith(b'%PDF-')
    assert client.get('/api/papers/'+first['id']+'/chunks').json()

@pytest.mark.parametrize('mode',['keyword','vector','hybrid'])
def test_retrieval_and_chinese_alias(client,mode):
    p=upload(client)
    r=client.post('/api/ask',json={'question':'论文使用了哪些数据集？','paper_ids':[p['id']],'mode':mode,'use_model':False}).json()
    assert r['status']=='answered'
    assert any('CedarQA' in c['quote'] and c['page']==1 for c in r['citations'])
    assert r['query_id']

def test_no_evidence(client):
    p=upload(client)
    r=client.post('/api/ask',json={'question':'火星地震震级具体数值是多少？','paper_ids':[p['id']],'use_model':False}).json()
    assert r['status']=='no_evidence' and not r['citations']

def test_scope_and_compare(client):
    cedar=upload(client);maple=upload(client,'maple')
    r=client.post('/api/ask',json={'question':'Which dataset was used?','paper_ids':[cedar['id']],'use_model':False}).json()
    assert all(c['paper_id']==cedar['id'] and 'MapleQA' not in c['quote'] for c in r['citations'])
    comparison=client.post('/api/compare',json={'paper_ids':[cedar['id'],maple['id']]}).json()
    assert len(comparison['rows'])==2
    assert all(set(row['fields'])=={'datasets','metrics','methods','limitations'} for row in comparison['rows'])

def test_quote_guard_rejects_hallucination():
    hits=[{'chunk_id':'c','paper_id':'p','page':1,'section':'method','bbox':[1,2,3,4],'text':'This exact original sentence is evidence.'}]
    assert validate_selection({'citations':[{'chunk_id':'c','quote':'This exact original sentence'}]},hits)
    for data in [{'citations':[{'chunk_id':'fake','quote':'This exact original sentence'}]},
                 {'citations':[{'chunk_id':'c','quote':'The accuracy is 99 percent.'}]}, {'citations':[]}]:
        with pytest.raises(ValueError):validate_selection(data,hits)

def test_extract_preserves_wrapped_sentence():
    text='Our method uses a sparse keyword retriever and a\nlocal reranker. Another unrelated statement follows.'
    hits=[{'chunk_id':'c','paper_id':'p','page':1,'section':'method','bbox':[1,2,3,4],'text':text}]
    result=answer('method',hits,use_model=False)
    assert result['citations'][0]['quote']=='Our method uses a sparse keyword retriever and a\nlocal reranker.'

def test_highlight_is_bound_to_paper_page(client):
    a=upload(client);b=upload(client,'maple')
    chunks=client.get('/api/papers/'+a['id']+'/chunks').json();chunk=chunks[0]
    url=f"/api/papers/{a['id']}/pages/{chunk['page']}.png?chunk_id={chunk['id']}"
    r=client.get(url);assert r.status_code==200 and r.content.startswith(b'\x89PNG')
    assert client.get(f"/api/papers/{b['id']}/pages/1.png?chunk_id={chunk['id']}").status_code==400
    assert client.get(f"/api/papers/{a['id']}/pages/99.png").status_code==404

def test_invalid_and_scanned_pdf(client):
    assert client.post('/api/papers',files={'file':('x.pdf',b'not a pdf')}).status_code==422
    doc=pymupdf.open();doc.new_page();blank=doc.tobytes();doc.close()
    r=client.post('/api/papers',files={'file':('blank.pdf',blank)})
    assert r.status_code==422 and 'OCR' in r.text

def test_encrypted_pdf(client):
    doc=pymupdf.open(ROOT/'fixtures/cedar.pdf')
    encrypted=doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,owner_pw='owner',user_pw='user');doc.close()
    assert client.post('/api/papers',files={'file':('locked.pdf',encrypted)}).status_code==422

def test_missing_paper_and_empty_scope(client):
    assert client.post('/api/ask',json={'question':'dataset','paper_ids':[]}).status_code==422
    assert client.post('/api/ask',json={'question':'dataset','paper_ids':['not-found']}).status_code==404

def test_model_failure_falls_back(monkeypatch):
    import httpx
    monkeypatch.setenv('OLLAMA_URL','http://unavailable')
    def fail(*args,**kwargs):raise httpx.ConnectError('offline')
    monkeypatch.setattr(httpx.Client,'post',fail)
    hits=[{'chunk_id':'c','paper_id':'p','page':1,'section':'dataset','bbox':[1,2,3,4],'text':'The dataset is CedarQA and contains 600 questions.'}]
    result=answer('dataset',hits,True)
    assert result['source']=='extractive' and result['citations'][0]['quote'] in hits[0]['text']
