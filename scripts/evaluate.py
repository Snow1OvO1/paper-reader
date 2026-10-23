"""Reproducible hand-authored synthetic benchmark; not a real-paper quality claim."""
import json
import os
from pathlib import Path
import time
import httpx

ROOT=Path(__file__).resolve().parents[1]
def main():
    base=os.getenv('PAPER_URL','http://127.0.0.1:18100');ids={};reports=[]
    with httpx.Client(base_url=base,timeout=120,trust_env=False) as client:
        for name in ['cedar','maple']:
            with (ROOT/'fixtures'/f'{name}.pdf').open('rb') as f:r=client.post('/api/papers',files={'file':(name+'.pdf',f,'application/pdf')})
            r.raise_for_status();ids[name]=r.json()['id']
        cases=json.loads((ROOT/'fixtures/evaluation.json').read_text())
        for mode in ['keyword','vector','hybrid']:
            rows=[];start=time.perf_counter()
            for item in cases:
                r=client.post('/api/ask',json={'question':item['question'],'paper_ids':[ids[item['paper']]],
                                             'mode':mode,'use_model':False,'top_k':5});r.raise_for_status();data=r.json()
                found=any(h['page']==item.get('page') and item.get('contains','') in h['text'] for h in data['hits'])
                correct=data['status']=='no_evidence' if item.get('abstain') else found
                loc_valid=all(any(c['chunk_id']==h['chunk_id'] and c['page']==h['page'] and c['quote'] in h['text'] and c['paper_id']==ids[item['paper']] for h in data['hits']) for c in data['citations'])
                rows.append({**item,'retrieval_or_abstention_ok':correct,'citation_location_ok':loc_valid,
                             'status':data['status'],'duration_ms':data['duration_ms'],'citations':len(data['citations'])})
            positive=[r for r in rows if not r.get('abstain')];negative=[r for r in rows if r.get('abstain')]
            cited_rows=[r for r in rows if r['citations']>0]
            reports.append({'mode':mode,'recall_at_5':sum(r['retrieval_or_abstention_ok'] for r in positive)/len(positive),
                'abstention_accuracy':sum(r['retrieval_or_abstention_ok'] for r in negative)/len(negative),
                'citation_location_check_rate':sum(r['citation_location_ok'] for r in cited_rows)/len(cited_rows) if cited_rows else None,
                'queries_with_citations':len(cited_rows),'total_citations':sum(r['citations'] for r in rows),
                'duration_ms':int((time.perf_counter()-start)*1000),'cases':rows})
        compare=client.post('/api/compare',json={'paper_ids':list(ids.values())});compare.raise_for_status()
        local_model=client.post('/api/ask',json={'question':'Which dataset was used?','paper_ids':[ids['cedar']],'use_model':True});local_model.raise_for_status()
        model_result=local_model.json()
        output={'dataset':'10 hand-authored synthetic questions; 2 synthetic PDFs; not a real-paper benchmark',
                'health':client.get('/health').json(),'paper_ids':ids,'reports':reports,
                'comparison_rows':len(compare.json()['rows']),'model_probe':model_result,
                'limits':'Quote substring and location checks do not measure entailment or completeness. No latency distribution or semantic accuracy claim.'}
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports/evaluation.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:output[k] for k in ['dataset','health','comparison_rows']},ensure_ascii=False))
    print([(r['mode'],r['recall_at_5'],r['abstention_accuracy']) for r in reports])
    print('Model probe:',model_result['source'])

if __name__=='__main__':main()
