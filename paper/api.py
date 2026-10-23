import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Literal
import pymupdf
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from .db import database, Paper, Chunk, QueryLog, uid, record
from .retrieval import search, embed, embed_mode
from .answers import answer

class Ask(BaseModel):
    question:str=Field(min_length=2,max_length=1000)
    paper_ids:list[str]=Field(min_length=1,max_length=8)
    mode:Literal['hybrid','keyword','vector']='hybrid'
    top_k:int=Field(default=5,ge=1,le=10)
    use_model:bool=True

class Compare(BaseModel):
    paper_ids:list[str]=Field(min_length=1,max_length=4)

def create_app(db_url=None,data_dir=None):
    root=Path(data_dir or os.getenv('DATA_DIR','data')).resolve();root.mkdir(parents=True,exist_ok=True)
    files=root/'pdfs';files.mkdir(exist_ok=True)
    factory=database(db_url)
    app=FastAPI(title='PaperTrail · Evidence-based paper reader',version='1.0.0')
    app.state.sessions=factory;app.state.data_dir=root
    parsing=threading.BoundedSemaphore(2)
    def session():
        with factory() as s:yield s
    def paper_or_404(s,paper_id):
        paper=s.get(Paper,paper_id)
        if not paper:raise HTTPException(404,'论文不存在')
        return paper
    # 同样是 384 维，不同模型的向量空间也不兼容；配置改变后必须重建索引。
    def check_papers(s,ids):
        papers=[paper_or_404(s,p) for p in dict.fromkeys(ids)]
        if any(p.embedding_model!=embed_mode() for p in papers):
            raise HTTPException(409,'嵌入配置与已索引文档不同。请使用原配置或在新数据库中重新导入。')
        return papers

    @app.get('/')
    def index():return FileResponse(Path(__file__).parent/'static/index.html')

    @app.get('/health')
    def health():return {'status':'ok','embedding_model':embed_mode(),'vector_store':factory.kw['bind'].dialect.name,
        'llm_configured':bool(os.getenv('OLLAMA_URL')),'model':os.getenv('LLM_MODEL') if os.getenv('OLLAMA_URL') else None,
        'limits':{'upload_mb':25,'pages':80},'scope':'单用户私有演示，扫描 PDF 不支持 OCR'}

    @app.get('/api/papers')
    def papers(s=Depends(session)):
        return [record(p) for p in s.scalars(select(Paper).order_by(Paper.created.desc()))]

    @app.post('/api/papers',status_code=201)
    def upload(file:UploadFile=File(...),s=Depends(session)):
        if not parsing.acquire(blocking=False):raise HTTPException(429,'解析任务繁忙，请稍后重试')
        try:
            with tempfile.TemporaryDirectory(prefix='ingest-',dir=root) as tmp:
                src=Path(tmp)/'input.pdf';result_path=Path(tmp)/'parsed.json'
                digest=hashlib.sha256();total=0
                with src.open('wb') as f:
                    while block:=file.file.read(1024*1024):
                        total+=len(block)
                        if total>25*1024*1024:raise HTTPException(413,'PDF 不能超过 25 MB')
                        digest.update(block);f.write(block)
                if not src.read_bytes()[:1024].lstrip().startswith(b'%PDF-'):raise HTTPException(422,'文件内容不是 PDF')
                sha=digest.hexdigest()
                # 内容哈希去重，避免同一 PDF 换文件名后重复解析和生成向量。
                existing=s.scalar(select(Paper).where(Paper.sha256==sha))
                if existing:return {**record(existing),'deduplicated':True}
                try:
                    # 解析放在独立子进程，并设置超时；Linux 子进程还限制 CPU 时间与内存。
                    run=subprocess.run([sys.executable,'-m','paper.parse_worker',str(src),str(result_path)],
                        cwd=Path(__file__).resolve().parents[1],timeout=45,capture_output=True)
                except subprocess.TimeoutExpired:raise HTTPException(422,'PDF 解析超时，请减少页数')
                if run.returncode or not result_path.exists():raise HTTPException(422,'PDF 无法解析或超过资源限制')
                parsed=json.loads(result_path.read_text(encoding='utf-8'))
                if parsed.get('error'):raise HTTPException(422,parsed['error'])
                # 向量使用“章节 + 正文”，但原文内容与坐标单独保留，引用仍可逐字核对。
                vectors=embed([c['section']+' '+c['content'] for c in parsed['chunks']])
                doc=Paper(id=uid(),sha256=sha,title=parsed['title'],pages=parsed['pages'],
                          warnings=parsed['warnings'],embedding_model=embed_mode())
                destination=files/(doc.id+'.pdf')
                s.add(doc);s.flush()
                s.add_all(Chunk(paper_id=doc.id,**chunk,embedding=vector) for chunk,vector in zip(parsed['chunks'],vectors))
                src.replace(destination)
                try:s.commit()
                except Exception:
                    s.rollback();destination.unlink(missing_ok=True);raise
                return {**record(doc),'chunks':len(vectors),'deduplicated':False}
        except IntegrityError:
            s.rollback()
            existing=s.scalar(select(Paper).where(Paper.sha256==sha))
            if existing:return {**record(existing),'deduplicated':True}
            raise
        finally:parsing.release();file.file.close()

    @app.get('/api/papers/{paper_id}/chunks')
    def chunks(paper_id:str,page:int|None=None,s=Depends(session)):
        paper_or_404(s,paper_id)
        query=select(Chunk).where(Chunk.paper_id==paper_id)
        if page is not None:query=query.where(Chunk.page==page)
        return [record(c) for c in s.scalars(query.order_by(Chunk.ordinal))]

    @app.get('/api/papers/{paper_id}/file')
    def pdf(paper_id:str,s=Depends(session)):
        paper_or_404(s,paper_id)
        return FileResponse(files/(paper_id+'.pdf'),media_type='application/pdf',
                            headers={'Content-Disposition':'inline; filename="paper.pdf"','X-Content-Type-Options':'nosniff'})

    @app.get('/api/papers/{paper_id}/pages/{page}.png')
    def page_image(paper_id:str,page:int,chunk_id:str|None=None,s=Depends(session)):
        p=paper_or_404(s,paper_id)
        if not 1<=page<=p.pages:raise HTTPException(404,'页码不存在')
        chunk=s.get(Chunk,chunk_id) if chunk_id else None
        # 联合校验文本块、论文和页码，防止把其他论文的框画到当前页面上。
        if chunk_id and (not chunk or chunk.paper_id!=paper_id or chunk.page!=page):
            raise HTTPException(400,'引用与论文页码不匹配')
        with pymupdf.open(files/(paper_id+'.pdf')) as doc:
            pg=doc[page-1]
            # 在内存中画框并渲染图片，不修改用户上传的 PDF 文件。
            if chunk:pg.draw_rect(pymupdf.Rect(chunk.bbox),color=(.9,.48,.1),fill=(1,.86,.3),fill_opacity=.22,width=1.5,overlay=True)
            scale=min(1.5,1800/max(pg.rect.width,pg.rect.height))
            image=pg.get_pixmap(matrix=pymupdf.Matrix(scale,scale),alpha=False).tobytes('png')
        return Response(image,media_type='image/png',headers={'Cache-Control':'private, max-age=600'})

    @app.post('/api/search')
    def retrieve(body:Ask,s=Depends(session)):
        check_papers(s,body.paper_ids)
        return {'hits':search(s,body.paper_ids,body.question,body.top_k,body.mode),'embedding_model':embed_mode()}

    @app.post('/api/ask')
    def ask(body:Ask,s=Depends(session)):
        start=time.perf_counter();check_papers(s,body.paper_ids)
        # 问答只检索用户选中的论文；然后选证、校验、保存问答结果与检索轨迹。
        hits=search(s,body.paper_ids,body.question,body.top_k,body.mode)
        result={**answer(body.question,hits,body.use_model),'hits':hits,'embedding_model':embed_mode()}
        elapsed=int((time.perf_counter()-start)*1000);result['duration_ms']=elapsed
        log=QueryLog(question=body.question,paper_ids=body.paper_ids,mode=body.mode,duration_ms=elapsed,result=result)
        s.add(log);s.commit()
        return {**result,'query_id':log.id}

    @app.post('/api/compare')
    def compare(body:Compare,s=Depends(session)):
        papers=check_papers(s,body.paper_ids);rows=[]
        # 每篇论文分别检索四个字段再做对照，不把不同数据集上的分数直接判成优劣。
        for p in papers:
            fields={}
            for name,q in {'datasets':'dataset benchmark 数据集','metrics':'metric accuracy f1 指标',
                           'methods':'method approach 方法','limitations':'limitations 局限'}.items():
                fields[name]=answer(q,search(s,[p.id],q,2),use_model=False)
            rows.append({'paper_id':p.id,'title':p.title,'fields':fields})
        return {'rows':rows,'note':'按字段检索的证据对照。不同数据集、拆分和指标定义不可直接比较数值；未找到的字段不作推断。'}

    @app.get('/api/history')
    def history(s=Depends(session)):
        return [record(q) for q in s.scalars(select(QueryLog).order_by(QueryLog.created.desc()).limit(30))]

    @app.get('/api/evaluations')
    def evaluations():
        path=Path(__file__).resolve().parents[1]/'reports/evaluation.json'
        if not path.exists():return {'status':'not_run','message':'请运行 scripts/evaluate.py 生成实际评测报告'}
        data=json.loads(path.read_text(encoding='utf-8'))
        return {'dataset':data['dataset'],'health':data['health'],
                'reports':[{k:v for k,v in r.items() if k!='cases'} for r in data['reports']],
                'model_probe_source':data['model_probe']['source'],'limits':data['limits']}

    return app
