import hashlib
import math
import os
import re
from collections import Counter
from functools import lru_cache
from sqlalchemy import select
from .db import Chunk, DIM

STOP=set('what which how the a an is are was were does do did this that paper study used use using of in for to and on with please tell me'.split())
ASPECTS={'数据集':'dataset datasets benchmark','指标':'metric accuracy f1 precision recall',
         '方法':'method approach model','对比':'baseline comparison compared',
         '消融':'ablation removed component','局限':'limitation limitations future',
         '训练':'training train','参数':'parameter parameters','准确率':'accuracy'}

# 英文按词、中文按连续二字切词，供 BM25 使用；不是训练出来的中文分词模型。
def tokens(text):
    english=[t for t in re.findall(r'[a-z0-9]+(?:[-.][a-z0-9]+)*',text.lower()) if t not in STOP]
    chinese=re.findall(r'[\u4e00-\u9fff]+',text)
    return english+[s[i:i+2] for s in chinese for i in range(len(s)-1)]

# 为常见中文字段补英文关键词，提高中文问题检索英文论文的可用性。
def expand(question):
    return question+' '+' '.join(v for k,v in ASPECTS.items() if k in question)

def embed_mode(): return os.getenv('EMBEDDING_MODEL','hash-384-v1')

@lru_cache(maxsize=1)
def semantic_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(embed_mode(),device='cpu',cache_folder=os.getenv('MODEL_CACHE','data/models'))

@lru_cache(maxsize=1)
# 缓存 CPU 模型与分词器，避免每次请求重新加载；不占用服务器已有模型的 GPU 配置。
def onnx_model():
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from pathlib import Path
    folder=Path(os.environ['E5_MODEL_DIR'])
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(folder/'onnx/model_qint8_avx512_vnni.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    tokenizer=Tokenizer.from_file(str(folder/'tokenizer.json'))
    tokenizer.enable_truncation(max_length=512);tokenizer.enable_padding(pad_id=1,pad_token='<pad>')
    return session,tokenizer

def embed(texts,query=False):
    if not texts:return []
    if embed_mode()=='e5-onnx-int8-v1':
        import numpy as np
        session,tokenizer=onnx_model();vectors=[]
        for start in range(0,len(texts),8):
            # E5 区分 query 和 passage 前缀；问题与文档必须使用同一模型和预处理方式。
            batch=tokenizer.encode_batch([('query: ' if query else 'passage: ')+t for t in texts[start:start+8]])
            ids=np.asarray([e.ids for e in batch],dtype=np.int64)
            mask=np.asarray([e.attention_mask for e in batch],dtype=np.int64)
            values={'input_ids':ids,'attention_mask':mask,'token_type_ids':np.zeros_like(ids)}
            hidden=session.run(None,{i.name:values[i.name] for i in session.get_inputs()})[0]
            # masked mean pooling 排除 padding，再做 L2 归一化，得到可用余弦比较的 384 维向量。
            pooled=(hidden*mask[:,:,None]).sum(axis=1)/np.maximum(mask.sum(axis=1)[:,None],1)
            pooled=pooled/np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-12)
            if pooled.shape[1]!=DIM:raise ValueError('Embedding dimensions do not match index')
            vectors.extend(pooled.tolist())
        return vectors
    if embed_mode()!='hash-384-v1':
        prefixed=[('query: ' if query else 'passage: ')+t for t in texts]
        return semantic_model().encode(prefixed,normalize_embeddings=True,batch_size=16).tolist()
    # 默认无模型时的词项哈希基线，不具备真正的语义理解能力，界面必须如实标明。
    vectors=[]
    for text in texts:
        vector=[0.0]*DIM
        for term,count in Counter(tokens(text)).items():
            digest=hashlib.sha256(term.encode()).digest()
            index=int.from_bytes(digest[:4],'little')%DIM
            vector[index]+=(1 if digest[4]%2 else -1)*(1+math.log(count))
        norm=math.sqrt(sum(v*v for v in vector)) or 1
        vectors.append([v/norm for v in vector])
    return vectors

def search(session,paper_ids,question,k=5,mode='hybrid'):
    chunks=list(session.scalars(select(Chunk).where(Chunk.paper_id.in_(paper_ids)).order_by(Chunk.ordinal)))
    if not chunks:return []
    query=expand(question); terms=set(tokens(query))
    # 把章节标题加入检索上下文：例如正文没写 limitations，也能通过所属章节找到它。
    counters=[Counter(tokens(c.section+' '+c.content)) for c in chunks]
    lengths=[sum(c.values()) for c in counters];avg=sum(lengths)/len(lengths) or 1
    # BM25 同时考虑词频、词的稀有程度和文本长度；不是简单计算关键词出现次数。
    df=Counter(t for c in counters for t in c)
    lexical={}
    for chunk,counter,length in zip(chunks,counters,lengths):
        score=0
        for term in terms:
            freq=counter[term]
            idf=math.log(1+(len(chunks)-df[term]+.5)/(df[term]+.5))
            score+=idf*(freq*2.5)/(freq+1.5*(.25+.75*length/avg)) if freq else 0
        lexical[chunk.id]=score
    ranked_lex=sorted(chunks,key=lambda c:lexical[c.id],reverse=True)
    # keyword 模式不加载嵌入模型，因此向量服务不可用时仍可检索。
    vector=embed([query],query=True)[0] if mode!='keyword' else None
    if mode=='keyword':ranked_vec=[]
    # pgvector 对限定论文集合做精确余弦距离排序；当前没有启用 HNSW/IVFFlat 近似索引。
    elif session.bind.dialect.name=='postgresql':
        ranked_vec=list(session.scalars(select(Chunk).where(Chunk.paper_id.in_(paper_ids))
                        .order_by(Chunk.embedding.cosine_distance(vector)).limit(max(20,k))))
    else:
        ranked_vec=sorted(chunks,key=lambda c:sum(a*b for a,b in zip(c.embedding,vector)),reverse=True)[:max(20,k)]
    scores=Counter()
    rankings=[ranked_lex] if mode=='keyword' else [ranked_vec] if mode=='vector' else [ranked_lex,ranked_vec]
    # RRF 融合名次：每条检索路线贡献 1/(60+排名)。
    # 不直接相加 BM25 原始分数和余弦分数，因为二者量纲不同。
    for ranking in rankings:
        for rank,c in enumerate(ranking[:30]): scores[c.id]+=1/(60+rank+1)
    # Hash vectors are a lexical baseline: collisions alone must never justify an answer.
    # 候选还需满足相关性门槛；0.86 为当前启发式阈值，不是通用正确概率。
    # 词项重合仍可能不相关，因此“有引用”不等于“答案充分正确”。
    candidates=[c for c in chunks if len(c.content)>=40 and c.id in scores and
                (lexical[c.id]>0 if mode=='keyword' or embed_mode()=='hash-384-v1' else (lexical[c.id]>0 or sum(a*b for a,b in zip(c.embedding,vector))>.86))]
    candidates.sort(key=lambda c:(scores[c.id],lexical[c.id]),reverse=True)
    return [{'chunk_id':c.id,'paper_id':c.paper_id,'page':c.page,'section':c.section,
             'bbox':c.bbox,'text':c.content,'score':round(scores[c.id],6),
             'keyword_score':round(lexical[c.id],4)} for c in candidates[:k]]
