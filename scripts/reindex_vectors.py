"""Recompute vectors without changing chunk IDs, page mappings or saved citations."""
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from paper.db import database,Chunk,Paper
from paper.retrieval import embed,embed_mode
from sqlalchemy import select

def main():
    from server import env
    os.environ.update(env())
    factory=database()
    with factory.begin() as s:
        chunks=list(s.scalars(select(Chunk)))
        vectors=embed([c.section+' '+c.content for c in chunks])
        for c,vector in zip(chunks,vectors):c.embedding=vector
        for p in s.scalars(select(Paper)):p.embedding_model=embed_mode()
    print('Reindexed',len(chunks),'existing chunks; IDs unchanged')

if __name__=='__main__':main()
