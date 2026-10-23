import os
from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import create_engine, String, Text, Integer, JSON, ForeignKey, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from pgvector.sqlalchemy import Vector

DIM=384
def uid(): return uuid4().hex
def now(): return datetime.now(timezone.utc).isoformat()
class Base(DeclarativeBase): pass

class Paper(Base):
    __tablename__='papers'
    id: Mapped[str]=mapped_column(String(32),primary_key=True,default=uid)
    sha256: Mapped[str]=mapped_column(String(64),unique=True)
    title: Mapped[str]=mapped_column(String(240))
    pages: Mapped[int]=mapped_column(Integer)
    created: Mapped[str]=mapped_column(String(40),default=now)
    warnings: Mapped[list]=mapped_column(JSON,default=list)
    embedding_model: Mapped[str]=mapped_column(String(120))

class Chunk(Base):
    __tablename__='chunks'
    id: Mapped[str]=mapped_column(String(32),primary_key=True,default=uid)
    paper_id: Mapped[str]=mapped_column(ForeignKey('papers.id'),index=True)
    page: Mapped[int]=mapped_column(Integer)
    ordinal: Mapped[int]=mapped_column(Integer)
    section: Mapped[str]=mapped_column(String(200))
    bbox: Mapped[list]=mapped_column(JSON)
    content: Mapped[str]=mapped_column(Text)
    embedding: Mapped[list]=mapped_column(Vector(DIM).with_variant(JSON,'sqlite'))

class QueryLog(Base):
    __tablename__='queries'
    id: Mapped[str]=mapped_column(String(32),primary_key=True,default=uid)
    created: Mapped[str]=mapped_column(String(40),default=now)
    question: Mapped[str]=mapped_column(Text)
    paper_ids: Mapped[list]=mapped_column(JSON)
    mode: Mapped[str]=mapped_column(String(30))
    duration_ms: Mapped[int]=mapped_column(Integer)
    result: Mapped[dict]=mapped_column(JSON)

def database(url=None):
    url=url or os.getenv('DATABASE_URL','sqlite:///./data/papers.db')
    engine=create_engine(url,connect_args={'check_same_thread':False} if url.startswith('sqlite') else {})
    if engine.dialect.name=='postgresql':
        with engine.begin() as conn: conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    Base.metadata.create_all(engine)
    return sessionmaker(engine,expire_on_commit=False)

def record(obj):
    return {c.name:getattr(obj,c.name) for c in obj.__table__.columns if c.name!='embedding'}
