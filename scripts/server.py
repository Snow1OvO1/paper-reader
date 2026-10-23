"""Start a private, user-owned PostgreSQL + pgvector and paper application."""
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'runtime'
PG=RUNTIME/'pgsql/usr/lib/postgresql/16/bin';DATA=RUNTIME/'pgdata'

def env():
    RUNTIME.mkdir(exist_ok=True);config=RUNTIME/'server-env.json'
    if not config.exists():
        fd=os.open(config,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:json.dump({'PGPASSWORD':secrets.token_hex(24)},f)
    values=json.loads(config.read_text())
    result={**os.environ,**values,'LD_LIBRARY_PATH':str(RUNTIME/'pgsql/usr/lib/x86_64-linux-gnu'),
            'DATA_DIR':str(ROOT/'data'),'PAPER_URL':'http://127.0.0.1:18100',
            'OLLAMA_URL':'http://127.0.0.1:11434','LLM_MODEL':'qwen2.5:1.5b-instruct'}
    model=RUNTIME/'e5'
    if (model/'onnx/model_qint8_avx512_vnni.onnx').exists():
        result['EMBEDDING_MODEL']='e5-onnx-int8-v1';result['E5_MODEL_DIR']=str(model)
    result['DATABASE_URL']=f"postgresql+psycopg://paper:{values['PGPASSWORD']}@127.0.0.1:15440/paper"
    return result

def pid():
    try:
        value=int((RUNTIME/'app.pid').read_text())
        if Path(f'/proc/{value}/cwd').resolve()!=ROOT:return None
        if str(ROOT/'.venv/bin/python').encode() not in Path(f'/proc/{value}/cmdline').read_bytes():return None
        return value
    except OSError:return None

def start(e):
    if not DATA.exists():
        pw=RUNTIME/'init-password'
        fd=os.open(pw,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(e['PGPASSWORD'])
        try:
            subprocess.run([str(PG/'initdb'),'-D',str(DATA),'-U','paper','--auth=scram-sha-256',
                '--pwfile='+str(pw),'--no-locale','--encoding=UTF8','-L',str(RUNTIME/'pgsql/usr/share/postgresql/16')],env=e,check=True)
        finally:pw.unlink(missing_ok=True)
    if subprocess.run([str(PG/'pg_ctl'),'-D',str(DATA),'status'],env=e,capture_output=True).returncode:
        subprocess.run([str(PG/'pg_ctl'),'-D',str(DATA),'-l',str(RUNTIME/'postgres.log'),
             '-o',f'-h 127.0.0.1 -p 15440 -k {RUNTIME}','start'],env=e,check=True)
    import psycopg
    from psycopg import sql
    with psycopg.connect(host='127.0.0.1',port=15440,user='paper',password=e['PGPASSWORD'],dbname='postgres',autocommit=True) as conn:
        for name in ['paper','paper_test']:
            if not conn.execute('SELECT 1 FROM pg_database WHERE datname=%s',(name,)).fetchone():
                conn.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
    if not pid():
        with (RUNTIME/'app.log').open('ab') as log:
            p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-m','uvicorn','paper.api:create_app','--factory',
                '--host','127.0.0.1','--port','18100'],cwd=ROOT,env=e,stdin=subprocess.DEVNULL,
                stdout=log,stderr=log,start_new_session=True)
        (RUNTIME/'app.pid').write_text(str(p.pid))
    for _ in range(60):
        try:
            with urllib.request.urlopen('http://127.0.0.1:18100/health',timeout=1) as r:
                if r.status==200:print('PaperTrail ready on 127.0.0.1:18100');return
        except OSError:time.sleep(.5)
    raise RuntimeError('Startup failed; inspect runtime/app.log')

def main():
    e=env();action=sys.argv[1] if len(sys.argv)>1 else 'status'
    if action=='start':start(e)
    elif action=='status':print('application PID:',pid() or 'stopped')
    elif action=='stop':
        if pid():os.kill(pid(),signal.SIGTERM);time.sleep(2)
        subprocess.run([str(PG/'pg_ctl'),'-D',str(DATA),'stop','-m','fast'],env=e,check=True)
    elif action=='test':
        e['TEST_DATABASE_URL']=e['DATABASE_URL']+'_test'
        subprocess.run([sys.executable,'-m','pytest','-q','--junitxml=reports/server-tests.xml'],cwd=ROOT,env=e,check=True)
    elif action=='evaluate':subprocess.run([sys.executable,'scripts/evaluate.py'],cwd=ROOT,env=e,check=True)
    else:raise SystemExit('Use start|stop|status|test|evaluate')
if __name__=='__main__':main()
