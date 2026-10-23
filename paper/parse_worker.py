import json
import os
from pathlib import Path
import sys

if os.name=='posix':
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(768*1024*1024,768*1024*1024))
    resource.setrlimit(resource.RLIMIT_CPU,(35,35))

from .parser import parse_pdf
try:
    result=parse_pdf(sys.argv[1])
except Exception as exc:
    result={'error':str(exc)[:500]}
Path(sys.argv[2]).write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
