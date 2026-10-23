"""Download the pinned official E5 ONNX model, validating every file's SHA-256."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
def main():
    manifest=json.loads((ROOT/'model-manifest.json').read_text())
    folder=ROOT/'runtime/e5';folder.mkdir(parents=True,exist_ok=True)
    for name,expected in manifest['files'].items():
        dest=folder/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==expected:continue
        tmp=dest.with_suffix(dest.suffix+'.part')
        if name=='revision.txt':tmp.write_text(manifest['revision'])
        else:
            url=f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{name}"
            with urllib.request.urlopen(url,timeout=90) as r,tmp.open('wb') as f:
                while block:=r.read(1024*1024):f.write(block)
        if hashlib.sha256(tmp.read_bytes()).hexdigest()!=expected:raise RuntimeError('Checksum mismatch: '+name)
        tmp.replace(dest);print('Verified',name,flush=True)
    print('Model ready under runtime/e5')
if __name__=='__main__':main()
