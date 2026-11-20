"""Small local JSON key-value persistence."""
import json,os
from pathlib import Path
class Store:
    def __init__(self,path=None):
        self.path=Path(path or os.environ.get("DATA_PATH","data/app.json")); self.path.parent.mkdir(parents=True,exist_ok=True)
        if not self.path.exists(): self.path.write_text("{}")
    def _read(self): return json.loads(self.path.read_text() or "{}")
    def put(self,key,value):
        data=self._read(); data[key]=value; self.path.write_text(json.dumps(data,indent=2))
    def get(self,key,default=None): return self._read().get(key,default)
    def all(self): return self._read()
