"""Reversible local dataset removal; source recordings are never targeted."""
import json
from pathlib import Path
import re
import threading

class GenerationTrash:
    def __init__(self,root):
        self.root=Path(root).resolve();self.lock=threading.RLock()

    def _path(self,identifier,deleted=False):
        if not isinstance(identifier,str) or not re.fullmatch(r'[a-f0-9]{32}',identifier):
            raise ValueError('Invalid dataset identifier')
        parent=self.root/'.trash' if deleted else self.root
        if parent.is_symlink():raise ValueError('Invalid dataset storage')
        folder=parent/identifier
        if folder.is_symlink() or not folder.is_dir() or not (folder/'job.json').is_file():
            raise ValueError('Dataset not found')
        return folder

    def delete(self,identifier):
        with self.lock:
            folder=self._path(identifier);job=json.loads((folder/'job.json').read_text())
            if job.get('status') in ('queued','running'):
                raise ValueError('Finish or cancel the active dataset build before deleting it')
            parent=self.root/'.trash'
            if parent.is_symlink():raise ValueError('Invalid dataset storage')
            parent.mkdir(exist_ok=True);destination=parent/identifier
            if destination.exists():raise ValueError('Dataset already exists in Trash')
            folder.rename(destination)
            return {'id':identifier,'name':job['name'],'deleted':True,'recoverable':True}

    def list(self):
        result=[]
        for path in sorted((self.root/'.trash').glob('*/job.json')):
            try:
                self._path(path.parent.name,True)
                job=json.loads(path.read_text())
                result.append({'id':path.parent.name,'name':job['name'],'episodes':job.get('progress',{}).get('accepted',0)})
            except (ValueError,OSError,KeyError):
                continue
        return result

    def restore(self,identifier):
        with self.lock:
            folder=self._path(identifier,True);destination=self.root/identifier
            if destination.exists():raise ValueError('Dataset already exists')
            job=json.loads((folder/'job.json').read_text());folder.rename(destination)
            return {'id':identifier,'name':job['name'],'restored':True}
