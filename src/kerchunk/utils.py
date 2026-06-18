import pyarrow as pa
import ujson
import zarr
from typing import Sequence, Any, Dict, Tuple
from collections.abc import MutableMapping

class MultiRefs(MutableMapping):
    def __init__(self, mrefs: Dict[str, Dict[str, Any]], is_zarr=False):
        self.mrefs = mrefs
        self.is_zarr = is_zarr

    def _trans_key(self, key: str) -> Tuple[str, str]:
        kdata = key.split("/")
        pref = kdata[0]
        rkey = "/".join(kdata[1:])
        return pref, rkey
        
    def __getitem__(self, key: str):
        if self.is_zarr and key == ".zgroup":
            return '{"zarr_format":2}'
        elif self.is_zarr and key.startswith("."):
            return self.mrefs[key]
        pref, rkey = self._trans_key(key)
        return self.mrefs[pref]["refs"][rkey]
        
    def __setitem__(self, key: str, value):
        if self.is_zarr and key.startswith("."):
            self.mrefs[key] = value
        else:
            pref, rkey = self._trans_key(key)
            self.mrefs[pref]["refs"][rkey] = value
        
    def __delitem__(self, key: str):
        pref, rkey = self._trans_key(key)
        del self.mrefs[pref]["refs"][rkey]
    
    def __iter__(self):
        for pref, refs in self.mrefs.items():
            if not hasattr(refs, "__len__"):
                yield refs
            for rkey in refs["refs"]:
                yield pref + "/" + rkey
        if self.is_zarr:
            yield ".zgroup"
    
    def __len__(self):
        size = sum([len(refs) for refs in self.mrefs.values()])
        if self.is_zarr:
            size += 1
        return size
        
class ConcLRUStoreCache(zarr.storage.LRUStoreCache):
    def getitems(self, keys: Sequence[str], contexts) -> Dict[str, Any]:
        cached_keys = set(keys).intersection(self._values_cache.keys())
        uncached_keys = set(keys).difference(cached_keys)
        items = super().getitems(cached_keys, contexts=contexts)
        if uncached_keys:
            uncached_items = self.base_store.getitems(
                list(uncached_keys),
                contexts=contexts
            )
            for key, value in uncached_items.items():
                if key not in self._values_cache:
                    self._cache_value(key, value)
            self.hits += len(uncached_items)
            self.misses += len(keys) - len(uncached_items)
            items.update(uncached_items)
        return items
    
    @property
    def base_store(self):
        return self._store
    
    @base_store.setter
    def base_store(self, store: MutableMapping):
        self._store = store


def expand_struct_column(table: pa.Table, column: str) -> pa.Table:
    keys = column.split(".")
    for key in keys:
        table = pa.Table.from_struct_array(table[key])
    return table
    
    
def stac_to_kerchunk(
    cube_stac: Dict[str, Any],
    dim_stac: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    out = dict(version=1, refs={})
    refs = out["refs"]
    refs[".zgroup"] = '{"zarr_format": 2}'
    for stac in [cube_stac, dim_stac]:
        for v, r in stac.items():
            refs[f"{v}/.zarray"] = r["kerchunk:zarray"]
            refs[f"{v}/.zattrs"] = r["kerchunk:zattrs"]
            kv = r["kerchunk:value"]
            for i, key in enumerate(kv["key"]):
                ckey = f"{v}/{key}"
                if kv["path"][i] is not None:
                    refs[ckey] = [
                        "s3://"+kv["path"][i],
                        kv["offset"][i],
                        kv["size"][i]
                    ]
                else:
                    refs[ckey] = kv["raw"][i]
    return out
