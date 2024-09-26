import base64
import functools
import copy
import os
import warnings
from io import BytesIO
from typing import Any, Dict

import click
import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
import json
import xstac
import zarr
from kerchunk.hdf import SingleHdf5ToZarr
from netCDF4 import Dataset


STAC_TEMPLATE = {
    "type": "Feature",
    "stac_version": "1.0.0",
    "properties": {},
    "links": [],
    "assets": {},
    "stac_extensions": ["https://stac-extensions.github.io/datacube/v2.2.0/schema.json"]
}


def refs_from_granule(url: str, s3_url: str = None) -> Dict[str, Any]:
    """Generate references from a single granule. Data is loaded directly
    into memory first."""
    if s3_url is None:
        s3_url = url
    data = fsspec.open(url).open().read()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refs = SingleHdf5ToZarr(BytesIO(data), s3_url).translate()
    nc = Dataset("t", memory=data)
    attrs = json.loads(refs["refs"][".zattrs"])
    # Inline coordinate arrays directly using delta encoding, this way
    # we don"t need to read them directly from netCDF in the future
    for c in ["x", "y"]:
        val = nc[c][:]
        z = zarr.array(val, compressor=zarr.Blosc(cname="zstd", clevel=5),
                       filters=[zarr.Delta(val.dtype)])
        refs["refs"][f"{c}/.zarray"] = z.store[".zarray"].decode()
        refs["refs"][f"{c}/0"] = "base64:" + base64.b64encode(z.store["0"]).decode()
    # Kerchunk can"t handle these structs properly so these are converted to
    # global attributes
    for variable in ["mapping", "img_pair_info"]:
        obj = nc[variable]
        for attr in obj.ncattrs():
            val = getattr(obj, attr)
            if hasattr(val, "item"):
                val = val.item()
            attrs[attr] = val
        del refs["refs"][variable+"/.zarray"]
    refs["refs"][".zattrs"] = json.dumps(attrs)
    return refs


class DuckVariable:
    """Simple mock xarray variable"""
    def __init__(self, name, attrs, chunks, dims, shape):
        self.name = name
        self.attrs = attrs
        self.chunks = chunks
        self.dims = dims
        self.shape = shape
        
    @property
    def chunksize(self):
        return self.chunks
        
    @property
    def data(self):
        return self

class DuckDataset:
    """Mock lightweight dataset object from kerchunk"""
    def __init__(self, refs: Dict[str, Any]):
        r = refs["refs"]
        vnames = [k.split("/")[0] for k in r.keys() if "/" in k]
        self.variables = {}
        self.dims = ["y", "x"]
        self.coords = self.dims
        for v in vnames:
            attrs = json.loads(r.get(f"{v}/.zattrs", "{}"))
            zarray = json.loads(r[f"{v}/.zarray"])
            shape = zarray["shape"]
            chunks = zarray["chunks"]
            dims = attrs.get("_ARRAY_DIMENSIONS")
            if dims is None:
              continue
            dims = dict(zip(dims, zarray["shape"]))
            self.variables[v] = DuckVariable(v, attrs, chunks, dims, shape)
            
    def __getitem__(self, attr: str):
        return self.variables[attr]
        
    def __getattr__(self, attr: str):
        if attr in self.variables:
            return self.variables[attr]
        return super().__getattr__(attr)


def coord_from_ref(ref: Dict[str, Any], dim: str):
    data = ref["refs"][f"{dim}/0"][6:]
    blosc = zarr.Blosc("zstd")
    delta = zarr.Delta(dtype=np.float64)
    decoders = [base64.b64decode, blosc.decode, delta.decode]
    for decode in decoders:
        data = decode(data)
    return data
    
        
def geom_from_refs(refs: Dict[str, Any]):
    def ensure_scalar(g):
        if isinstance(g, tuple):
            return g[0]
        return g
    crs = []
    n = len(refs)
    bbox = dict(xmin=np.empty(n), ymin=np.empty(n),
                xmax=np.empty(n), ymax=np.empty(n))
    for i, ref in enumerate(refs.values()):
        attrs = json.loads(ref["refs"][".zattrs"])
        crs.append(attrs["crs_wkt"])
        coords = {dim: coord_from_ref(ref, dim) for dim in ["x", "y"]}
        for stat in ["min", "max"]:
            for dim, coord in coords.items():
                bbox[dim+stat][i] = getattr(coord, stat)()
    geoms = pd.Series(shapely.box(**bbox), index=list(refs.keys()))
    geoms = pd.concat([gpd.GeoSeries(geom, crs=ensure_scalar(g)).to_crs(epsg=4326)
                       for g, geom in geoms.groupby(crs)])
    return geoms
        

def refs_to_stac_item(ref: Dict[str, Any]) -> Dict[str, Any]:
    """Convert kerchunk references JSON to STAC item JSON"""
    @functools.lru_cache(maxsize=None)
    def _crs_json_from_epsg(epsg):
        return pyproj.CRS.from_epsg(epsg).to_json_dict()
    uri = ref["refs"]["v/0.0"][0]
    if not uri.startswith("s3://"):
        uri = "s3://" + uri
    name = os.path.basename(uri)
    poly = geom_from_refs({name: ref}).values[0]
    attrs = json.loads(ref["refs"][".zattrs"])
    epsg = attrs["spatial_epsg"]
    time = attrs["date_center"]
    # dateutil can"t parse ISO times ending in "."
    if time.endswith("."):
        time += "0"
    template = copy.deepcopy(STAC_TEMPLATE)
    template["properties"]["datetime"] = time
    template["id"] = name
    template["assets"] = {name: {"href": uri}}
    reference_system = _crs_json_from_epsg(epsg)

    x = coord_from_ref(ref, "x")
    y = coord_from_ref(ref, "y")
    item = xstac.xarray_to_stac(DuckDataset(ref),
                                template,
                                temporal_dimension=False,
                                x_dimension="x",
                                y_dimension="y",
                                validate=False,
                                reference_system=reference_system,
                                x_step=x[1]-x[0],
                                y_step=y[1]-y[0],
                                y_extent=[y.min(), y.max()],
                                x_extent=[x.min(), x.max()])
    
    item.geometry = json.loads(shapely.to_geojson(poly))
    item.bbox = list(poly.bounds)
    return item.to_dict()

@click.command()
@click.argument("path")
@click.option("--s3-path", type=str, default=None)
def make_granule_stac_kerchunk(path: str, s3_path: str):
    fname = os.path.basename(path)
    refs = refs_from_granule(path, s3_path)
    stac = refs_to_stac_item(refs)
    with open(fname + ".refs.json", "w") as f:
        json.dump(refs, f, indent=4)
    with open(fname + ".stac.json", "w") as f:
        json.dump(stac, f, indent=4)
        
if __name__ == '__main__':
    make_granule_stac_kerchunk()
