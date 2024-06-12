import click
import base64
import zarr
import ujson
from io import BytesIO
from kerchunk.hdf import SingleHdf5ToZarr
from netCDF4 import Dataset
import fsspec
from typing import List, Dict, Any


def process_granule(url: str, data: bytes) -> Dict[str, Any]:
    """Generate references from a single granule. Data is loaded directly
    into memory first."""
    refs = SingleHdf5ToZarr(BytesIO(data), url).translate()
    nc = Dataset('t', memory=data)
    attrs = ujson.loads(refs["refs"][".zattrs"])
    # Inline coordinate arrays directly using delta encoding, this way
    # we don't need to read them directly from netCDF in the future
    for c in ["x", "y"]:
        val = nc[c][:]
        z = zarr.array(val, compressor=zarr.Blosc(cname='zstd', clevel=5),
                       filters=[zarr.Delta(val.dtype)])
        refs["refs"][f"{c}/.zarray"] = z.store[".zarray"].decode()
        refs["refs"][f"{c}/0"] = 'base64:' + base64.b64encode(z.store["0"]).decode()
    # Kerchunk can't handle these structs properly so these are converted to
    # global attributes
    for variable in ["mapping", "img_pair_info"]:
        obj = nc[variable]
        for attr in obj.ncattrs():
            val = getattr(obj, attr)
            if hasattr(val, "item"):
                val = val.item()
            attrs[attr] = val
        del refs["refs"][variable+'/.zarray']
    refs["refs"][".zattrs"] = ujson.dumps(attrs)
    return refs

def process_granules(urls: List[str]) -> Dict[str, Dict[str, Any]]:
    """Process a batch of granules. Data are downloaded into memory
    concurrently (async) in order to better saturate available bandwidth."""
    proc = process_granule
    s3 = fsspec.filesystem("s3", anon=True)
    data = s3.cat(urls, on_error='omit')
    result = {url: proc(url, d) for url, d in data.items()}
    return result

def parse_catalogs_list_file(url: str, **storage_options) -> List[str]:
    """Get list of urls for geojson catalogs from master file"""
    with fsspec.open(url, mode='r', **storage_options) as f:
        catalogs = f.readlines()
    return catalogs
    
def granule_urls_from_feautures(features: List[Dict[str, Any]]) -> List[str]:
    """Build granule URLs from GeoJSON features"""
    urls = []
    for feature in features:
        p = feature['properties']
        url = '/'.join(['s3://its-live-data', p['directory'], p['filename']])
        urls.append(url)
    return urls

@click.command()
@click.argument("catalog_file")
@click.argument("start_index", type=int)
@click.argument("out_dir")
@click.option("--batch", type=int, default=1)
@click.option("--batch_size", type=int, default=100)
@click.option("--compress", is_flag=True, default=True)
def process_all_granules(
    catalog_file: str,
    start_index: int,
    out_dir: str,
    batch: int = 1,
    batch_size: int = 100,
    compress: bool = True
):
    """Generate kerchunk references JSON for granules in batches."""
    catalogs = parse_catalogs_list_file(catalog_file, anon=True)
    with fsspec.open(catalogs[start_index]) as f:
        features = ujson.load(f)['features']
    urls = granule_urls_from_feautures(features)
    result = {}
    granules = []
    blosc = zarr.Blosc('zstd')
    for i, url in enumerate(urls):
        granules.append(url)
        if len(granules) == batch_size or i == len(urls) - 1:
            print(f"Processing Batch {batch}/{len(urls) // batch_size}")
            result = process_granules(granules)
            granules = []
            ext = "json"
            if compress:
                ext += ".zstd"
            with fsspec.open(f"{out_dir}/{start_index}.{batch-1}.{ext}", "wb") as f:
                data = ujson.dumps(result).encode()
                if compress:
                    data = blosc.encode(data)
                f.write(data)
            batch += 1
            
if __name__ == '__main__':
    process_all_granules()
