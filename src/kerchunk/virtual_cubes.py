import fsspec
import ujson
import shapely
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import xarray as xr
import numpy as np
import hashlib
import dask
import dask.array as da
import pandas as pd
import geopandas as gpd
from dask_geopandas.geohash import _geohash
from zarr.storage import FSStore
from numcodecs import Zlib, Shuffle, Blosc, Delta
from itertools import groupby
from typing import Dict, Tuple, List, Any
from utils import (
    ConcLRUStoreCache,
    stac_to_kerchunk,
    expand_struct_column,
    MultiRefs
)

class VirtualDataset:
    # Info for decoding chunks, maybe we should be reading these
    # individually from each reference instead?
    shuf = Shuffle(elementsize=2)
    zlib = Zlib(level=2)
    blosc = Blosc("zstd")
    delta = Delta("<f8")
    fill_values = {"chip_size_height": 0, "chip_size_width": 0, "interp_mask": 0,
                   "v": -32767, "v_error": -32767, "vx": -32767, "vy": -32767,
                   "M11": -32767, "M12": -32767}

    def __init__(
        self,
        catalog_path: str,
        cache_size: int = None,
        **storage_options
    ):
        self.cache_size = cache_size
        proto, catalog_path = fsspec.core.split_protocol(catalog_path)
        protocol = proto or storage_options.pop("protocol", "local")
        self.fs = fsspec.filesystem(protocol, **storage_options)
        rg_path = f"{catalog_path}/.row_groups.parquet"
        self.pds = pq.ParquetDataset(catalog_path, filesystem=self.fs)
        self.rg_geoms = gpd.read_parquet(
            f"{protocol}://{rg_path}",
            storage_options=storage_options
        ).set_index('index')

    def build_cubes(
        self,
        selection: Tuple[float, float, float, float] | shapely.Polygon | str,
        time_range=None,
        split_by=None,
        **filters
    ) -> List[xr.Dataset]:
        """Generate virtual xarray datasets from a catalog search.
        
        Note that as data are paritioned by time and not space, it is possible
        for a single search to return results from multiple projections. You
        may use the split_by / **filters arguments to ensure all layers in a cube
        are on the same projection:
        
        # Only return results with EPSG:3031
        cube = vds.build_cubes(selection, time_range=time_range, spatial_epsg=3031)[0]
        
        # Return separate cubes by projection
        cubes = vds.build_cubes(selection, time_range=time_range, split_by="spatial_epsg")
        
        Parameters
        ----------
        selection : str, tuple(float), Polygon
            Spatial domain to search, can be a list of bounding box coordinates,
            a GeoJSON string, or a shapely Polygon object.
        time_range : tuple(datetimelike), optional
            Start and end datetimes for the search, if omitted then the entire record
            is used.
        split_by : str, list(str), or None, optional
            Default option (None) returns just one dataset for the entire search,
            but you may opt to split by a single or list of global attribute values
            in the granules (eg, spatial_epsg, flight_direction_img1). Default is
            split by projection (spatial_epsg).
        **filters : dict
            Keyword arguments where keys, like for split_by, are global attributes
            in the granule metadata, values represent a single
        """
        if isinstance(selection, tuple):
            selection = shapely.geometry.box(*selection)
        elif isinstance(selection, str):
            selection = shapely.from_geojson(selection)
        refs, times, attrs = self._load_refs(
            selection,
            time_range=time_range,
            split_by=split_by,
            **filters
        )
        return self._cubes_from_refs(refs, times=times, attrs=attrs, selection=selection)
        
    def _load_refs(
        self,
        selection: shapely.Polygon,
        time_range=None,
        split_by: str | List[str] = None,
        geohash_precision: int = 2,
        **filters
    ) -> List[Dict[str, Dict[str, Any]]]:
        # Find which row groups in geoparquet catalog contain data of interest
        intersection = self.rg_geoms[self.rg_geoms.intersects(selection)]
        if time_range is not None:
            time_range = pd.date_range(*time_range, tz="UTC", unit="ns")
            years = time_range.year.unique()
        else:
            years = intersection.year.unique()

        pred_filters = []
        if time_range is not None:
            time_range = pd.to_datetime(time_range, unit="ns", utc=True)
            pred_filters.append((pc.field("datetime") >= time_range[0])
                               & (pc.field("datetime") <= time_range[-1]))

        # Iterate over selected fragments, selectinlg only row groups that
        # intersect the bounding box
        cols = ['id', 'datetime', 'attrs', 'cube:variables', 'geometry',
                'cube:dimensions', 'kerchunk']
        groups = intersection.groupby("year").groups
        tables = []
        for year in years:
            idx = groups[year].values.tolist()
            iyear = year - self.rg_geoms.year.min()
            frag = self.pds.fragments[iyear].subset(row_group_ids=idx)
            for filt in pred_filters:
                frag = frag.subset(filter=filt)
            table = frag.to_table(columns=cols)
            for filt in pred_filters:
                table = table.filter(filt)
            frag_geoms = gpd.GeoSeries.from_wkb(table["geometry"])
            spatial_subset = frag_geoms.intersects(selection).values
            tables.append(table.filter(spatial_subset))
        table = pa.concat_tables(tables)
        
        # Additional splitting / filtering
        if np.isscalar(split_by):
            split_by = [split_by]
        elif split_by is None:
            split_by = ["spatial_epsg"]
        
        if split_by or filters:
            # Generate metadata DataFrame for splitting or filtering output
            t1 = expand_struct_column(table, "attrs")
            t2 = table.select(["id", "datetime"])
            meta = gpd.GeoDataFrame(
                pd.concat(
                    [t.to_pandas(types_mapper=pd.ArrowDtype) for t in [t1, t2]],
                    axis=1
                ),
                geometry=gpd.GeoSeries.from_wkb(table["geometry"])
            )
            region = meta.geometry.copy()
            unary_union = region.unary_union
            if not hasattr(unary_union, "geoms"):
                unary_union = shapely.MultiPolygon([unary_union])
            polys = unary_union.geoms
            inds = [meta.intersects(p) for p in polys]
            for i, ind in enumerate(inds):
                region[ind] = polys[i]
            meta["region"] = region
        
        if filters:
            for attr, value in filters.items():
                if callable(value):
                    filtered = meta[attr].apply(value)
                elif np.isscalar(value):
                    filtered = meta[attr] == value
                else:
                    filtered = meta[attr].isin(value)
                table = table.filter(np.asarray(filtered.values))
                meta = meta[filtered]
        
        if split_by:
            tables = {
                group: table.filter(meta.index.isin(ind))
                for group, ind in meta.groupby(split_by).groups.items()
            }
        else:
            tables = {(): table}
        
        # Finally extract kerchunk references from selection.
        # Output is mapping of filename => references_dict
        refs = []
        times = []
        attrs = []
        for k, table in tables.items():
            keys = table["id"].to_pylist()
            refs_json = ",".join(table["kerchunk"].to_pylist())
            refs_dict = ujson.loads(f"[{refs_json}]")
            r = dict(zip(keys, refs_dict))
            t = dict(zip(keys, table["datetime"].to_numpy()))
            if len(split_by) == 1:
                k = [k]
            refs.append(r)
            times.append(list(t.values()))
            attrs.append(dict(zip(split_by, k)))
                
        return refs, times, attrs

    def _cubes_from_refs(
        self,
        refs: List[Dict[str, Dict[str, Any]]],
        times=None,
        attrs: List[Dict[str, Any]] = None,
        selection: shapely.Polygon = None
    ) -> List[xr.Dataset]:
        combined_refs = {k: v for ref in refs for k, v in ref.items()}
        mrefs = MultiRefs(combined_refs)
        fs = fsspec.filesystem("reference", fo=mrefs, remote_protocol="s3", remote_options=dict(anon=True))
        store = FSStore("reference://", fs=fs)
        self.store = ConcLRUStoreCache(store, max_size=self.cache_size)
        return [self._cube_from_ref(ref, time, attr, selection) for ref, time, attr in zip(refs, times, attrs)]
    
    def _cube_from_ref(
        self,
        ref: Dict[str, Dict[str, Any]],
        time=None,
        attr: Dict[str, Any] = None,
        selection: shapely.Polygon
    ) -> List[xr.Dataset]:
        mrefs = MultiRefs(ref)
        grouped, out_grid, grids = self._gen_grid(mrefs)
        ds = xr.Dataset(coords=dict(mid_date=time, **out_grid), attrs=attr)
        for field, fill_value in self.fill_values.items():
            groups = self._filter_groups(grouped, field)
            zarray = ujson.loads(mrefs[f"{list(mrefs.mrefs.keys())[0]}/{field}/.zarray"])
            dtype = zarray["dtype"]
            array = self._process_chunks(groups, out_grid, grids, fill_value, dtype)
            ds[field] = ("mid_date", "y", "x"), array
            ds[field] = ds[field].where(ds[field] != fill_value)
        if selection:
            # TODO This would be more efficient if selection was done at task level
            # rather than at top level, but this is much simpler logic
            crs = ds.attrs["spatial_epsg"]
            bnds = gpd.GeoSeries([selection], crs=4326)
            xmin, ymin, xmax, ymax = bnds.to_crs(crs).bounds.values.ravel()
            ix = (ds.x.values >= xmin) & (ds.x.values <= xmax)
            iy = (ds.y.values >= ymin) & (ds.y.values <= ymax)
            ds = ds.isel(x=ix, y=iy)
        return ds

    def _gen_grid(
        self,
        refs: MultiRefs
    ) -> Tuple[
        Dict[str, List[str]],
        Dict[str, np.typing.NDArray[float]],
        Dict[str, np.typing.NDArray[float]]
        ]:
        coord_keys = [k for k in refs if "/x/0" in k or "/y/0" in k]
        coords = self.store.getitems(coord_keys, contexts=None)
        keyfunc = lambda k: k[0].split("/")[0]
        grouped = {
            k: [v[0] for v in j]
            for k, j in groupby(sorted(refs.items()), key=keyfunc)
        }
        isneg = {"x": None, "y": None}
        grids = {}
        out_grid = {}
        ranges = {"x": [None, None], "y": [None, None]}
        for fname, fkeys in grouped.items():
            for dim in ["x", "y"]:
                k = fname + f"/{dim}/0"
                fkeys.remove(k)
                grid = self.delta.decode(self.blosc.decode(coords[k]))
                gmin, gmax = ranges[dim]
                if gmin is None:
                    gmin = grid.min()
                if gmax is None:
                    gmax = grid.max()
                if isneg[dim] is None:
                    isneg[dim] = grid[0] > grid[-1]
                ranges[dim] = [min(grid.min(), gmin), max(grid.max(), gmax)]
                grids[k] = grid
        for dim in ["x", "y"]:
            ranges[dim][1] += 120
            out_grid[dim] = np.arange(*ranges[dim], 120)
            if isneg[dim]:
                out_grid[dim] = out_grid[dim][::-1]
        return grouped, out_grid, grids

    def _process_chunks(self,
        grouped: Dict[str, List[str]],
        out_grid: Dict[str, np.typing.NDArray[float]],
        grids: Dict[str, Any],
        fill_value: int,
        dtype: np.typing.DTypeLike
    ) -> da.array:
        ckeys = []
        for fkeys in grouped.values():
            ckeys.extend(fkeys)
        cdatas = self._fetch_chunks(ckeys)
        out_shape = (out_grid["y"].size, out_grid["x"].size)
        results = []
        for fname, fkeys in grouped.items():
            i = {}
            sizes = []
            for dim in ["y", "x"]:
                grid = grids[fname+f"/{dim}/0"]
                sizes.append(grid.size)
                start = int(abs(grid[0] - out_grid[dim][0]) // 120)
                i[dim] = slice(start, start+grid.size)
            size = np.prod(sizes)
            chunks = self._select_chunks(cdatas, fkeys)
            iy, ix = i["y"], i["x"]
            result = da.from_delayed(
                self._decompress_chunks(
                    chunks, out_shape, sizes, dtype, fill_value, ix, iy
                ),
                dtype=dtype,
                shape=out_shape
            )
            results.append(result)
        return da.stack(results)

    def _filter_groups(
        self,
        grouped: Dict[str, List[str]],
        field: str
    ) -> Dict[str, List[str]]:
        return {k: [x for x in v if f'/{field}/' in x and '/.z' not in x]
                for k, v in grouped.items()}

    @dask.delayed
    def _fetch_chunks(self, ckeys: List[str]) -> Dict[str, Any]:
        return self.store.getitems(ckeys, contexts=None)

    @dask.delayed
    def _select_chunks(
        self,
        cdatas: Dict[str, Any],
        fkeys: List[str]
    ) -> List[Any]:
        return [cdatas[fkey] for fkey in fkeys]

    @dask.delayed
    def _decompress_chunks(
        self,
        chunks: List[Any],
        shape: Tuple[int, int],
        sizes: Tuple[int, int],
        dtype: np.typing.DTypeLike,
        fill_value: int | float,
        ix: int,
        iy: int
    ) -> np.typing.NDArray:
        decoded = [self.shuf.decode(self.zlib.decode(chunk)) for chunk in chunks]
        data = np.concatenate(decoded).view(dtype)[:np.prod(sizes)].reshape(sizes)
        output = np.full(shape, fill_value, dtype=dtype)
        output[iy, ix] = data
        return output
