"""
Granule catalog search utilities for the ITS_LIVE project.
"""
import logging

# STAC catalog related
from itslive import EQ, GTE, search

from itslive_utils import timing_decorator

# Granule catalog to search. 'serverless' (default) queries the geoparquet
# warehouse via duckdb; 'pgstac' queries the STAC API via pystac_client.
# STAC_CATALOG overrides the catalog location (s3:// for serverless,
# https:// for pgstac); None uses the itslive defaults.
SEARCH_TYPE = 'serverless'
STAC_CATALOG = None


@timing_decorator
def serverless_search(
    epsg_code: str,
    start_date: str,
    end_date: str,
    roi: dict,
    percent_valid_pixels: float = 1.0,
):
    """Get list of granules using itslive Python package.

    Returns:
        list(str): Found list of granule URLs.

    For example, this query should return all the granules that intersect
    with the provided polygon and have at least 1% valid pixels in EPSG:epsg_code
    projection between start and end dates.
    """
    logging.info(
        f"Quering catalog: {roi=} {start_date=} {end_date=} {SEARCH_TYPE=} "
        f"{STAC_CATALOG=} {epsg_code=} {percent_valid_pixels=}"
    )

    return search(
        geojson=roi,
        start=start_date,
        end=end_date,
        type=SEARCH_TYPE,
        engine="duckdb",
        base_catalog_href=STAC_CATALOG,
        filters={
            "percent_valid_pixels": GTE(percent_valid_pixels),
            "proj:code": EQ(f"EPSG:{epsg_code}"),
        }
    )
