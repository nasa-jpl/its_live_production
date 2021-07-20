import requests
import pyproj
import numpy as np
import os


def transform_coord(proj1, proj2, lon, lat):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+proj1)
    proj2 = pyproj.Proj("+init=EPSG:"+proj2)
    # Convert coordinates
    return pyproj.transform(proj1, proj2, lon, lat)

def get_granule_urls(params):
    # base_url = 'https://nsidc.org/apps/itslive-search/velocities/urls'
    base_url = 'https://staging.nsidc.org/apps/itslive-search/velocities/urls'
    resp = requests.get(base_url, params=params, verify=False)
    return resp.json()

#
# Author: Mark Fahnestock
#
def point_to_prefix(lat: float, lon: float, dir_path: str = None) -> str:
    """
    Returns a string (for example, N78W124) for directory name based on
    granule centerpoint lat,lon
    """
    NShemi_str = 'N' if lat >= 0.0 else 'S'
    EWhemi_str = 'E' if lon >= 0.0 else 'W'

    outlat = int(10*np.trunc(np.abs(lat/10.0)))
    if outlat == 90: # if you are exactly at a pole, put in lat = 80 bin
        outlat = 80

    outlon = int(10*np.trunc(np.abs(lon/10.0)))

    if outlon >= 180: # if you are at the dateline, back off to the 170 bin
        outlon = 170

    dirstring = f'{NShemi_str}{outlat:02d}{EWhemi_str}{outlon:03d}'
    if dir_path is not None:
        dirstring = os.path.join(dir_path, dirstring)

    return dirstring
