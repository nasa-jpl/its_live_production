import json
import requests
import pyproj
import numpy as np
import os
import logging

from grid import Bounds


def transform_coord(proj1, proj2, lon, lat):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+proj1)
    proj2 = pyproj.Proj("+init=EPSG:"+proj2)
    # Convert coordinates
    return pyproj.transform(proj1, proj2, lon, lat)

def get_granule_urls(params):
    base_url = 'https://nsidc.org/apps/itslive-search/velocities/urls'
    # base_url = 'https://staging.nsidc.org/apps/itslive-search/velocities/urls'
    # Allow for longer query time from searchAPI: 10 minutes
    resp = requests.get(base_url, params=params, verify=False, timeout=500)
    return resp.json()

def get_granule_urls_streamed(params):
    """
    Use streamed retrieval of the response from URL request.
    """
    token = ']['
    base_url = 'https://nsidc.org/apps/itslive-search/velocities/urls'

    # Format request URL:
    url = f'{base_url}?'
    for each_key, each_value in params.items():
        url += f'{each_key}={each_value}&'

    # Add requested granules version (TODO: should be configurable on startup?)
    url += 'version=2'

    # Get rid of all single quotes if any in URL
    url = url.replace("'", "")
    logging.info(f'Submitting searchAPI request with url={url}')
    resp = requests.get(url, stream=True, timeout=300)

    # Save response to local file:
    local_path = 'searchAPI_urls.json'

    logging.info(f'Saving searchAPI response to {local_path}')
    with open(local_path, 'a') as fh:
        for chunk in resp.iter_content(10240, decode_unicode=True):
            _ = fh.write(chunk)

    # Read data from local file
    data = ''
    with open(local_path) as fh:
        data = fh.readline()

    # if multiple json strings are returned,  then possible to see '][' within
    # the string, replace it by ','
    if token in data:
        logging.info('Got multiple json variables within the same string (len(data)={len(data)})')
        data = data.replace(token, ',')

        logging.info('Merged multiple json variables into one list (len(data)={len(data)})')

    data = json.loads(data)

    # Remove local file
    os.unlink(local_path)

    return data

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

#
# Author: Mark Fahnestock, Masha Liukis
#
def add_five_points_to_polygon_side(polygon):
    """
    Define 5 points per each polygon side. This is done before re-projecting
    polygon to longitude/latitude coordinates.
    This function assumes rectangular polygon where min/max x/y define all
    4 polygon vertices.

    polygon: list of lists
        List of polygon vertices.
    """
    fracs = [0.25, 0.5, 0.75]
    polylist = [] # closed ring of polygon points

    # Determine min/max x/y values for the polygon
    x = Bounds([each[0] for each in polygon])
    y = Bounds([each[1] for each in polygon])

    polylist.append((x.min, y.min))
    dx = x.max - x.min
    dy = y.min - y.min
    for frac in fracs:
        polylist.append((x.min + frac * dx, y.min + frac * dy))

    polylist.append((x.max, y.min))
    dx = x.max - x.max
    dy = y.max - y.min
    for frac in fracs:
        polylist.append((x.max + frac * dx, y.min + frac * dy))

    polylist.append((x.max, y.max))
    dx = x.min - x.max
    dy = y.max - y.max
    for frac in fracs:
        polylist.append((x.max + frac * dx, y.max + frac * dy))

    polylist.append((x.min, y.max))
    dx = x.min - x.min
    dy = y.min - y.max
    for frac in fracs:
        polylist.append((x.min + frac * dx, y.max + frac * dy))

    polylist.append((x.min, y.min))

    return polylist
