import numpy as np
import os


#
# Author: Mark Fahnestock
#
def point_to_prefix(dir_path: str, lat: float, lon: float) -> str:
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

    dirstring = os.path.join(dir_path, f'{NShemi_str}{outlat:02d}{EWhemi_str}{outlon:03d}')
    return dirstring

