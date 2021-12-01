#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Yang Lei, Jet Propulsion Laboratory
# November 2017:
#   Original code.
#
# Masha Liukis, Jet Propulsion Laboratory
# November 2021:
#   Add automatic detection of parametrization files used by calculation.
#   Restructure the code to be able to invoke the code from another script.
#   TODO: need to specify encoding when writing corrected dataset to the file.
#

import argparse
import numpy as np
import shelve
import os
import datetime
import pdb
import xarray as xr
from osgeo import ogr, gdal

class ITSLiveException (Exception):
    """
    Exception class to handle ITS_LIVE special cases.
    """
    def __init__(self, msg):
        super().__init__(msg)


def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser( description='Converting netcdf variables to geotiff')
    parser.add_argument('-i', '--nc', dest='nc', type=str, required=True,
            help = 'input netcdf file path')
    parser.add_argument('-vx', '--vx', dest='VXref', type=str, required=True,
            help = 'input reference velocity x file path')
    parser.add_argument('-vy', '--vy', dest='VYref', type=str, required=True,
            help = 'input reference velocity y file path')
    parser.add_argument('-ssm', '--ssm', dest='SSM', type=str, required=True,
            help = 'input stable surface mask file path')
    parser.add_argument('-o','--out', dest='output', type=str, required=True,
            help = 'Variable to output')
#    parser.add_argument('-e','--epsg', dest='epsg', type=int, default=3413,
#            help = 'EPSG code')
    return parser.parse_args()



def v_error_cal(vx_error, vy_error):
    vx = np.random.normal(0, vx_error, 1000000)
    vy = np.random.normal(0, vy_error, 1000000)
    v = np.sqrt(vx**2 + vy**2)
    return np.std(v)


# Authors: Joe Kennedy, Masha Liukis
def find_jpl_parameter_info(ds: xr.Dataset, ds_filename: str) -> dict:
    driver = ogr.GetDriverByName('ESRI Shapefile')
    parameter_file = ds.attrs['autoRIFT_parameter_file']
    shapes = driver.Open(f"/vsicurl/{parameter_file}", gdal.GA_ReadOnly)

    parameter_info = None
    # centroid = flip_point_coordinates(polygon.Centroid())
    centroid  = ogr.Geometry(ogr.wkbPoint)
    longitude = np.float(ds.img_pair_info.attrs['longitude'])
    latitude  = np.float(ds.img_pair_info.attrs['latitude'])
    centroid.AddPoint(longitude, latitude)

    try:
        for feature in shapes.GetLayer(0):
            if feature.geometry().Contains(centroid):
                parameter_info = {
                    'name': f'{feature["name"]}',
                    'epsg': feature['epsg'],
                    'geogrid': {
                        'dem': f"/vsicurl/{feature['h']}",
                        'ssm': f"/vsicurl/{feature['StableSurfa']}",
                        'dhdx': f"/vsicurl/{feature['dhdx']}",
                        'dhdy': f"/vsicurl/{feature['dhdy']}",
                        'vx': f"/vsicurl/{feature['vx0']}",
                        'vy': f"/vsicurl/{feature['vy0']}",
                        'srx': f"/vsicurl/{feature['vxSearchRan']}",
                        'sry': f"/vsicurl/{feature['vySearchRan']}",
                        'csminx': f"/vsicurl/{feature['xMinChipSiz']}",
                        'csminy': f"/vsicurl/{feature['yMinChipSiz']}",
                        'csmaxx': f"/vsicurl/{feature['xMaxChipSiz']}",
                        'csmaxy': f"/vsicurl/{feature['yMaxChipSiz']}",
                        'sp': f"/vsicurl/{feature['sp']}",
                        'dhdxs': f"/vsicurl/{feature['dhdxs']}",
                        'dhdys': f"/vsicurl/{feature['dhdys']}",
                    },
                    'autorift': {
                        'grid_location': 'window_location.tif',
                        'init_offset': 'window_offset.tif',
                        'search_range': 'window_search_range.tif',
                        'chip_size_min': 'window_chip_size_min.tif',
                        'chip_size_max': 'window_chip_size_max.tif',
                        'offset2vx': 'window_rdr_off2vel_x_vec.tif',
                        'offset2vy': 'window_rdr_off2vel_y_vec.tif',
                        'stable_surface_mask': 'window_stable_surface_mask.tif',
                        'mpflag': 0,
                    }
                }
                break
    except Exception as exc:
        # Debug failure to access feature's geometry
        raise RuntimeError(f'Error accessing {parameter_file} for {ds_filename}: {exc}. Feature keys: {list(feature.keys())}')

    if parameter_info is None:
        raise RuntimeError('Could not determine appropriate DEM for:\n'
                          f'    centroid: {centroid}\n'
                          f'    using: {parameter_file}')

    dem_geotransform = gdal.Info(parameter_info['geogrid']['dem'], format='json')['geoTransform']
    parameter_info['xsize'] = abs(dem_geotransform[1])
    parameter_info['ysize'] = abs(dem_geotransform[5])

    return parameter_info

def main(xds: xr.Dataset, vxref_file: str=None, vyref_file: str=None, ssm_file: str=None, output_file: str=None, ds_filename: str=None):
    """
    Main function to re-calculate stable_shift for the image velocity pair.
    """
    # If granule is just for cataloging purposes when ROI=0, skip the
    # stable shift and error corrections
    if xds['img_pair_info'].attrs['roi_valid_percentage'] == 0.0:
        raise ITSLiveException(f"{ds_filename} is used for cataloging only.")

    param_info = find_jpl_parameter_info(xds)

    if vxref_file is None:
        vxref_file = param_info['geogrid']['vx']

    if vyref_file is None:
        vyref_file = param_info['geogrid']['vy']

    if ssm_file is None:
        ssm_file = param_info['geogrid']['ssm']

    VX = xds['vx'].data
    VY = xds['vy'].data
    V = xds['v'].data
    V_error = xds['v_error'].data
    if xds.attrs['scene_pair_type'] == 'radar':
        try:
            VR = xds['vr'].data
            VA = xds['va'].data
            M11 = xds['M11'].data * xds['M11'].dr_to_vr_factor
            M12 = xds['M12'].data * xds['M12'].dr_to_vr_factor
            VXP = xds['vxp'].data
            VYP = xds['vyp'].data
            VP = xds['vp'].data
            VP_error = xds['vp_error'].data

        except Exception as exc:
            raise RuntimError(f"Error processing {ds_filename}: {exc}")

    if np.logical_not(np.isnan(xds['vx'].stable_shift)):
        VX += xds['vx'].stable_shift
        VY += xds['vy'].stable_shift
        if xds.attrs['scene_pair_type'] == 'radar':
            VR += xds['vr'].stable_shift
            VA += xds['va'].stable_shift
            VXP += xds['vxp'].stable_shift
            VYP += xds['vyp'].stable_shift


    ds = gdal.Open(vxref_file)
    band = ds.GetRasterBand(1)
    tran = ds.GetGeoTransform()
    xoff = int(round((np.min(xds['vx'].x.data)-tran[0]-tran[1]/2)/tran[1]))
    yoff = int(round((np.max(xds['vx'].y.data)-tran[3]-tran[5]/2)/tran[5]))
    xcount = VX.shape[1]
    ycount = VX.shape[0]
#    pdb.set_trace()
    VXref = band.ReadAsArray(xoff, yoff, xcount, ycount)
    ds = None
    band = None

    ds = gdal.Open(vyref_file)
    band = ds.GetRasterBand(1)
    VYref = band.ReadAsArray(xoff, yoff, xcount, ycount)
    ds = None
    band = None

    ds = gdal.Open(ssm_file)
    band = ds.GetRasterBand(1)
    SSM = band.ReadAsArray(xoff, yoff, xcount, ycount)
    SSM = SSM.astype('bool')
    ds = None
    band = None

#    pdb.set_trace()
    if xds.attrs['scene_pair_type'] == 'radar':
        VRref = M11 * VXref + M12 * VYref
        VAref = VRref * 0.0

    NoDataValue = -32767

#   VX and VY stable shift
    stable_count = np.sum(SSM & np.logical_not(np.isnan(VX)))

    V_temp = np.sqrt(VXref**2 + VYref**2)
    try:
        V_temp_threshold = np.percentile(V_temp[np.logical_not(np.isnan(V_temp))],25)
        SSM1 = (V_temp <= V_temp_threshold)
    except IndexError:
        SSM1 = np.zeros(V_temp.shape).astype('bool')

    stable_count1 = np.sum(SSM1 & np.logical_not(np.isnan(VX)))

    vx_mean_shift = 0.0
    vy_mean_shift = 0.0
    vx_mean_shift1 = 0.0
    vy_mean_shift1 = 0.0

    if stable_count != 0:
        temp = VX.copy() - VXref.copy()
        temp[np.logical_not(SSM)] = np.nan
        vx_mean_shift = np.median(temp[np.logical_not(np.isnan(temp))])-0.0

        temp = VY.copy() - VYref.copy()
        temp[np.logical_not(SSM)] = np.nan
        vy_mean_shift = np.median(temp[np.logical_not(np.isnan(temp))])-0.0

    if stable_count1 != 0:
        temp = VX.copy() - VXref.copy()
        temp[np.logical_not(SSM1)] = np.nan
        vx_mean_shift1 = np.median(temp[np.logical_not(np.isnan(temp))])-0.0

        temp = VY.copy() - VYref.copy()
        temp[np.logical_not(SSM1)] = np.nan
        vy_mean_shift1 = np.median(temp[np.logical_not(np.isnan(temp))])-0.0

    if stable_count == 0:
        if stable_count1 == 0:
            stable_shift_applied = 0
        else:
            stable_shift_applied = 2
            VX = VX - vx_mean_shift1
            VY = VY - vy_mean_shift1
    else:
        stable_shift_applied = 1
        VX = VX - vx_mean_shift
        VY = VY - vy_mean_shift


#   VX and VY error
    if stable_count != 0:
        temp = VX.copy() - VXref.copy()
        temp[np.logical_not(SSM)] = np.nan
        vx_error_mask = np.std(temp[np.logical_not(np.isnan(temp))])
    else:
        vx_error_mask = np.nan
    if stable_count1 != 0:
        temp = VX.copy() - VXref.copy()
        temp[np.logical_not(SSM1)] = np.nan
        vx_error_slow = np.std(temp[np.logical_not(np.isnan(temp))])
    else:
        vx_error_slow = np.nan
    if stable_shift_applied == 1:
        vx_error = vx_error_mask
    elif stable_shift_applied == 2:
        vx_error = vx_error_slow
    else:
        vx_error = xds['vx'].vx_error_modeled

    if stable_count != 0:
        temp = VY.copy() - VYref.copy()
        temp[np.logical_not(SSM)] = np.nan
        vy_error_mask = np.std(temp[np.logical_not(np.isnan(temp))])
    else:
        vy_error_mask = np.nan
    if stable_count1 != 0:
        temp = VY.copy() - VYref.copy()
        temp[np.logical_not(SSM1)] = np.nan
        vy_error_slow = np.std(temp[np.logical_not(np.isnan(temp))])
    else:
        vy_error_slow = np.nan
    if stable_shift_applied == 1:
        vy_error = vy_error_mask
    elif stable_shift_applied == 2:
        vy_error = vy_error_slow
    else:
        vy_error = xds['vy'].vy_error_modeled

#   V and V_error
    V = np.sqrt(VX**2+VY**2)
    V_error = np.sqrt((vx_error * VX / V)**2 + (vy_error * VY / V)**2)

    VX[np.isnan(VX)] = NoDataValue
    VY[np.isnan(VY)] = NoDataValue
    V[np.isnan(V)] = NoDataValue

    VX = np.round(np.clip(VX, -32768, 32767)).astype(np.int16)
    VY = np.round(np.clip(VY, -32768, 32767)).astype(np.int16)
    V = np.round(np.clip(V, -32768, 32767)).astype(np.int16)

    v_error = v_error_cal(vx_error, vy_error)
    V_error[V==0] = v_error
    V_error[np.isnan(V_error)] = NoDataValue
    V_error = np.round(np.clip(V_error, -32768, 32767)).astype(np.int16)

#   Update nc file
    try:
        if isinstance(vx_error, np.ndarray) and vx_error.size == 1:
            vx_error = vx_error[0]

        xds['vx'].attrs['vx_error'] = int(round(vx_error*10))/10

    except Exception as exc:
        raise RuntimeError(f"Error processing {ds_filename}: {exc}. vx_error={vx_error} stable_count={stable_count} stable_count1={stable_count1} stable_shift_applied={stable_shift_applied}")

    if stable_shift_applied == 2:
        xds['vx'].attrs['stable_shift'] = int(round(vx_mean_shift1*10))/10
    elif stable_shift_applied == 1:
        xds['vx'].attrs['stable_shift'] = int(round(vx_mean_shift*10))/10
    else:
        xds['vx'].attrs['stable_shift'] = np.nan

    xds['vx'].attrs['stable_count_mask'] = stable_count
    xds['vx'].attrs['stable_count_slow'] = stable_count1
    if stable_count != 0:
        xds['vx'].attrs['stable_shift_mask'] = int(round(vx_mean_shift*10))/10
    else:
        xds['vx'].attrs['stable_shift_mask'] = np.nan
    if stable_count1 != 0:
        xds['vx'].attrs['stable_shift_slow'] = int(round(vx_mean_shift1*10))/10
    else:
        xds['vx'].attrs['stable_shift_slow'] = np.nan

    if stable_count != 0:
        xds['vx'].attrs['vx_error_mask'] = int(round(vx_error_mask*10))/10
    else:
        xds['vx'].attrs['vx_error_mask'] = np.nan
    if stable_count1 != 0:
        xds['vx'].attrs['vx_error_slow'] = int(round(vx_error_slow*10))/10
    else:
        xds['vx'].attrs['vx_error_slow'] = np.nan

    if isinstance(vy_error, np.ndarray) and vy_error.size == 1:
        vy_error = vy_error[0]

    xds['vy'].attrs['vy_error'] = int(round(vy_error*10))/10
    if stable_shift_applied == 2:
        xds['vy'].attrs['stable_shift'] = int(round(vy_mean_shift1*10))/10
    elif stable_shift_applied == 1:
        xds['vy'].attrs['stable_shift'] = int(round(vy_mean_shift*10))/10
    else:
        xds['vy'].attrs['stable_shift'] = np.nan

    xds['vy'].attrs['stable_count_mask'] = stable_count
    xds['vy'].attrs['stable_count_slow'] = stable_count1
    if stable_count != 0:
        xds['vy'].attrs['stable_shift_mask'] = int(round(vy_mean_shift*10))/10
    else:
        xds['vy'].attrs['stable_shift_mask'] = np.nan
    if stable_count1 != 0:
        xds['vy'].attrs['stable_shift_slow'] = int(round(vy_mean_shift1*10))/10
    else:
        xds['vy'].attrs['stable_shift_slow'] = np.nan

    if stable_count != 0:
        xds['vy'].attrs['vy_error_mask'] = int(round(vy_error_mask*10))/10
    else:
        xds['vy'].attrs['vy_error_mask'] = np.nan
    if stable_count1 != 0:
        xds['vy'].attrs['vy_error_slow'] = int(round(vy_error_slow*10))/10
    else:
        xds['vy'].attrs['vy_error_slow'] = np.nan

    xds['vx'].data = VX
    xds['vy'].data = VY
    xds['v'].data = V
    xds['v_error'].data = V_error


    if xds.attrs['scene_pair_type'] == 'radar':
        #   VR and VA stable shift
        vr_mean_shift = 0.0
        va_mean_shift = 0.0
        vr_mean_shift1 = 0.0
        va_mean_shift1 = 0.0

        if stable_count != 0:
            temp = VR.copy() - VRref.copy()
            temp[np.logical_not(SSM)] = np.nan
            vr_mean_shift = np.median(temp[np.logical_not(np.isnan(temp))])-0.0

            temp = VA.copy() - VAref.copy()
            temp[np.logical_not(SSM)] = np.nan
            va_mean_shift = np.median(temp[np.logical_not(np.isnan(temp))])-0.0

        if stable_count1 != 0:
            temp = VR.copy() - VRref.copy()
            temp[np.logical_not(SSM1)] = np.nan
            vr_mean_shift1 = np.median(temp[np.logical_not(np.isnan(temp))])-0.0

            temp = VA.copy() - VAref.copy()
            temp[np.logical_not(SSM1)] = np.nan
            va_mean_shift1 = np.median(temp[np.logical_not(np.isnan(temp))])-0.0

        if stable_count == 0:
            if stable_count1 == 0:
                stable_shift_applied = 0
            else:
                stable_shift_applied = 2
                VR = VR - vr_mean_shift1
                VA = VA - va_mean_shift1
        else:
            stable_shift_applied = 1
            VR = VR - vr_mean_shift
            VA = VA - va_mean_shift


        #   VR and VA error
        if stable_count != 0:
            temp = VR.copy() - VRref.copy()
            temp[np.logical_not(SSM)] = np.nan
            vr_error_mask = np.std(temp[np.logical_not(np.isnan(temp))])
        else:
            vr_error_mask = np.nan
        if stable_count1 != 0:
            temp = VR.copy() - VRref.copy()
            temp[np.logical_not(SSM1)] = np.nan
            vr_error_slow = np.std(temp[np.logical_not(np.isnan(temp))])
        else:
            vr_error_slow = np.nan
        if stable_shift_applied == 1:
            vr_error = vr_error_mask
        elif stable_shift_applied == 2:
            vr_error = vr_error_slow
        else:
            vr_error = xds['vr'].vr_error_modeled

        if stable_count != 0:
            temp = VA.copy() - VAref.copy()
            temp[np.logical_not(SSM)] = np.nan
            va_error_mask = np.std(temp[np.logical_not(np.isnan(temp))])
        else:
            va_error_mask = np.nan
        if stable_count1 != 0:
            temp = VA.copy() - VAref.copy()
            temp[np.logical_not(SSM1)] = np.nan
            va_error_slow = np.std(temp[np.logical_not(np.isnan(temp))])
        else:
            va_error_slow = np.nan
        if stable_shift_applied == 1:
            va_error = va_error_mask
        elif stable_shift_applied == 2:
            va_error = va_error_slow
        else:
            va_error = xds['va'].va_error_modeled

        #   V and V_error
        VR[np.isnan(VR)] = NoDataValue
        VA[np.isnan(VA)] = NoDataValue

        VR = np.round(np.clip(VR, -32768, 32767)).astype(np.int16)
        VA = np.round(np.clip(VA, -32768, 32767)).astype(np.int16)

        #   Update nc file
        if isinstance(vr_error, np.ndarray) and vr_error.size == 1:
            vr_error = vr_error[0]

        xds['vr'].attrs['vr_error'] = int(round(vr_error*10))/10
        if stable_shift_applied == 2:
            xds['vr'].attrs['stable_shift'] = int(round(vr_mean_shift1*10))/10
        elif stable_shift_applied == 1:
            xds['vr'].attrs['stable_shift'] = int(round(vr_mean_shift*10))/10
        else:
            xds['vr'].attrs['stable_shift'] = np.nan

        xds['vr'].attrs['stable_count_mask'] = stable_count
        xds['vr'].attrs['stable_count_slow'] = stable_count1
        if stable_count != 0:
            xds['vr'].attrs['stable_shift_mask'] = int(round(vr_mean_shift*10))/10
        else:
            xds['vr'].attrs['stable_shift_mask'] = np.nan
        if stable_count1 != 0:
            xds['vr'].attrs['stable_shift_slow'] = int(round(vr_mean_shift1*10))/10
        else:
            xds['vr'].attrs['stable_shift_slow'] = np.nan

        if stable_count != 0:
            xds['vr'].attrs['vr_error_mask'] = int(round(vr_error_mask*10))/10
        else:
            xds['vr'].attrs['vr_error_mask'] = np.nan
        if stable_count1 != 0:
            xds['vr'].attrs['vr_error_slow'] = int(round(vr_error_slow*10))/10
        else:
            xds['vr'].attrs['vr_error_slow'] = np.nan


        if isinstance(va_error, np.ndarray) and va_error.size == 1:
            va_error = va_error[0]

        xds['va'].attrs['va_error'] = int(round(va_error*10))/10
        if stable_shift_applied == 2:
            xds['va'].attrs['stable_shift'] = int(round(va_mean_shift1*10))/10
        elif stable_shift_applied == 1:
            xds['va'].attrs['stable_shift'] = int(round(va_mean_shift*10))/10
        else:
            xds['va'].attrs['stable_shift'] = np.nan

        xds['va'].attrs['stable_count_mask'] = stable_count
        xds['va'].attrs['stable_count_slow'] = stable_count1
        if stable_count != 0:
            xds['va'].attrs['stable_shift_mask'] = int(round(va_mean_shift*10))/10
        else:
            xds['va'].attrs['stable_shift_mask'] = np.nan
        if stable_count1 != 0:
            xds['va'].attrs['stable_shift_slow'] = int(round(va_mean_shift1*10))/10
        else:
            xds['va'].attrs['stable_shift_slow'] = np.nan

        if stable_count != 0:
            xds['va'].attrs['va_error_mask'] = int(round(va_error_mask*10))/10
        else:
            xds['va'].attrs['va_error_mask'] = np.nan
        if stable_count1 != 0:
            xds['va'].attrs['va_error_slow'] = int(round(va_error_slow*10))/10
        else:
            xds['va'].attrs['va_error_slow'] = np.nan

        xds['vr'].data = VR
        xds['va'].data = VA


        #   VXP and VYP stable shift
        stable_count_p = np.sum(SSM & np.logical_not(np.isnan(VXP)))

        stable_count1_p = np.sum(SSM1 & np.logical_not(np.isnan(VXP)))

        vxp_mean_shift = 0.0
        vyp_mean_shift = 0.0
        vxp_mean_shift1 = 0.0
        vyp_mean_shift1 = 0.0

        if stable_count_p != 0:
            temp = VXP.copy() - VXref.copy()
            temp[np.logical_not(SSM)] = np.nan
            vxp_mean_shift = np.median(temp[np.logical_not(np.isnan(temp))])-1.3

            temp = VYP.copy() - VYref.copy()
            temp[np.logical_not(SSM)] = np.nan
            vyp_mean_shift = np.median(temp[np.logical_not(np.isnan(temp))])-1.3

        if stable_count1_p != 0:
            temp = VXP.copy() - VXref.copy()
            temp[np.logical_not(SSM1)] = np.nan
            vxp_mean_shift1 = np.median(temp[np.logical_not(np.isnan(temp))])-1.3

            temp = VYP.copy() - VYref.copy()
            temp[np.logical_not(SSM1)] = np.nan
            vyp_mean_shift1 = np.median(temp[np.logical_not(np.isnan(temp))])-1.3

        if stable_count_p == 0:
            if stable_count1_p == 0:
                stable_shift_applied_p = 0
            else:
                stable_shift_applied_p = 2
                VXP = VXP - vxp_mean_shift1
                VYP = VYP - vyp_mean_shift1
        else:
            stable_shift_applied_p = 1
            VXP = VXP - vxp_mean_shift
            VYP = VYP - vyp_mean_shift


        #   VXP and VYP error
        if stable_count_p != 0:
            temp = VXP.copy() - VXref.copy()
            temp[np.logical_not(SSM)] = np.nan
            vxp_error_mask = np.std(temp[np.logical_not(np.isnan(temp))])
        else:
            vxp_error_mask = np.nan
        if stable_count1_p != 0:
            temp = VXP.copy() - VXref.copy()
            temp[np.logical_not(SSM1)] = np.nan
            vxp_error_slow = np.std(temp[np.logical_not(np.isnan(temp))])
        else:
            vxp_error_slow = np.nan
        if stable_shift_applied_p == 1:
            vxp_error = vxp_error_mask
        elif stable_shift_applied_p == 2:
            vxp_error = vxp_error_slow
        else:
            vxp_error = xds['vxp'].vxp_error_modeled

        if stable_count_p != 0:
            temp = VYP.copy() - VYref.copy()
            temp[np.logical_not(SSM)] = np.nan
            vyp_error_mask = np.std(temp[np.logical_not(np.isnan(temp))])
        else:
            vyp_error_mask = np.nan
        if stable_count1_p != 0:
            temp = VYP.copy() - VYref.copy()
            temp[np.logical_not(SSM1)] = np.nan
            vyp_error_slow = np.std(temp[np.logical_not(np.isnan(temp))])
        else:
            vyp_error_slow = np.nan
        if stable_shift_applied_p == 1:
            vyp_error = vyp_error_mask
        elif stable_shift_applied_p == 2:
            vyp_error = vyp_error_slow
        else:
            vyp_error = xds['vyp'].vyp_error_modeled

        #   V and V_error
        VP = np.sqrt(VXP**2+VYP**2)
        VP_error = np.sqrt((vxp_error * VXP / VP)**2 + (vyp_error * VYP / VP)**2)

        VXP[np.isnan(VXP)] = NoDataValue
        VYP[np.isnan(VYP)] = NoDataValue
        VP[np.isnan(VP)] = NoDataValue

        VXP = np.round(np.clip(VXP, -32768, 32767)).astype(np.int16)
        VYP = np.round(np.clip(VYP, -32768, 32767)).astype(np.int16)
        VP = np.round(np.clip(VP, -32768, 32767)).astype(np.int16)

        vp_error = v_error_cal(vxp_error, vyp_error)
        VP_error[VP==0] = vp_error
        VP_error[np.isnan(VP_error)] = NoDataValue
        VP_error = np.round(np.clip(VP_error, -32768, 32767)).astype(np.int16)

        #   Update nc file
        if isinstance(vxp_error, np.ndarray) and vxp_error.size == 1:
            vxp_error = vxp_error[0]

        xds['vxp'].attrs['vxp_error'] = int(round(vxp_error*10))/10
        if stable_shift_applied_p == 2:
            xds['vxp'].attrs['stable_shift'] = int(round(vxp_mean_shift1*10))/10
        elif stable_shift_applied_p == 1:
            xds['vxp'].attrs['stable_shift'] = int(round(vxp_mean_shift*10))/10
        else:
            xds['vxp'].attrs['stable_shift'] = np.nan

        xds['vxp'].attrs['stable_count_mask'] = stable_count_p
        xds['vxp'].attrs['stable_count_slow'] = stable_count1_p
        if stable_count_p != 0:
            xds['vxp'].attrs['stable_shift_mask'] = int(round(vxp_mean_shift*10))/10
        else:
            xds['vxp'].attrs['stable_shift_mask'] = np.nan
        if stable_count1_p != 0:
            xds['vxp'].attrs['stable_shift_slow'] = int(round(vxp_mean_shift1*10))/10
        else:
            xds['vxp'].attrs['stable_shift_slow'] = np.nan

        if stable_count_p != 0:
            xds['vxp'].attrs['vxp_error_mask'] = int(round(vxp_error_mask*10))/10
        else:
            xds['vxp'].attrs['vxp_error_mask'] = np.nan
        if stable_count1_p != 0:
            xds['vxp'].attrs['vxp_error_slow'] = int(round(vxp_error_slow*10))/10
        else:
            xds['vxp'].attrs['vxp_error_slow'] = np.nan


        if isinstance(vyp_error, np.ndarray) and vyp_error.size == 1:
            vyp_error = vyp_error[0]

        xds['vyp'].attrs['vyp_error'] = int(round(vyp_error*10))/10
        if stable_shift_applied_p == 2:
            xds['vyp'].attrs['stable_shift'] = int(round(vyp_mean_shift1*10))/10
        elif stable_shift_applied_p == 1:
            xds['vyp'].attrs['stable_shift'] = int(round(vyp_mean_shift*10))/10
        else:
            xds['vyp'].attrs['stable_shift'] = np.nan

        xds['vyp'].attrs['stable_count_mask'] = stable_count_p
        xds['vyp'].attrs['stable_count_slow'] = stable_count1_p
        if stable_count_p != 0:
            xds['vyp'].attrs['stable_shift_mask'] = int(round(vyp_mean_shift*10))/10
        else:
            xds['vyp'].attrs['stable_shift_mask'] = np.nan
        if stable_count1_p != 0:
            xds['vyp'].attrs['stable_shift_slow'] = int(round(vyp_mean_shift1*10))/10
        else:
            xds['vyp'].attrs['stable_shift_slow'] = np.nan

        if stable_count_p != 0:
            xds['vyp'].attrs['vyp_error_mask'] = int(round(vyp_error_mask*10))/10
        else:
            xds['vyp'].attrs['vyp_error_mask'] = np.nan
        if stable_count1_p != 0:
            xds['vyp'].attrs['vyp_error_slow'] = int(round(vyp_error_slow*10))/10
        else:
            xds['vyp'].attrs['vyp_error_slow'] = np.nan

        xds['vxp'].data = VXP
        xds['vyp'].data = VYP
        xds['vp'].data = VP
        xds['vp_error'].data = VP_error

    if output_file is not None:
        # TODO: writing to the file should take encoding dictionary
        xds.to_netcdf(inps.output)

    return xds

if __name__ == '__main__':

    #####Parse command line
    inps = cmdLineParse()

    xds = xr.open_dataset(inps.nc)

    _ = main(xds, inps.VXref, inps.VYref, inps.SSM, inps.output)

#    xds.rio.write_crs("epsg:"+str(inps.epsg), inplace=True)
#    xds[inps.output].rio.to_raster(os.path.dirname(inps.input)+'/'+inps.output+'.tif')


##    pdb.set_trace()

#   stable shift comparison
#   original: M^{-1} * median([DX;DY] - M * [VXref;VYref])
#   patch:    median(M^{-1} * [DX;DY] - [VXref;VYref])

#   error comparison
#   original: std(M^{-1} * [DX;DY] - stable_shift_original)
#   patch:    std(M^{-1} * [DX;DY] - stable_shift_patch)
