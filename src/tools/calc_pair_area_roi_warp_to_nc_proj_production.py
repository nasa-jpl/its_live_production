
from osgeo import gdal,osr
import sys
import os
import numpy as np
import datetime as dt

# import random
# import string

import xarray as xr

import subprocess as sp

import time

# from matplotlib import pyplot as plt

# import boto3
import fiona
from shapely.geometry import Point, Polygon, shape
import pyproj # used to find lon,lat from img1 midpoint

# for reading parquet file
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

import s3fs

import argparse
from tqdm import tqdm



start = time.time()
last_time = start

verbose = False # get some debug info... Not implemented much yet

# read in istlive zones shapefile once here so we can use it multiple times below
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')


gdal.SetConfigOption('AWS_NO_SIGN_REQUEST','YES')
itslive_zones=[]
with fiona.open('s3://its-live-data/autorift_parameters/v001/autorift_landice_0120m.shp') as its_live_zones:
    for feature in its_live_zones:
        itslive_zones.append((shape(feature['geometry']),feature))
gdal.SetConfigOption('AWS_NO_SIGN_REQUEST','NO')


now_time = time.time()
print(f'read itslive zones shapefile from s3 {now_time - start:6.1f}',flush=True)
last_time = now_time



# gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
# gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')



####################
#
# next 3 functions set up outputs for the specified conditions - change here to what you need in an output file for new p000, errors, etc.
#
####################

def output_function_for_v_all_nan_values(pydict,out_v02_v_allnan):
    print(f">>>> skipping  - all nan values for v02,  {pydict['v2_url'][0].split('/')[-1]}")
    out_v02_v_allnan.write(f">>>> skipping  - all nan values for v02,  {pydict['v2_url'][0]}\n")

def output_function_for_nc_file_not_found(pydict,e,v02_v_not_found):
    print(f">>>>>> skipping {pydict['v2_url'][0].split('/')[-1]} not found ")
    print(f"exception {e}")
    v02_v_not_found.write(f">>>>>> skipping {pydict['v2_url'][0].split('/')[-1]} not found exception:{e}\n")

def output_function_for_new_pvalid_found(pydict,new_p_valid,out_pvalid_fix):
    v_name = pydict['v2_url'][0].split('/')[-1]
    v1_pvalid = pydict['v1_pvalid'][0]
    print(f'{v_name} ==>  new_pvalid {new_p_valid*100.1:3.1f}  (v1_pvalid {v1_pvalid})')
    out_pvalid_fix.write(f"{pydict['v2_url'][0]}  new_pvalid {new_p_valid*100.1:3.1f}  v1_pvalid {v1_pvalid}\n")

def output_function_for_ROI_mask_all_zero(pydict,out_ROI_allnan):
    print(f">>>>>> skipping {pydict['v2_url'][0].split('/')[-1]} ROI mask all zero")
    out_ROI_allnan.write(f"{pydict['v2_url'][0].split('/')[-1]} ROI mask all zero \n")

def output_function_for_common_image_areas_in_ROI_all_zero(pydict):
    print(f">>>>>> skipping {pydict['v2_url'][0].split('/')[-1]} common_image_areas_in_ROI all zero")




def return_its_live_land_mask_URL_and_epsg(lat=None,lon=None, itslive_zones=itslive_zones, verbose=verbose):
#     gdal.SetConfigOption('AWS_NO_SIGN_REQUEST','YES')
#     with fiona.open('s3://its-live-data/autorift_parameters/v001/autorift_landice_0120m.shp') as its_live_zones:
#         itslive_zones=[]
#         for feature in its_live_zones:
#             itslive_zones.append((shape(feature['geometry']),feature))
    pt = Point(lon,lat)
    ptzone = [l for geom,l in itslive_zones if geom.contains(pt)] # should only return 1 feature in list
    landmask_tif_url = ptzone[0]['properties']['ROI']
    landmask_epsg = ptzone[0]['properties']['epsg']
#     print(ptzone[0])
#     gdal.SetConfigOption('AWS_NO_SIGN_REQUEST','NO')
    return(landmask_tif_url,landmask_epsg)

# def return_its_live_land_mask_URL(lat=None,lon=None):
#     gdal.SetConfigOption('AWS_NO_SIGN_REQUEST','YES')
#     with fiona.open('s3://its-live-data/autorift_parameters/v001/autorift_landice_0120m.shp') as its_live_zones:
#         itslive_zones=[]
#         for feature in its_live_zones:
#             itslive_zones.append((shape(feature['geometry']),feature))
#     pt = Point(lon,lat)
#     ptzone = [l for geom,l in itslive_zones if geom.contains(pt)] # should only return 1 feature in list
#     landmask_tif_url = ptzone[0]['properties']['ROI']
#     gdal.SetConfigOption('AWS_NO_SIGN_REQUEST','NO')
#     return(landmask_tif_url)




class GeoImg_noload:
    """geocoded image input and info
        a=GeoImg(in_file_name,indir='.', datestr=None, datefmt='%m/%d/%y')
            a.img will contain image
            a.parameter etc...
            datefmt is datetime format string dt.datetime.strptime()"""

    # LXSS_LLLL_PPPRRR_YYYYMMDD_yyymmdd_CC_TX, in which:
    # L = Landsat
    # X = Sensor
    # SS = Satellite
    # PPP = WRS path
    # RRR = WRS row
    # YYYYMMDD = Acquisition date
    # yyyymmdd = Processing date
    # CC = Collection number
    # TX = Collection category

    def __init__(self, in_filename,in_dir='.',datestr=None,datefmt='%m/%d/%y'):
        self.filename = in_filename
        self.in_dir_path = in_dir  #in_dir can be relative...
        self.in_dir_abs_path=os.path.abspath(in_dir)  # get absolute path for later ref if needed
        self.gd=gdal.Open(self.in_dir_path + '/' + self.filename)
        if not(self.gd):
            print('Error: open of %s failed: gdal_error: %s'%(self.in_dir_path + os.path.sep + self.filename, gdal.GetLastErrorMsg()))
            sys.exit(0)
        self.nodata_value=self.gd.GetRasterBand(1).GetNoDataValue()
        self.srs=osr.SpatialReference(wkt=self.gd.GetProjection())
        self.gt=self.gd.GetGeoTransform()
        self.proj=self.gd.GetProjection()
        self.intype=self.gd.GetDriver().ShortName
        self.min_x=self.gt[0]
        self.max_x=self.gt[0]+self.gd.RasterXSize*self.gt[1]
        self.min_y=self.gt[3]+self.gt[5]*self.gd.RasterYSize
        self.max_y=self.gt[3]
        self.pix_x_m=self.gt[1]
        self.pix_y_m=self.gt[5]
        self.num_pix_x=self.gd.RasterXSize
        self.num_pix_y=self.gd.RasterYSize
        self.XYtfm=np.array([self.min_x,self.max_y,self.pix_x_m,self.pix_y_m]).astype('float')
        if (datestr is not None):   # date specified in GeoImg call directly - could be any GeoTiff...
            self.imagedatetime=dt.datetime.strptime(datestr,datefmt)
        elif (self.filename.count('_')>=7 and self.filename[0]=='L'): # looks like collection 1 landsat
            b=self.filename.split('_')
            self.sensor=b[0]
            self.path=int(b[2][0:3])
            self.row=int(b[2][3:6])
            self.year=int(b[3][0:4])
            self.imagedatetime=dt.datetime.strptime(b[3],'%Y%m%d')
            self.doy=self.imagedatetime.timetuple().tm_yday
        elif ((self.filename.find('LC8') == 0) | (self.filename.find('LO8') == 0) | \
                (self.filename.find('LE7') == 0) | (self.filename.find('LT5') == 0) | \
                (self.filename.find('LT4') == 0)):    # looks landsat like (old filenames) - try parsing the date from filename (contains day of year)
            self.sensor=self.filename[0:3]
            self.path=int(self.filename[3:6])
            self.row=int(self.filename[6:9])
            self.year=int(self.filename[9:13])
            self.doy=int(self.filename[13:16])
            self.imagedatetime=dt.datetime.fromordinal(dt.date(self.year-1,12,31).toordinal()+self.doy)
        elif ( (self.filename.find('S2A') == 0) | (self.filename.find('S2B') == 0) | \
                ((self.filename.find('T') == 0) & (self.filename.find('_') == 6)) ):    # looks like sentinal 2 data (old or new format) - try parsing the date from filename (contains day of year)
            if self.filename.find('S2') == 0:  # old format Sentinel 2 data
                self.sensor=self.filename[0:3]
                b=re.search('_(?P<date>\d{8})T(?P<time>\d{6})_T(?P<tile>[A-Z0-9]{5})_A(?P<orbit>\d{6})_R(?P<rel_orbit>\d{3})_',self.filename)
                self.path=np.mod(int(b.group('orbit')),143)+3  # why + 3?  there is an offset between rel_orbit and absolute orbit numbers for S2A
                self.tile=b.group('tile')
                self.imagedatetime=dt.datetime.strptime(b.group('date'),'%Y%m%d')
            else:
                self.sensor='S2'  # would have to get S2A or when it flies S2B from full file path, which I may not maintain
                b=re.search('T(?P<tile>[A-Z0-9]{5})_(?P<date>\d{8})T(?P<time>\d{6})',self.filename)
                self.tile=b.group('tile')
                self.imagedatetime=dt.datetime.strptime(b.group('date'),'%Y%m%d')
        else:
            self.imagedatetime=None  # need to throw error in this case...or get it from metadata
        #         self.img=self.gd.ReadAsArray().astype(np.float32)   # works for L8 and earlier - and openCV correlation routine needs float or byte so just use float...
    def imageij2XY(self,ai,aj,outx=None,outy=None):
        it = np.nditer([ai,aj,outx,outy],
                        flags = ['external_loop', 'buffered'],
                        op_flags = [['readonly'],['readonly'],
                                    ['writeonly', 'allocate', 'no_broadcast'],
                                    ['writeonly', 'allocate', 'no_broadcast']])
        for ii,jj,ox,oy in it:
            ox[...]=(self.XYtfm[0]+((ii+0.5)*self.XYtfm[2]));
            oy[...]=(self.XYtfm[1]+((jj+0.5)*self.XYtfm[3]));
        return np.array(it.operands[2:4])
    def XY2imageij(self,ax,ay,outi=None,outj=None):
        it = np.nditer([ax,ay,outi,outj],
                        flags = ['external_loop', 'buffered'],
                        op_flags = [['readonly'],['readonly'],
                                    ['writeonly', 'allocate', 'no_broadcast'],
                                    ['writeonly', 'allocate', 'no_broadcast']])
        for xx,yy,oi,oj in it:
            oi[...]=((xx-self.XYtfm[0])/self.XYtfm[2])-0.5;  # if python arrays started at 1, + 0.5
            oj[...]=((yy-self.XYtfm[1])/self.XYtfm[3])-0.5;  # " " " " "
        return np.array(it.operands[2:4])




class GeoImg_noload_from_gd:
    """geocoded image input and info
        a=GeoImg(in_file_name,indir='.', datestr=None, datefmt='%m/%d/%y')
            a.img will contain image
            a.parameter etc...
            datefmt is datetime format string dt.datetime.strptime()"""

    # LXSS_LLLL_PPPRRR_YYYYMMDD_yyymmdd_CC_TX, in which:
    # L = Landsat
    # X = Sensor
    # SS = Satellite
    # PPP = WRS path
    # RRR = WRS row
    # YYYYMMDD = Acquisition date
    # yyyymmdd = Processing date
    # CC = Collection number
    # TX = Collection category

    def __init__(self, in_gd):
        self.gd=in_gd
        self.nodata_value=self.gd.GetRasterBand(1).GetNoDataValue()
        self.srs=osr.SpatialReference(wkt=self.gd.GetProjection())
        self.gt=self.gd.GetGeoTransform()
        self.proj=self.gd.GetProjection()
        self.intype=self.gd.GetDriver().ShortName
        self.min_x=self.gt[0]
        self.max_x=self.gt[0]+self.gd.RasterXSize*self.gt[1]
        self.min_y=self.gt[3]+self.gt[5]*self.gd.RasterYSize
        self.max_y=self.gt[3]
        self.pix_x_m=self.gt[1]
        self.pix_y_m=self.gt[5]
        self.num_pix_x=self.gd.RasterXSize
        self.num_pix_y=self.gd.RasterYSize
        self.XYtfm=np.array([self.min_x,self.max_y,self.pix_x_m,self.pix_y_m]).astype('float')

    def imageij2XY(self,ai,aj,outx=None,outy=None):
        it = np.nditer([ai,aj,outx,outy],
                        flags = ['external_loop', 'buffered'],
                        op_flags = [['readonly'],['readonly'],
                                    ['writeonly', 'allocate', 'no_broadcast'],
                                    ['writeonly', 'allocate', 'no_broadcast']])
        for ii,jj,ox,oy in it:
            ox[...]=(self.XYtfm[0]+((ii+0.5)*self.XYtfm[2]));
            oy[...]=(self.XYtfm[1]+((jj+0.5)*self.XYtfm[3]));
        return np.array(it.operands[2:4])

    def XY2imageij(self,ax,ay,outi=None,outj=None):
        it = np.nditer([ax,ay,outi,outj],
                        flags = ['external_loop', 'buffered'],
                        op_flags = [['readonly'],['readonly'],
                                    ['writeonly', 'allocate', 'no_broadcast'],
                                    ['writeonly', 'allocate', 'no_broadcast']])
        for xx,yy,oi,oj in it:
            oi[...]=((xx-self.XYtfm[0])/self.XYtfm[2])-0.5;  # if python arrays started at 1, + 0.5
            oj[...]=((yy-self.XYtfm[1])/self.XYtfm[3])-0.5;  # " " " " "
        return np.array(it.operands[2:4])



# set up command line arguments
parser = argparse.ArgumentParser( \
    description="""calc_pair_area_roi_warp_to_nc_proj_production.py

                    estimates area (pixel count) of possible data for an image pair,
                    divides actual granule valid pixels by this count

                    output for p_valid fix is current_nc_s3URL, new_p_valid, v1_p_valid, v2_p_valid

                    also outputs to different files - v02 v all nan; ROI mask all nan""",
    epilog='',
    formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('start_path',
                    action='store',
                    type=int,
                    default=1,
                    help='Landsat Path to start with [%(default)d]')
parser.add_argument('stop_path',
                    action='store',
                    type=int,
                    default=233,
                    help='Landsat Path to stop at [%(default)d]')

parser.add_argument('start_row',
                    action='store',
                    type=int,
                    default=0,
                    help='Landsat Path row index to start at [%(default)d]')

parser.add_argument('-output_file_new_p_valid',
                    action='store',
                    type=str,
                    default='new_p_valid_fixes_(startpath)_(stoppath).txt',
                    help='output file for pairs needing fix to p_valid [%(default)s]')

parser.add_argument('-output_file_v02_v_allnan',
                    action='store',
                    type=str,
                    default='v02_v_allnan_(startpath)_(stoppath).txt',
                    help='output file for pairs v02_v_allnan [%(default)s]')

parser.add_argument('-output_file_ROI_allnan',
                    action='store',
                    type=str,
                    default='ROI_allnan_(startpath)_(stoppath).txt',
                    help='output file for pairs ROI_allnan [%(default)s]')

parser.add_argument('-output_file_v02_v_not_found',
                    action='store',
                    type=str,
                    default='v02_v_not_found_(startpath)_(stoppath).txt',
                    help='output file for pairs 02_v_not_found [%(default)s]')


args = parser.parse_args()




gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
# gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')

# Acquire access key and secret access key for AWS access (may be use environment
# variable or store in the external file
gdal.SetConfigOption('AWS_ACCESS_KEY_ID', '')
gdal.SetConfigOption('AWS_SECRET_ACCESS_KEY', '')
gdal.SetConfigOption('AWS_REQUEST_PAYER','requester')

# gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', 'TIF')
gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', None)





# read parguet file of all L7 during interval with processing issue

# tbl = pq.read_table('l7_pvalue_comp_with_fix.parquet')
tbl = pq.read_table('s3://its-live-data/L7_PV_fix/l7_pvalue_comp_with_fix.parquet')
print(f'read s3://its-live-data/L7_PV_fix/l7_pvalue_comp_with_fix.parquet', flush=True)

# lines from the parquet file (.to_pydict()) - this is what ends up in pydict below in loop...
#
# pydict_lines = [
#  {'reference': ['LE07_L1GT_229114_20121115_20200908_02_T2'],
#  'secondary': ['LE07_L1GT_232114_20121206_20200908_02_T2'],
#  'ref_path': [229],
#  'ref_row': [114],
#  'ref_s3_path': ['s3://usgs-landsat/collection02/level-1/standard/etm/2012/229/114/LE07_L1GT_229114_20121115_20200908_02_T2/LE07_L1GT_229114_20121115_20200908_02_T2_B8.TIF'],
#  'ref_url': [None],
#  'sec_path': [232],
#  'sec_row': [114],
#  'sec_s3_path': ['s3://usgs-landsat/collection02/level-1/standard/etm/2012/232/114/LE07_L1GT_232114_20121206_20200908_02_T2/LE07_L1GT_232114_20121206_20200908_02_T2_B8.TIF'],
#  'sec_url': [None],
#  'geometry': [b"\x01\x03\x00\x00\x00\x01\x00\x00\x00\r\x00\x00\x00\xb2\xd4\x1b\xe5\xcf\xf9W\xc0\xc3r\\\x7f\xf0;S\xc0\xbb\tE\xec<\xffW\xc0\x9bR\xc6~\xdb=S\xc0\xc5 \xb0rh\x05X\xc0+\x18\x95\xd4\t@S\xc0)y\xc9?S\x06X\xc0\xfe\xf9\x1e\xec\xdf?S\xc0Zd;\xdfOmY\xc0yX\xa85\xcd\xffR\xc0L\xe5b\xcd\x13gY\xc0\x89\xebC!\xdd\xfdR\xc0^\x95\xe4x}}X\xc00\xf0\xd6\x03C\xb5R\xc0k+\xf6\x97\xddwX\xc0:#J{\x83\xb3R\xc0x\xa7\xe8\x12`\x1fW\xc0\x0c\xc3;\x1d`\xeeR\xc0\xbb'\x0f\x0b\xb5\x1eW\xc0\x1f\xf4lV}\xeeR\xc0\xae[\x12I\xfa#W\xc0\x11\x15\x90EZ\xf0R\xc0B*\xe3U\xf2)W\xc0\xf5\x07Piv\xf2R\xc0\xb2\xd4\x1b\xe5\xcf\xf9W\xc0\xc3r\\\x7f\xf0;S\xc0"],
#  'mission': ['L7'],
#  'days_separation': [21],
#  'is_xpr': [True],
#  'v1_url': ['https://its-live-data.s3.us-west-2.amazonaws.com/velocity_image_pair/landsatOLI/v01/3031/LE07_L1GT_232114_20121206_20161127_01_T2_X_LE07_L1GT_229114_20121115_20161127_01_T2_G0240V01_P057.nc'],
#  'v1_pvalid': [57],
#  'v2_url': ['https://its-live-data.s3.us-west-2.amazonaws.com/velocity_image_pair/landsatOLI-latest/S70W090/LE07_L1GT_229114_20121115_20200908_02_T2_X_LE07_L1GT_232114_20121206_20200908_02_T2_G0120V02_P011.nc'],
#  'v2_pvalid': [11],
#  'fix_request_time': [None],
#  'fix_url': [None],
#  'fix_pvalid': [None]},





# plt.ion()


# loop over  Landsat paths (itslive has every path) - then over Landsat rows in that path
for path in range(args.start_path,(args.stop_path+1)):
    tbl2 = tbl.filter(pc.equal(tbl["ref_path"],path))
    print(f"{path} {tbl2.num_rows}")
    rows = np.unique(tbl2['ref_row'])

    out_pvalid_filename = args.output_file_new_p_valid.replace("(startpath)",f"{path:03d}").replace("(stoppath)",f"{path:03d}")
    out_pvalid_fix = open(out_pvalid_filename,'w')

    out_v02_v_allnan_filename = args.output_file_v02_v_allnan.replace("(startpath)",f"{path:03d}").replace("(stoppath)",f"{path:03d}")
    out_v02_v_allnan = open(out_v02_v_allnan_filename,'w')

    out_ROI_allnan_filename = args.output_file_ROI_allnan.replace("(startpath)",f"{path:03d}").replace("(stoppath)",f"{path:03d}")
    out_ROI_allnan = open(out_ROI_allnan_filename,'w')

    v02_v_not_found_filename = args.output_file_v02_v_not_found.replace("(startpath)",f"{path:03d}").replace("(stoppath)",f"{path:03d}")
    v02_v_not_found = open(v02_v_not_found_filename,'w')

    start_row = args.start_row
    for row in tqdm(rows[start_row:],desc=f"working on {len(rows)} rows for path {path}"):
        pr_starttime = time.time()

        print(f'working on path {path} row {row}',flush=True)

        # filter to single path/row to limit list size and so we can do the ROI lookup once for all scenes
        #         tbl2 = tbl.filter(pc.equal(tbl["ref_path"],path))
        tbl3 = tbl2.filter(pc.equal(tbl2["ref_row"],row))

        # convert so we can get one row in table at a time as pydict
        tbl3_iter = tbl3.to_batches(max_chunksize=1)
        num_rows = tbl3.num_rows



        now_time = time.time()
        if verbose:
            print(f'time {now_time-start:4.1f} sec read and pulled one P/R ({path}/{row}) ({num_rows} nc files) from parquet file {now_time - last_time:6.1f}\n processing {num_rows} pairs for this path/row',flush=True)
        last_time = now_time


        # open s3fs filesystem to read nc files
        s3_fs = s3fs.S3FileSystem(anon=True)




        # iterate over all the nc files for this path/row (or n of them starting at index n_start)
        for entry in tbl3_iter:
            pydict = entry.to_pydict() # convert from parquet row to python dictionary
            if verbose:
                print(f"{pydict['reference']}")
            now_time = time.time()
            if verbose:
                print(f'starting image reads {now_time - start:6.1f}',flush=True)
            last_time = now_time


            try:
                file_found = True
                # open velocity nc file with xarray (first open is s3)
                in_nc_file = s3_fs.open(pydict['v2_url'][0].replace('https','s3').replace('.s3.us-west-2.amazonaws.com',''))
                in_ds = xr.open_dataset(in_nc_file)
                v = in_ds.v.values
            except Exception as e:
                #########################
                #
                # input nc file not found, do what needs to be done in this function (defined above)
                #
                #########################

                output_function_for_nc_file_not_found(pydict,e,v02_v_not_found)
                file_found = False

            if file_found:
                read_v_time = time.time()

                if np.sum(~np.isnan(v)) == 0:

                    #########################
                    #
                    # nc v field is all nans (skip rest), do what needs to be done in this function (defined above)
                    #
                    #########################

                    output_function_for_v_all_nan_values(pydict,out_v02_v_allnan)

                else:

                    # pull values from input itslive granule
                    in_ds_epsg = int(in_ds.mapping.attrs['spatial_epsg'])
                    in_ds_gt = in_ds.mapping.attrs['GeoTransform']
                    res = float(in_ds_gt.split()[1])

                    destination_epsg_str = f'EPSG:{in_ds_epsg}'

                    v_name = pydict['v2_url'][0].split('/')[-1]
                    v1_pvalid = pydict['v1_pvalid'][0]



                    img1_s3_fullpath = pydict['ref_s3_path'][0]

                    # img2_name = pydict['secondary'][0] + "_B8.TIF"
                    # img2_dir = pydict['secondary'][0]
                    img2_s3_fullpath = pydict['sec_s3_path'][0]


                    # ok - now a round-about (but still fast) way of getting to the l8 images in the projection of the nc file, sampled at 120 m pixels to match the nc grid more or less
                    # first open as GeoImg_noload just to get the .gd (gdal.Open, returns a gdal image object)
                    # second make in memory VRTs of these warped to the nc projection and resolution (returned from gdal.Warp as a gd object, no read yet)
                    # third open these as GeoImg_noload_from_gd objects so we can use all the .min_x type fields to find common areas, etc. below
                    #
                    # sorry for the convoluted path, but I won't have to rewrite earlier code this way
                    #
                    # use these to load from s3

                    try:
                        img1_orig = GeoImg_noload_from_gd(gdal.Open(img1_s3_fullpath.replace('s3://','/vsis3/')))
                        img2_orig = GeoImg_noload_from_gd(gdal.Open(img2_s3_fullpath.replace('s3://','/vsis3/')))
                    except Exception as e:
                        print(f"image open from S3 exception {e} - >>>> skipping")
                        continue # should drop out the rest of this iteration, but loop continues...

                    test_1 = img1_orig.gd.GetSpatialRef().GetAuthorityCode('PROJCS')
                    # Landsat 3031 images don't report the projection code properly, but gdal/osr can find it...
                    if not(test_1):
                        a = img1_orig.gd.GetSpatialRef()
                        b = a.FindMatches()[0][0]
                        test_1 = b.GetAuthorityCode('PROJCS')
                    img1_orig_epsg = int(test_1)

#                     img2_orig = GeoImg_noload_from_gd(gdal.Open(img2_s3_fullpath.replace('s3://','/vsis3/')))
                    test_2 = img2_orig.gd.GetSpatialRef().GetAuthorityCode('PROJCS')
                    if not(test_2):
                        a = img2_orig.gd.GetSpatialRef()
                        b = a.FindMatches()[0][0]
                        test_2 = b.GetAuthorityCode('PROJCS')
                    img2_orig_epsg = int(test_2)


                    if (in_ds_epsg == 3413) or (img1_orig_epsg != img2_orig_epsg): # if northern NSIDC PS, or cross-path/rows from different UTM, warp Landsats to nc file projection
                        l8_1_gd = gdal.Warp('', img1_orig.gd, dstSRS=destination_epsg_str, format='VRT',
                                       outputType=gdal.GDT_Float32, xRes=res, yRes=res,
                        #                outputBounds=outputBounds, dstNodata=np.nan,
                                       resampleAlg='nearest',multithread=True)
                        l8_2_gd = gdal.Warp('', img2_orig.gd, dstSRS=destination_epsg_str, format='VRT',
                                       outputType=gdal.GDT_Float32, xRes=res, yRes=res,
                        #                outputBounds=outputBounds, dstNodata=np.nan,
                                       resampleAlg='nearest',multithread=True)
                    else: # all other projections are the same as original Landsat images, no warping needed
                        l8_1_gd = gdal.Translate('', img1_orig.gd, format='VRT',
                                       outputType=gdal.GDT_Float32, xRes=res, yRes=res,
                        #                outputBounds=outputBounds, dstNodata=np.nan,
                                       resampleAlg='nearest')
                        l8_2_gd = gdal.Translate('', img2_orig.gd, format='VRT',
                                       outputType=gdal.GDT_Float32, xRes=res, yRes=res,
                        #                outputBounds=outputBounds, dstNodata=np.nan,
                                       resampleAlg='nearest')

                    # now open these resampled in-memory images
                    img1 = GeoImg_noload_from_gd(l8_1_gd)


                    # l8_1_img = l8_1_gd.ReadAsArray()


                    img2 = GeoImg_noload_from_gd(l8_2_gd)


                    # l8_2_img = l8_2_gd.ReadAsArray()



                    # fig,ax = plt.subplots(1,1,figsize=[16,10])
                    i1_arr = img1.gd.ReadAsArray()
                    read_img1_time = time.time()

                    i2_arr = img2.gd.ReadAsArray()
                    read_img2_time = time.time()


                    # this is all for using the its-live land masks that are on S3
                    midx = np.mean([img1.min_x,img1.max_x])
                    midy = np.mean([img1.min_y,img1.max_y])
                    transformer = pyproj.Transformer.from_crs(img1.proj,'EPSG:4326',always_xy=True)
                    lon,lat = transformer.transform(midx,midy)
                    maskfullpath,maskepsg = return_its_live_land_mask_URL_and_epsg(lat=lat,lon=lon, itslive_zones=itslive_zones, verbose=verbose)
                    got_mask_time = time.time()

                    splitpath = maskfullpath.split('/')
                    maskdir = '/vsicurl/' + '/'.join(splitpath[:-1])
                    maskfile = splitpath[-1]

            ###############
            # now find common area bounding box for Landsat images in the nc file projection (at 120 m resolution) and get mask for that box
            # code here is taken from pycorr, so it has buffers and stuff like that still in it.
            ###############

                    com_max_x=np.min([img1.max_x, img2.max_x])
                    com_max_y=np.min([img1.max_y, img2.max_y])
                    com_min_x=np.max([img1.min_x, img2.min_x])
                    com_min_y=np.max([img1.min_y, img2.min_y])

                    bbox=[com_min_x, com_min_y, com_max_x, com_max_y]
                    # use this box to find overlap range in i and j (array coordinates) for both images
                    com_ul_i_j_img1=img1.XY2imageij(*(com_min_x,com_max_y))+0.5
                    com_ul_i_j_img2=img2.XY2imageij(*(com_min_x,com_max_y))+0.5
                    com_lr_i_j_img1=img1.XY2imageij(*(com_max_x,com_min_y))-0.5
                    com_lr_i_j_img2=img2.XY2imageij(*(com_max_x,com_min_y))-0.5


                    c1_minj=int(com_ul_i_j_img1[1])
                    c1_maxjp1=int(com_lr_i_j_img1[1]+1) # may be one more than the max index, for use in slices (p1 means plus 1...)
                    c1_mini=int(com_ul_i_j_img1[0])
                    c1_maxip1=int(com_lr_i_j_img1[0]+1) # same

                    c2_minj=int(com_ul_i_j_img2[1])
                    c2_maxjp1=int(com_lr_i_j_img2[1]+1) # same
                    c2_mini=int(com_ul_i_j_img2[0])
                    c2_maxip1=int(com_lr_i_j_img2[0]+1) # same


                    com_num_pix_i=c1_maxip1-c1_mini  # number of pixels per line in common (overlap) area
                    com_num_pix_j=c1_maxjp1-c1_minj  # number of lines in common (overlap) area

    #                 if verbose:
    #                     print('numpix %f numlines %f in box'%(com_num_pix_i,com_num_pix_j))
    #                     print('ul X %f Y %f of box'%(com_min_x,com_max_y))
    #                     print('ul image1 i %f j%f  image2 i %f j%f'%(c1_mini,c1_minj,c2_mini,c2_minj))
    #                     print('lr X %f Y %f of box'%(com_max_x,com_min_y))
    #                     print('lr image1 i %f j%f  image2 i %f j%f'%(c1_maxip1,c1_maxjp1,c2_maxip1,c2_maxjp1))

                    inc = 8 # 120 m (120 m / 15 m/pixel)
                    half_target_chip = 10
                    half_source_chip = 5

                    half_inc_rim = 0  # inc was checked to be even (above) so int is redundant but included for clarity - want integer-width rim...

                    num_grid_i=int(com_num_pix_i/inc)       # these are integer divisions...
                    num_grid_j=int(com_num_pix_j/inc)

                    ul_i_1_chip_grid=int(c1_mini + half_inc_rim)
                    ul_j_1_chip_grid=int(c1_minj + half_inc_rim)
                    lr_i_1_chip_grid=int(c1_mini + half_inc_rim + ((num_grid_i-1) * inc))
                    lr_j_1_chip_grid=int(c1_minj + half_inc_rim + ((num_grid_j-1) * inc))

                    ul_i_2_chip_grid=int(c2_mini + half_inc_rim)
                    ul_j_2_chip_grid=int(c2_minj + half_inc_rim)
                    lr_i_2_chip_grid=int(c2_mini + half_inc_rim + ((num_grid_i-1) * inc))
                    lr_j_2_chip_grid=int(c2_minj + half_inc_rim + ((num_grid_j-1) * inc))

                    r_i_1=range(ul_i_1_chip_grid,lr_i_1_chip_grid+1,inc) # range over common area
                    r_j_1=range(ul_j_1_chip_grid,lr_j_1_chip_grid+1,inc)

                    r_i_2=range(ul_i_2_chip_grid,lr_i_2_chip_grid+1,inc) # range over common area
                    r_j_2=range(ul_j_2_chip_grid,lr_j_2_chip_grid+1,inc)
                    min_ind_i_1=np.min(np.nonzero(np.array(r_i_1)>half_target_chip))# this is width of the rim of zeros on left or top in the correlation arrays
                    min_ind_j_1=np.min(np.nonzero(np.array(r_j_1)>half_target_chip))# this is width of the rim of zeros on left or top in the correlation arrays
                    min_ind_i_2=np.min(np.nonzero(np.array(r_i_2)>half_target_chip))# this is width of the rim of zeros on left or top in the correlation arrays
                    min_ind_j_2=np.min(np.nonzero(np.array(r_j_2)>half_target_chip))# this is width of the rim of zeros on left or top in the correlation arrays
                    min_ind_i = np.max([min_ind_i_1,min_ind_i_2])
                    min_ind_j = np.max([min_ind_j_1,min_ind_j_2])

                    max_ind_i_1=np.max(np.nonzero(half_target_chip<=(img1.num_pix_x - np.array(r_i_1))))  # maximum allowed i index (buffer right edge of grid to accomodate target chip size)
                    max_ind_j_1=np.max(np.nonzero(half_target_chip<=(img1.num_pix_y - np.array(r_j_1))))  # maximum allowed i index (buffer bottom edge of grid to accomodate target chip size)
                    max_ind_i_2=np.max(np.nonzero(half_target_chip<=(img2.num_pix_x - np.array(r_i_2))))  # maximum allowed i index (buffer right edge of grid to accomodate target chip size)
                    max_ind_j_2=np.max(np.nonzero(half_target_chip<=(img2.num_pix_y - np.array(r_j_2))))  # maximum allowed i index (buffer bottom edge of grid to accomodate target chip size)
                    max_ind_i=np.min([max_ind_i_1,max_ind_i_2])  # maximum allowed i index (buffer right edge of grid to accomodate target chip size)
                    max_ind_j=np.min([max_ind_j_1,max_ind_j_2])  # maximum allowed i index (buffer bottom edge of grid to accomodate target chip



                    # the -0.5 in what follows shifts from the ij of the element to it's upper left corner (chip center)
                    r_grid=np.meshgrid(np.array(r_i_1)-0.5,np.array(r_j_1)-0.5,indexing='xy')
                    chip_center_grid_xy=img1.imageij2XY(*r_grid)
                    # note output_array_ul_corner will be the same as [com_min_x, com_max_y] but is derived here just to check coordinates
                    output_array_ul_corner=chip_center_grid_xy[:,0,0]-((inc/2.0)*img1.pix_x_m,(inc/2.0)*img1.pix_y_m)
                    output_array_pix_x_m=inc*img1.pix_x_m
                    output_array_pix_y_m=inc*img1.pix_y_m   # note this will nearly always be negative, for image stored top line first, just as in world file
                    output_array_num_pix_x=r_i_2.__len__()
                    output_array_num_pix_y=r_j_2.__len__()

                    try:
                        tmp_lgo_orig=GeoImg_noload(maskfile,maskdir)
                    except Exception as e:
                        print(f"mask file open exception {e} - >>>> skipping")
                        continue # should drop out the rest of this iteration, but loop continues...

                    outputBounds = (com_min_x, com_min_y, com_max_x, com_max_y)

                    lgo_mask_cropped_gd = gdal.Warp('', tmp_lgo_orig.gd, dstSRS=destination_epsg_str, format='VRT',
                                   outputType=gdal.GDT_Float32, xRes=res, yRes=res,
                                   outputBounds=outputBounds,
                    #                dstNodata=np.nan,
                                   resampleAlg='nearest',multithread=True)

            #         if in_ds_epsg == 3413: # if northern NSIDC PS, warp Landsats to nc file projection
            #         else: # no reprojection needed
            #             #             lgo_mask_cropped_gd = gdal.Translate('', tmp_lgo_orig.gd, format='VRT',
            #             #                            outputType=gdal.GDT_Float32, xRes=res, yRes=res,
            #             #                            outputBounds=outputBounds,
            #             #             #                dstNodata=np.nan,
            #             #                            resampleAlg='nearest')
            #             lgo_mask_cropped_gd = gdal.Translate('', tmp_lgo_orig.gd, format='VRT',
            #                            outputType=gdal.GDT_Float32,
            #                            # xRes=res, yRes=res,
            #                            outputBounds=outputBounds,
            #             #                dstNodata=np.nan,
            #                            resampleAlg='nearest')


                    lgo_mask_cropped_geoimg = GeoImg_noload_from_gd(lgo_mask_cropped_gd)

                    lgo_mask_image_local=lgo_mask_cropped_geoimg.gd.ReadAsArray().astype(np.byte)
                    read_mask_time = time.time()





                    c1_minj=int(com_ul_i_j_img1[1])
                    c1_maxjp1=int(com_lr_i_j_img1[1]) # may be one more than the max index, for use in slices (p1 means plus 1...)
                    c1_mini=int(com_ul_i_j_img1[0])
                    c1_maxip1=int(com_lr_i_j_img1[0]) # same

                    c2_minj=int(com_ul_i_j_img2[1])
                    c2_maxjp1=int(com_lr_i_j_img2[1]) # same
                    c2_mini=int(com_ul_i_j_img2[0])
                    c2_maxip1=int(com_lr_i_j_img2[0]) # same

                    # now crop the max end if one is bigger (for either dimension)
                    s1j,s1i = c1_maxjp1 - c1_minj,c1_maxip1 - c1_mini
                    s2j,s2i = c2_maxjp1 - c2_minj,c2_maxip1 - c2_mini

                    if s1j > s2j:
                        c1_maxjp1 -= (s1j - s2j)
                    elif s1j < s2j:
                        c2_maxjp1 -= (s2j - s1j)

                    if s1i > s2i:
                        c1_maxip1 -= (s1i - s2i)
                    elif s1i < s2i:
                        c2_maxip1 -= (s2i - s1i)

                    commonarea_img1 = i1_arr[c1_minj:c1_maxjp1,c1_mini:c1_maxip1]
                    commonarea_img2 = i2_arr[c2_minj:c2_maxjp1,c2_mini:c2_maxip1]


            #             # now find all nonzero pix in both images  -  nonzero(img1 or img2)
            #             combined_i1i2_or = commonarea_img1 + commonarea_img2
            #             reduced_testarr_or = combined_i1i2_or != 0

                    # now find all pixels that are nonzero in both images  -  nonzero(img1 and img2)
                    combined_i1i2_and = np.zeros_like(commonarea_img1)
                    combined_i1i2_and[commonarea_img2!=0] = commonarea_img1[commonarea_img2!=0]
                    reduced_testarr_and = combined_i1i2_and != 0




                    if lgo_mask_image_local.shape != reduced_testarr_and.shape:
                        lgo_max_j,lgo_max_i = lgo_mask_image_local.shape
                        rtarr_max_j,rtarr_max_i = reduced_testarr_and.shape
                        max_j = min([lgo_max_j, rtarr_max_j])
                        max_i = min([lgo_max_i, rtarr_max_i])
                        new_lgo_mask_image_local = lgo_mask_image_local[:max_j,:max_i]
            #                 new_rtarr_or= reduced_testarr_or[:max_j,:max_i]
                        new_rtarr_and= reduced_testarr_and[:max_j,:max_i]
                    else:
                        new_lgo_mask_image_local = lgo_mask_image_local
            #                 new_rtarr_or = reduced_testarr_or
                        new_rtarr_and = reduced_testarr_and

                    if np.count_nonzero(new_lgo_mask_image_local) == 0:
                        output_function_for_ROI_mask_all_zero(pydict,out_ROI_allnan)

                    elif np.count_nonzero(new_rtarr_and[new_lgo_mask_image_local==1]) == 0:
                        output_function_for_common_image_areas_in_ROI_all_zero(pydict)

                    else: # keep going

                        pctvalid_mod_and= np.count_nonzero(~(np.isnan(v)))/np.count_nonzero(new_rtarr_and[new_lgo_mask_image_local==1])

                        #########################
                        #
                        # new p_valid found, do what needs to be done in this function (defined above)
                        #
                        #########################

                        output_function_for_new_pvalid_found(pydict,pctvalid_mod_and,out_pvalid_fix)
            #             print(f'{v_name} ==>  new_pvalid {pctvalid_mod_and*100.1:3.1f}  (v1_pvalid {v1_pvalid})')


    #                     if verbose:
    #                         zredtestarr_and = np.zeros_like(new_rtarr_and)
    #                         zredtestarr_and[new_lgo_mask_image_local==1] = new_rtarr_and[new_lgo_mask_image_local==1]
    #
    #             #                 zredtestarr_or = np.zeros_like(new_rtarr_or)
    #             #                 zredtestarr_or[new_lgo_mask_image_local==1] = new_rtarr_or[new_lgo_mask_image_local==1]
    #
    #
    #                         fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=[16,10])
    #                         ax3.imshow(zredtestarr_and,cmap='jet')
    #             #                 ax3.imshow(zredtestarr_or,cmap='jet')
    #                         ax2.imshow(lgo_mask_image_local,cmap='jet',vmin = 0.0, vmax = 1.0)
    #                         ax1.imshow(v)
    #
    #                         ax1.set_title('v')
    #                         ax2.set_title('ROI mask')
    #             #                 ax3.set_title('valid pixel in either (or)')
    #                         ax3.set_title('valid pixels in both (and)')
    #
    #             #                 plt.suptitle(f'{v_name} ==>  or: {pctvalid_mod_or*100.1:3.1f}  and: {pctvalid_mod_and*100.1:3.1f} new_pvalid (v1_pvalid {v1_pvalid})')
    #                         plt.suptitle(f'{v_name} ==>  new_pvalid {pctvalid_mod_and*100.1:3.1f}  (v1_pvalid {v1_pvalid})')
    #
                    now_time = time.time()
                    if verbose:
                        print(f'total_time: {now_time - start:6.1f} seconds, del_time: {now_time - last_time:6.1f}')
                        print(f'v read: {read_v_time - last_time:6.1f} seconds')
                        print(f'img1 read: {read_img1_time - read_v_time:6.1f} seconds')
                        print(f'img2 read: {read_img2_time - read_img1_time:6.1f} seconds')
                        print(f'got_mask_time: {got_mask_time - read_img2_time:6.1f} seconds')
                        print(f'read_mask_time: {read_mask_time - got_mask_time:6.1f} seconds')

                    in_ds = None
                    in_nc_file = None
                    l8_1_gd = None
                    l8_2_gd = None
                    img1_orig = None
                    img2_orig = None
                    lgo_mask_cropped_geoimg = None
                    lgo_mask_cropped_gd = None

                    last_time = now_time


        now_time = time.time()
        print(f'P/R time took {now_time - pr_starttime:6.1f} seconds, average {(now_time - pr_starttime)/num_rows:6.1f} seconds',flush=True)

    out_pvalid_fix.close()
    out_v02_v_allnan.close()
    out_ROI_allnan.close()
    v02_v_not_found.close()


now_time = time.time()
print(f'full time took {now_time - start:6.1f} seconds')

