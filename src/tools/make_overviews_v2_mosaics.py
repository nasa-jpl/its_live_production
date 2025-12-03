import boto3
import subprocess as sp
from tqdm import tqdm


# from https://stackoverflow.com/questions/30249069/listing-contents-of-a-bucket-with-boto3
def keys(bucket_name, prefix='/', delimiter='/', start_after=''):
    prefix = prefix.lstrip(delimiter)
    start_after = (start_after or prefix) if prefix.endswith(delimiter) else start_after
    for page in s3_paginator.paginate(Bucket=bucket_name, Prefix=prefix, StartAfter=start_after):
        for content in page.get('Contents', ()):
            yield content['Key']

bucketname = 'its-live-data'

s3_paginator = boto3.client('s3').get_paginator('list_objects_v2')

k = keys(bucketname,prefix='velocity_mosaic/v2/annual/cog')
all_v2_annual = [x for x in k if '_v.tif' in x]

in_list = all_v2_annual

out_list = [x.rsplit('/',maxsplit=1)[-1] for x in in_list]

# for file in tmp:
#     cmd = f"aws s3 cp s3://its-live-data/{file} {file.rsplit('/',maxsplit=1)[-1]}"
#     print(cmd)


tmp = [x for x in in_list if '_2022_' in x]

rgizones = [x.split('_')[-4] for x in tmp]

zonedict = {x:{} for x in rgizones}

for file in tqdm(tmp):
    zone = file.split('_')[-4]
    cmd = f"gdalinfo --config GDAL_DISABLE_READDIR_ON_OPEN EMPTY_DIR https://its-live-data.s3.amazonaws.com/{file}"
    run_out = sp.run(cmd.split(),capture_output=True)
    op = [x for x in run_out.stdout.split(b'\n') if b'Overviews' in x]
    b = op[0].decode()
    zonedict[zone]['num_ovrs'] = len(b.split()) - 2 # one for "Overviews:", 1 because overviews are numbered from 0
    zonedict[zone]['overview_dimstr'] = b.split()[-1]


# had to hardwire this dict because the following was failing (all other 2022 zones worked ok)
# gdalinfo https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2/annual/cog/ITS_LIVE_velocity_120m_RGI19A_2022_v02_v.tif

# note this only runs for zone RGI19A - because that fails in the commented out loop above
cmd = "gdalinfo ../ITS_LIVE_velocity_120m_RGI19A_0000_v02_v.tif"
run_out = sp.run(cmd.split(),capture_output=True)
op = [x for x in run_out.stdout.split(b'\n') if b'Overviews' in x]
b = op[0].decode()
zonedict[zone]['num_ovrs'] = len(b.split()) - 2 # one for "Overviews:", 1 because overviews are numbered from 0
zonedict[zone]['overview_dimstr'] = b.split()[-1]


# zonedict = {
#             'RGI01A': {'num_ovrs': 5, 'overview_dimstr': '156x325'},
#             'RGI02A': {'num_ovrs': 4, 'overview_dimstr': '289x294'},
#             'RGI03A': {'num_ovrs': 4, 'overview_dimstr': '260x208'},
#             'RGI04A': {'num_ovrs': 4, 'overview_dimstr': '130x416'},
#             'RGI05A': {'num_ovrs': 5, 'overview_dimstr': '208x364'},
#             'RGI06A': {'num_ovrs': 3, 'overview_dimstr': '312x260'},
#             'RGI07A': {'num_ovrs': 3, 'overview_dimstr': '208x312'},
#             'RGI08A': {'num_ovrs': 4, 'overview_dimstr': '208x390'},
#             'RGI09A': {'num_ovrs': 4, 'overview_dimstr': '364x338'},
#             'RGI10A': {'num_ovrs': 3, 'overview_dimstr': '398x380'},
#             'RGI11A': {'num_ovrs': 3, 'overview_dimstr': '398x269'},
#             'RGI12A': {'num_ovrs': 3, 'overview_dimstr': '361x165'},
#             'RGI14A': {'num_ovrs': 5, 'overview_dimstr': '421x278'},
#             'RGI17A': {'num_ovrs': 4, 'overview_dimstr': '158x450'},
#             'RGI18A': {'num_ovrs': 3, 'overview_dimstr': '301x265'},
#             'RGI19A': {'num_ovrs': 6, 'overview_dimstr': '358x299'}
#             }
# 




# for file in tmp:
#     cmd = f"aws s3 cp s3://its-live-data/{file} {file.rsplit('/',maxsplit=1)[-1]}"
#     print(cmd)
#     
# 
# cmd = 'gdalinfo https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2/annual/cog/ITS_LIVE_velocity_120m_RGI01A_2022_v02_v.tif'
# 
# run_out = sp.run(cmd.split(),capture_output=True)
# op = [x for x in run_out.stdout.split(b'\n') if b'Overviews' in x]
# b = op[0].decode()
# len(b.split())
# 
# cmd2 = 'gdal_translate -ovr 6 https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2/annual/cog/ITS_LIVE_velocity_120m_RGI01A_2022_v02_v.tif ITS_LIVE_velocity_120m_RGI01A_2022_v02_v.tif'
# 


zone='RGI01A'
zoneinfo = zonedict[zone]
infiles = [x for x in out_list if zone in x]

for file in tqdm(infiles,desc=f'fetching {zone}'):
    cmd2 = f"gdal_translate --config GDAL_DISABLE_READDIR_ON_OPEN EMPTY_DIR -ovr {zoneinfo['num_ovrs']} https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2/annual/cog/{file} {file.replace('_v02_v','_v02_v_browse')}"
    run_out2 = sp.run(cmd2.split(),capture_output=False)

RGIzone="RGI05A"
basefile="ITS_LIVE_velocity_120m_"$RGIzone"_2020_v02_v_browse.tif"
outfile="land_ice_blank_"$RGIzone".tif"
gdal_create -if $basefile $outfile
gdal_rasterize -burn -10.0 -l ne_10m_land ../land_shapefiles/ne_10m_land/ne_10m_land.shp $outfile
gdal_rasterize -burn -5.0 -l ne_10m_glaciated_areas ../land_shapefiles/ne_10m_glaciated_areas/ne_10m_glaciated_areas.shp $outfile
gdaldem color-relief $outfile ../color_tables/velocity_slow_default_with_grey_land_ice_ocean_for_gdal.txt ${outfile/tif/png} -of PNG
#  1029  ls
#  1030  gdal_merge.py
#  1031  gdal_merge.py -o junk2.tif junk.tif ITS_LIVE_velocity_120m_RGI05A_2020_v02_v_browse.tif.tif
# 
# 
# GDAL_DISABLE_READDIR_ON_OPEN EMPTY_DIR

RGIzone="RGI03A"
noveriews="4"
basefile="../production/ITS_LIVE_velocity_120m_"$RGIzone"_0000_v02_v_browse.tif"
outfile="land_ice_blank_"$RGIzone".tif"
gdal_translate --config GDAL_DISABLE_READDIR_ON_OPEN EMPTY_DIR -ovr $noveriews ${basefile/_v02_v_browse/_v02_v} $basefile
gdal_create -if -burn -20.0 $basefile $outfile
gdal_rasterize -burn -10.0 -l ne_10m_land ../land_shapefiles/ne_10m_land/ne_10m_land.shp $outfile
gdal_rasterize -burn -5.0 -l ne_10m_glaciated_areas ../land_shapefiles/ne_10m_glaciated_areas/ne_10m_glaciated_areas.shp $outfile
gdaldem color-relief $outfile ../color_tables/velocity_slow_default_with_grey_land_ice_ocean_for_gdal.txt ${outfile/tif/png} -of PNG




# gdal_translate --config GDAL_DISABLE_READDIR_ON_OPEN EMPTY_DIR -ovr {zoneinfo['num_ovrs']} https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2/annual/cog/{file} {file.replace('_v02_v','_v02_v_browse')}
# 
# 









# from https://stackoverflow.com/questions/30249069/listing-contents-of-a-bucket-with-boto3
def keys(bucket_name, prefix='/', delimiter='/', start_after=''):
    prefix = prefix.lstrip(delimiter)
    start_after = (start_after or prefix) if prefix.endswith(delimiter) else start_after
    for page in s3_paginator.paginate(Bucket=bucket_name, Prefix=prefix, StartAfter=start_after):
        for content in page.get('Contents', ()):
            yield content['Key']

bucketname = 'its-live-data'

s3_paginator = boto3.client('s3').get_paginator('list_objects_v2')

k = keys(bucketname,prefix='velocity_mosaic/v2/static/cog')
all_v2_static = [x for x in k if '_v.tif' in x]

in_list = all_v2_static

out_list = [x.rsplit('/',maxsplit=1)[-1] for x in in_list]

# for file in tmp:
#     cmd = f"aws s3 cp s3://its-live-data/{file} {file.rsplit('/',maxsplit=1)[-1]}"
#     print(cmd)


# tmp = [x for x in in_list if '_2022_' in x]
tmp = in_list

rgizones = [x.split('_')[-4] for x in tmp]

zonedict = {x:{} for x in rgizones}

for file in tqdm(tmp):
    zone = file.split('_')[-4]
    cmd = f"gdalinfo --config GDAL_DISABLE_READDIR_ON_OPEN EMPTY_DIR https://its-live-data.s3.amazonaws.com/{file}"
    run_out = sp.run(cmd.split(),capture_output=True)
    op = [x for x in run_out.stdout.split(b'\n') if b'Overviews' in x]
    b = op[0].decode()
    zonedict[zone]['num_ovrs'] = len(b.split()) - 2 # one for "Overviews:", 1 because overviews are numbered from 0
    zonedict[zone]['overview_dimstr'] = b.split()[-1]


# had to hardwire this dict because the following was failing (all other 2022 zones worked ok)
# gdalinfo https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2/annual/cog/ITS_LIVE_velocity_120m_RGI19A_2022_v02_v.tif

# note this only runs for zone RGI19A - because that fails in the commented out loop above
cmd = "gdalinfo ../ITS_LIVE_velocity_120m_RGI19A_0000_v02_v.tif"
run_out = sp.run(cmd.split(),capture_output=True)
op = [x for x in run_out.stdout.split(b'\n') if b'Overviews' in x]
b = op[0].decode()
zonedict[zone]['num_ovrs'] = len(b.split()) - 2 # one for "Overviews:", 1 because overviews are numbered from 0
zonedict[zone]['overview_dimstr'] = b.split()[-1]


for zone in tqdm(zonedict.keys(),desc='fetching overviews from aws'):
    zoneinfo = zonedict[zone]
    infiles = [x for x in out_list if zone in x]
    
    for file in tqdm(infiles,desc=f'fetching {zone}'):
        cmd2 = f"gdal_translate --config GDAL_DISABLE_READDIR_ON_OPEN EMPTY_DIR -ovr {zoneinfo['num_ovrs']} https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2/static/cog/{file} {file.replace('_v02_v','_v02_v_browse')}"
        run_out2 = sp.run(cmd2.split(),capture_output=False)

# same issue - ran local: gdal_translate --config GDAL_DISABLE_READDIR_ON_OPEN EMPTY_DIR -ovr 6 ../../production/static/cog/ITS_LIVE_velocity_120m_RGI19A_0000_v02_v.tif ITS_LIVE_velocity_120m_RGI19A_0000_v02_v_browse.tif

# shape_home_dir = '../../land_shapefiles'
# RGIzone="RGI02A"
# noveriews="4"
# basefile="./ITS_LIVE_velocity_120m_"$RGIzone"_0000_v02_v_browse.tif"
# outfile="land_ice_blank_"$RGIzone".tif"
# gdal_translate --config GDAL_DISABLE_READDIR_ON_OPEN EMPTY_DIR -ovr $noveriews ${basefile/_v02_v_browse/_v02_v} $basefile
# gdal_create -burn -20.0 -if $basefile $outfile
# gdal_rasterize -burn -10.0 -l ne_10m_land $shape_home_dir/ne_10m_land/ne_10m_land.shp $outfile
# gdal_rasterize -burn -5.0 -l ne_10m_glaciated_areas $shape_home_dir/ne_10m_antarctic_ice_shelves_polys/ne_10m_antarctic_ice_shelves_polys.shp $outfile
# if [ ]$RGIzone -eq "RGI19A" ]; then gdal_rasterize -burn -7.0 -l ne_10m_glaciated_areas $shape_home_dir/ne_10m_glaciated_areas/ne_10m_  $outfile; fi
# gdaldem color-relief $outfile ../color_tables/velocity_slow_default_with_grey_land_ice_ocean_for_gdal.txt ${outfile/tif/png} -of PNG


# 
# shape_home_dir = '../../land_shapefiles'
# color_tables_dir = '../../color_tables'
# RGIzone="RGI03A"
# cmd_list = []
# noveriews=zonedict[RGIzone]['num_ovrs']
# basefile=f"./ITS_LIVE_velocity_120m_{RGIzone}_0000_v02_v_browse.tif"
# outfile=f"land_ice_blank_{RGIzone}.tif"
# # cmd_list.append(f"gdal_translate --config GDAL_DISABLE_READDIR_ON_OPEN EMPTY_DIR -ovr {noveriews} {basefile.replace('_v02_v_browse','_v02_v')} {basefile}")
# cmd_list.append(f"gdal_create -burn -20.0 -if {basefile} {outfile}")
# cmd_list.append(f'gdal_rasterize -burn -10.0 -co "OGR_ENABLE_PARTIAL_REPROJECTION=TRUE" -l ne_10m_land {shape_home_dir}/ne_10m_land/ne_10m_land.shp {outfile}')
# cmd_list.append(f'gdal_rasterize -burn -5.0 -co "OGR_ENABLE_PARTIAL_REPROJECTION=TRUE" -l ne_10m_glaciated_areas {shape_home_dir}/ne_10m_glaciated_areas/ne_10m_glaciated_areas.shp {outfile}')
# if RGIzone == "RGI19A":
#     cmd_list.append(f"gdal_rasterize -burn -7.0 -configure OGR_ENABLE_PARTIAL_REPROJECTION TRUE -l ne_10m_antarctic_ice_shelves_polys {shape_home_dir}/ne_10m_antarctic_ice_shelves_polys/ne_10m_antarctic_ice_shelves_polys.shp {outfile}")
# 
# cmd_list.append(f"gdaldem color-relief {outfile} {color_tables_dir}/velocity_slow_default_with_grey_land_ice_ocean_for_gdal.txt {outfile.replace('tif','png')} -of PNG")
# 
# 
# for cmd in cmd_list:
#     run_out2 = sp.run(cmd.split(),capture_output=False)
# 
# 
# 
# 
# 
# 
# 
# # -clipsrc [<xmin> <ymin> <xmax> <ymax>]
# 



shape_home_dir = '../../land_shapefiles'
color_tables_dir = '../../color_tables'

# the following zones need cropped versions of the ne_10m_land shapefile to render properly
# 11 also has the same for glaciated areas
special_landshape_zones = ["02","10","11","12","14","17","18"] # have custom cropped land shapefiles

for RGIzone in zonedict.keys():
    print(f"working on {RGIzone}",flush=True)
    glacier_shape_name = "ne_10m_glaciated_areas"
    if RGIzone[3:5] in special_landshape_zones:
        land_shape_name = f"ne_10m_landc{RGIzone[3:5]}"
        if RGIzone[3:5] == "11":
            glacier_shape_name  = "ne_10m_glaciated_areasc11"
    else:
        land_shape_name = "ne_10m_land"
    cmd_list = []
    noveriews=zonedict[RGIzone]['num_ovrs']
    basefile=f"./ITS_LIVE_velocity_120m_{RGIzone}_0000_v02_v_browse.tif"
    outfile=f"land_ice_blank_{RGIzone}.tif"
    # cmd_list.append(f"gdal_translate --config GDAL_DISABLE_READDIR_ON_OPEN EMPTY_DIR -ovr {noveriews} {basefile.replace('_v02_v_browse','_v02_v')} {basefile}")
    cmd_list.append(f"gdal_create -burn -20.0 -if {basefile} {outfile}")
    cmd_list.append(f'gdal_rasterize -burn -10.0 -l {land_shape_name} {shape_home_dir}/ne_10m_land/{land_shape_name}.shp {outfile}')
    cmd_list.append(f'gdal_rasterize -burn -5.0 -l {glacier_shape_name} {shape_home_dir}/ne_10m_glaciated_areas/{glacier_shape_name}.shp {outfile}')
    if RGIzone == "RGI19A":
        cmd_list.append(f"gdal_rasterize -burn -7.0 -l ne_10m_antarctic_ice_shelves_polys {shape_home_dir}/ne_10m_antarctic_ice_shelves_polys/ne_10m_antarctic_ice_shelves_polys.shp {outfile}")
    
    cmd_list.append(f"gdaldem color-relief {outfile} {color_tables_dir}/velocity_slow_default_with_grey_land_ice_ocean_for_gdal.txt {outfile.replace('tif','png')} -of PNG")
    
    
    for cmd in cmd_list:
        run_out2 = sp.run(cmd.split(),capture_output=False)