#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reads image pair list (directory listing from itslive S3 zone produced from
make_filelists_by_zone.py (s3fs ls)), finds all USGS L8 image pairs from the same 
path/rows (from current USGS C2 image list - needs to be updated regularly), 
then finds missing pairs that have both images in USGS C2 collection

Outputs a file for the zone that is the appropriate scene ids for the pair,
separated by a space

v2 sorts images by date to ensure ordered comparison for pairs

@author: mark
"""

import csv
import json
import datetime as dt
import numpy as np
import argparse
import pathlib
import glob

import s3fs

import requests
import gzip


def img_short_name(full_name):
    """ returns first four fields of name connected with _"""
    return('_'.join(full_name.split('_')[:4]))
    



parser = argparse.ArgumentParser(
            description = __doc__,
#             prog = 'find_all_USGS_Collection2_imagepairs_for_zone_list_v2.py',
            formatter_class=argparse.RawDescriptionHelpFormatter
            )
         
# parser.add_argument('zone', help='original ITS_LIVE zone to find new image pairs for (e.g. 32622) (UTM 326XX and 327XX, ?3413 for GRE, 3031 for ANT)')
parser.add_argument('-startdatestr', default='20130101',help='oldest date considered for more recent image in a pair (do not process earlier missing pairs)(format is 0 padded YYYYMMDD) [%(default)s]')
# >>>> default for stopdatestr should be a future date to process up to present <<<<
parser.add_argument('-stopdatestr', default='20291231',help='youngest date considered for more recent image in a pair (do not process later missing pairs)(format is 0 padded YYYYMMDD) [%(default)s]')
parser.add_argument('-USGS_list_file', default='LANDSAT_OT_C2_L1.csv.gz',help='USGS csv file of Collection2 L8 images - if not current (<2 days old), code will download current from  https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_OT_C2_L1.csv.gz [%(default)s]')
parser.add_argument('-max_time_separation_days', default=544, type=int, help='maximum time separation for image pair in days [%(default)d]')
parser.add_argument('-cloudfraction_cutoff', default=60.0, type=float, help='maximum cloud percent to include scene [%(default)3.1f]')
parser.add_argument('-read_geojsons_from_itslive_S3', action='store_true',help='read the geojson catalog files from S3, rathther than local directory [False]')
parser.add_argument('-geojson_cat_dir', default='/Users/mark/itslive/catalog_geojson/landsat/v02',help='directory with geojson catalogs (synced with same dir on itslive AWS) [%(default)s]')
parser.add_argument('-ignore_existing_itslive_pairs', action='store_true',help='produce complete list of possible pairs, ignoring already existing itslive pairs [False]')
parser.add_argument('-v', action='store_true',help='verbose output about existing pairs [False]')
parser.add_argument('-log_numbers_file', default=None, help='log file to append existing and new numbers to [%(default)s]')
args = parser.parse_args()	

if not(args.read_geojsons_from_itslive_S3):
    geojson_cat_dir = args.geojson_cat_dir
    injsonlist = glob.glob(f'{geojson_cat_dir}/imgpair*.json')
else:
    s3_path = 'its-live-data.jpl.nasa.gov/catalog_geojson/landsat/v02/'
    s3 = s3fs.S3FileSystem(anon=True)
    injsonlist = s3.glob(s3_path+'imgpair_v02*.json')


cloudfraction_cutoff = args.cloudfraction_cutoff # percent cloud cover over land
max_time_separation_days = args.max_time_separation_days

if args.log_numbers_file:
    logfile = open(args.log_numbers_file, 'a')

# zone = args.zone
# if zone[:3] == '326' or zone[:3] == '327':
#     if zone[-2] == '0':
#         UTM_zonestr = f'{zone[-1:]}.0' # for matching USGS zone str e.g. '2.0' below
#     else:
#         UTM_zonestr = f'{zone[-2:]}.0' # for matching USGS zone str e.g. '22.0' below
# else:
#     UTM_zonestr = ''

# only find pairs with second image from startdatestr and later; don't process earlier missing ones
startdatestr = args.startdatestr
# only find pairs with second image from stopdatestr and earlier; don't process later missing ones
stopdatestr = args.stopdatestr

# get the Landsat C2 image list here: https://www.usgs.gov/core-science-systems/nli/landsat/bulk-metadata-service
# https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_OT_C2_L1.csv.gz
USGS_list_file = args.USGS_list_file

fname = pathlib.Path(USGS_list_file)
# assert fname.exists(), f'No such file: {fname}'  # check that the file exists

l8USGSfile_age = None
if fname.exists(): # get its age
    mtime = dt.datetime.fromtimestamp(fname.stat().st_mtime)
    l8USGSfile_age = dt.datetime.now() - mtime
if not(fname.exists()) or (l8USGSfile_age and l8USGSfile_age.days > 2):
    # get new USGS file and gunzip it
    url = 'https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_OT_C2_L1.csv.gz'
    response = requests.get(url)
    with open(url.split('/')[-1],'wb') as outgzfp:
        outgzfp.write(response.content)

    # get the path record again (?needed to reset the time?)
    fname = pathlib.Path(USGS_list_file)
    mtime = dt.datetime.fromtimestamp(fname.stat().st_mtime)
    l8USGSfile_age = dt.datetime.now() - mtime





print(f'using USGS list {USGS_list_file} that is {l8USGSfile_age.days} days old with cloudfraction_cutoff {cloudfraction_cutoff} and max_delt {max_time_separation_days}')
if args.log_numbers_file:
    logfile.write(f'using USGS list {USGS_list_file} that is {l8USGSfile_age.days} days old with cloudfraction_cutoff {cloudfraction_cutoff} and max_delt {max_time_separation_days}\n')
# first read its_live existing image pair list for this zone directory
# skipping all pairs with contributions from LE07, LT05, LT04 (just L8 right now) and the occasional '.mat' file that finds its way in to s3 directory listing
# print(f'reading list of existing .nc files from {args.zone_list_directory}/{zone}_filelist.txt')
# with open(f'{args.zone_list_directory}/{zone}_filelist.txt','r') as infile:
#     existing_pairs_files = [x.rstrip().split('/')[-1] for x in infile.readlines() if 'LT05' not in x and 'LE07' not in x and 'LT04' not in x and 'mat' not in x]

existing_pairs_files = []
for injsonfile in injsonlist:
    print(f'{injsonfile}')
    if args.read_geojsons_from_itslive_S3:
        with s3.open(injsonfile) as infile:
            imgs = json.load(infile)    
    else: # local file
        with open(injsonfile) as infile:
            imgs = json.load(infile)
    for f in imgs['features']:
        existing_pairs_files.append(f['properties']['filename'])



total_existing_its_live_pairs_all_pr_combos = len(existing_pairs_files)
existing_its_live_pairs_same_prs = 0
existing_its_live_pairs_crossed_prs = 0
existing_its_live_pairs_same_prs_P000 = 0
existing_its_live_pairs_same_prs_RT = 0
existing_its_live_pairs_crossed_prs_P000 = 0
min_delt = 200.0
max_delt = 0.0
# find all included path_rows
existing_pathrow_set = set()
for pair_name in existing_pairs_files:
    split_name = pair_name.split('_')
    if len(split_name) > 14:
        existing_pathrow_set.add(split_name[2])
        existing_pathrow_set.add(split_name[10])
        if split_name[2] == split_name[10]: # same path/row for both images
            if '_RT' in pair_name:
                existing_its_live_pairs_same_prs_RT += 1
            else:
                existing_its_live_pairs_same_prs += 1
                if 'P000' in pair_name:
                    existing_its_live_pairs_same_prs_P000 += 1
            dt_img1 = dt.datetime.strptime(split_name[3],"%Y%m%d")
            dt_img2 = dt.datetime.strptime(split_name[11],"%Y%m%d")
            # old its_live had younger image first in filename - this is now switched for newer files that are older_younger
#             delt = (dt_img1 - dt_img2).days
            delt = (dt_img2 - dt_img1).days
            if delt > max_delt:
                max_delt = delt
            if delt < min_delt:
                min_delt = delt
            if delt < 16:
                print(f'?delt {delt} for {pair_name}')
        else:
            existing_its_live_pairs_crossed_prs += 1
            if 'P000' in pair_name:
                existing_its_live_pairs_crossed_prs_P000 += 1
                
print(f'its_live has {total_existing_its_live_pairs_all_pr_combos} L8 pairs, {existing_its_live_pairs_same_prs} same path/row non_RT min_delt {min_delt} max_delt {max_delt} ({existing_its_live_pairs_same_prs_RT} _RT, {existing_its_live_pairs_same_prs_P000} P000), {existing_its_live_pairs_crossed_prs} crossed path/row ({existing_its_live_pairs_crossed_prs_P000} P000)')
if args.log_numbers_file:
    logfile.write(f'its_live has {total_existing_its_live_pairs_all_pr_combos} L8 pairs, {existing_its_live_pairs_same_prs} same path/row non_RT min_delt {min_delt} max_delt {max_delt} ({existing_its_live_pairs_same_prs_RT} _RT, {existing_its_live_pairs_same_prs_P000} P000), {existing_its_live_pairs_crossed_prs} crossed path/row ({existing_its_live_pairs_crossed_prs_P000} P000)')

same_pr_delt_first_occurence = 0
same_pr_duplicates = 0
#now build dictionary entries for each image pair file
img_dict = {x:{} for x in existing_pathrow_set}
for pair_name in existing_pairs_files:
    split_name = pair_name.split('_')
    if len(split_name) > 14 and 'RT' not in pair_name:
        dt_img1 = dt.datetime.strptime(split_name[3],"%Y%m%d")
        dt_img2 = dt.datetime.strptime(split_name[11],"%Y%m%d")
        # old its_live had younger image first in filename - this is now switched for newer files that are older_younger
#         delt = (dt_img1 - dt_img2).days
        delt = (dt_img2 - dt_img1).days
        pr_key = split_name[2] # path_row from first image
#         img1_key = '_'.join(split_name[0:4])
        img1 = '_'.join(split_name[0:7])
        img2_key = '_'.join(split_name[8:12])
        img2 = '_'.join(split_name[8:15])
        proc_info = '_'.join(split_name[15:])
        if img2_key in img_dict[pr_key].keys():
            if delt in img_dict[pr_key][img2_key].keys():
                newkey = len(img_dict[pr_key][img2_key][delt].keys()) + 1
                img_dict[pr_key][img2_key][delt][newkey] = {'img1':img1, 'img2':img2}
                same_pr_duplicates += 1
            else:
                img_dict[pr_key][img2_key][delt] = {1:{'img1':img1, 'img2':img2}}
                same_pr_delt_first_occurence += 1
        else:
            img_dict[pr_key][img2_key] = {delt:{1:{'img1':img1, 'img2':img2}}}
            same_pr_delt_first_occurence += 1
    else:
        if args.v:
            print(f'image pair list value {pair_name} not valid, skipping this pair')

    

print(f'reading USGS list of available C2 scenes from {USGS_list_file} that is {l8USGSfile_age.days} days old')

# reader = csv.DictReader(open(USGS_list_file, newline=''))

# using gzipped version of file so it doesn't have to be unzipped after download
reader = csv.DictReader(gzip.open(USGS_list_file, 'rt',newline=''))

# find all images in USGS current list that are in the present path_row set
usgs_imagelists_dict = {x:{} for x in img_dict.keys()}

counter = 0
nadir_failed_to_match_zone = 0
off_nadir = 0
rt_or_lt_count = 0
#row = next(reader)
for row in reader:
    counter += 1
    if counter % 100000 == 0:
        print(f'scanned {counter} L8 scenes so far')
    if not(row['Collection Category'] == 'RT') and not(row['Landsat Product Identifier L1'][:2] == 'LT'):  # skip near Real Time and Thermal only acquisitions
        if row['Nadir/Off Nadir'] == 'NADIR':

            fields = row['Landsat Product Identifier L1'].split('_') # 'LC08_L1TP_087231_20200604_20200604_01_T1'

            if fields[2] in img_dict.keys(): # a path/row we want 
                img_short = '_'.join(fields[:4])
                if img_short not in usgs_imagelists_dict[fields[2]]:
                    usgs_imagelists_dict[fields[2]][img_short] = row
                else:
                    print(f"DUPLICATE IMAGE in USGS list {usgs_imagelists_dict[fields[2]][img_short]['Landsat Product Identifier L1']} and {row['Landsat Product Identifier L1']}")

        else:
            off_nadir += 1
    else:
        rt_or_lt_count += 1

reader = None

print(f'scanned {counter} L8 scenes: nadir_failed_to_match_zone {nadir_failed_to_match_zone}  off_nadir {off_nadir}  rt_or_lt_count {rt_or_lt_count}')  

        
        
print('finding missing pairs') 
# now find pairs from USGS list that are missing from its_live archive (img_dict)
total_needed_pairs = 0
usgs_needed_pairs_dict = {x:{} for x in img_dict.keys()}

for pathrow in usgs_needed_pairs_dict:

    temp_usgs_lowcloud = []
    
    for image in usgs_imagelists_dict[pathrow]:
        if float(usgs_imagelists_dict[pathrow][image]['Land Cloud Cover']) <= cloudfraction_cutoff:
            temp_usgs_lowcloud.append((image.split('_')[3],image))
    
    temp_usgs_lowcloud.sort(reverse=True)
    usgs_lowcloud = [x[1] for x in temp_usgs_lowcloud]
    
    
    
    usgs_needed_pairs_dict[pathrow] = {x:{} for x in usgs_lowcloud}

    for index,scene in enumerate(usgs_lowcloud[:-1]): # skip oldest (USGS list returns in reverse time order because file has youngest at top
        dt_img2 = dt.datetime.strptime(scene.split('_')[3],"%Y%m%d")
        n=index+1
        dt_img1 = dt.datetime.strptime(usgs_lowcloud[n].split('_')[3],"%Y%m%d")
        delt = (dt_img2 - dt_img1).days
        while n < len(usgs_lowcloud) and delt <= max_time_separation_days:
# a hack to see if this is preventing matches because its_live went to older_younger order in new filenames, but code was written originally for reverse (see changes above to delt calculations)
#             temp_pair_dict = {
#                             'img1':usgs_imagelists_dict[pathrow][scene]['Landsat Product Identifier L1'], 
#                             'img2':usgs_imagelists_dict[pathrow][usgs_lowcloud[n]]['Landsat Product Identifier L1']
#                             }
            temp_pair_dict = {
                            'img1':usgs_imagelists_dict[pathrow][usgs_lowcloud[n]]['Landsat Product Identifier L1'],
                            'img2':usgs_imagelists_dict[pathrow][scene]['Landsat Product Identifier L1'] 
                            }
            found_match = False
            if not(args.ignore_existing_itslive_pairs): # skip this section if we are creating all possible combinations and ignoring what is in archive already
                if scene in img_dict[pathrow].keys() and delt in img_dict[pathrow][scene].keys():
                    posspairs = img_dict[pathrow][scene][delt]
                    for c in posspairs:  # iterate over versions for this delt pair if they are there
                        # print(posspairs[c])
                        if img_short_name(posspairs[c]['img1']) == img_short_name(temp_pair_dict['img1']) and \
                                    img_short_name(posspairs[c]['img2']) == img_short_name(temp_pair_dict['img2']):
                            found_match = True
            if not found_match: # record this pair as needing to be processed
                usgs_needed_pairs_dict[pathrow][scene][delt] = temp_pair_dict
#                 usgs_needed_pairs_dict[pathrow][scene] = {delt:temp_pair_dict}
                total_needed_pairs += 1
            n += 1
            if n<len(usgs_lowcloud):
#                 dt_img2 = dt.datetime.strptime(usgs_lowcloud[n].split('_')[3],"%Y%m%d")
#                 delt = (dt_img1 - dt_img2).days
                dt_img1 = dt.datetime.strptime(usgs_lowcloud[n].split('_')[3],"%Y%m%d")
                delt = (dt_img2 - dt_img1).days
                
# total_existing_its_live_pairs = np.sum([len(img_dict[x][y]) for x in img_dict.keys() for y in img_dict[x]])
print(f'found {total_needed_pairs} missing/new scene pairs to process; its_live has {same_pr_delt_first_occurence} or {existing_its_live_pairs_same_prs - same_pr_duplicates} (non-duplicate) ({same_pr_duplicates} same_pr_delt_duplicates) L8-only same path/row pairs already')
if args.log_numbers_file:
    logfile.write(f'found {total_needed_pairs} missing/new scene pairs to process; its_live has {same_pr_delt_first_occurence} or {existing_its_live_pairs_same_prs - same_pr_duplicates} (non-duplicate) ({same_pr_duplicates} same_pr_delt_duplicates) L8-only same path/row pairs already')

######### Collection 2 on AWS search not returning full list for a path/row query (12/11/2020) so just pull possible filenames from USGS list, assume they should be there on AWS

####################
#
# COLLECTION 2 LEVEL 1 AWS path - not used here, as we only provide image names here
#
####################
# aws s3 ls --request-payer requester s3://usgs-landsat/collection02/level-1/standard/oli-tirs/2020/009/011/LC08_L1TP_009011_20201007_20201018_02_T1/LC08_L1TP_009011_20201007_20201018_02_T1_B8.TIF
# 2020-10-18 15:05:31  330904916 LC08_L1TP_009011_20201007_20201018_02_T1_B8.TIF


image_pair_list = []
pre_cutoff_pairs = 0
for pathrow in usgs_needed_pairs_dict:
    for scene_short in usgs_needed_pairs_dict[pathrow]:
        # if more recent image in pair is at or after startdatestr date
        if scene_short.split('_')[-1] >= startdatestr and scene_short.split('_')[-1] <= stopdatestr:
            for delt in usgs_needed_pairs_dict[pathrow][scene_short]:
                img1 = usgs_needed_pairs_dict[pathrow][scene_short][delt]['img1']
                img2 = usgs_needed_pairs_dict[pathrow][scene_short][delt]['img2']
#                 image_pair_list.append((img2,img1))  # img1 is actually the older of the two images, so switch order
                image_pair_list.append((img1,img2))
        else:
            for delt in usgs_needed_pairs_dict[pathrow][scene_short]:
                pre_cutoff_pairs += 1
                img1 = usgs_needed_pairs_dict[pathrow][scene_short][delt]['img1']
                img2 = usgs_needed_pairs_dict[pathrow][scene_short][delt]['img2']
                if args.v:
                    if scene_short.split('_')[-1] <= startdatestr:
                        print(f'               -> second image too old {img1} {img2}') 
                    else:
                        print(f'               -> second image too young {img1} {img2}')
            

print(f'found {len(image_pair_list)} pairs of images to process; for {pre_cutoff_pairs} pairs both images were earlier than {startdatestr} or second image was later than {stopdatestr}')
if args.log_numbers_file:
    logfile.write(f'found {len(image_pair_list)} pairs of images to process; for {pre_cutoff_pairs} pairs both images were earlier than {startdatestr} or second image was later than {stopdatestr}\n\n')

with open(f'L8_pairs_to_process_{dt.datetime.now().strftime("%Y_%m_%d")}.txt','w') as outfile:
    for (img1,img2) in image_pair_list:
        outfile.write(f'{img1} {img2}\n')

if args.log_numbers_file:
    logfile.close()

