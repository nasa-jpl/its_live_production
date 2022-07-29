import json
import requests
import datetime as dt
import numpy as np
import hyp3_sdk
import argparse
# import subprocess as sp


# use the first one: 
S2_SEARCH_URL_L1C = 'https://earth-search.aws.element84.com/v0/collections/sentinel-s2-l1c/items'
S2_SEARCH_URL_L2A= 'https://earth-search.aws.element84.com/v0/collections/sentinel-s2-l2a/items'

def get_s2_scenes_by_tile(tile: str, searchURL, cloud_max=50, data_coverage_min=50, min_date=None, max_date=None) -> list:
    """Search Element 84's STAC catalog

    Args:
        tile: Sentinel-2 tile designator (13CVB)
        cloud_max: max allowable percent cloud cover
        data_coverage_min: minimum data coverage percent for tile
        min_date '2021-01-01T00:00:00Z'
        max_date '2021-01-01T00:00:00Z'
        Only one of min_date/max_date may be specified; none => all available data
    Returns:
        list of metadata dicts: dictionaries of scenes and their metadata
    """

    utmzone = int(tile[:2])
    lat_band = tile[2:3]
    grid_square = tile[3:5]
    # fallback to querying by ESA name
#     'datetime': '2021-02-02T10:02:23Z'

    payload = {
        'query': {
            'constellation': {
                    'eq': 'sentinel-2',
                    },
            'sentinel:utm_zone': {
                'eq': utmzone,
                },
            'sentinel:latitude_band': {
                'eq': lat_band,
                },
            'sentinel:grid_square': {
                'eq': grid_square,
                },
            'eo:cloud_cover': {
                'lt': cloud_max,
                },
            'sentinel:data_coverage': {
                'gt': data_coverage_min,
                },
            },
        'limit': 1000,
    }
    if min_date and not(max_date):
        payload['query']['datetime'] = {'gt': min_date}
    elif max_date and not(min_date):
        payload['query']['datetime'] = {'lt': max_date}
    elif min_date and max_date:
        print(f'Error asking for both min_date and max_date, only one allowed')
        raise ValueError(f'both min and max dates specified, only one allowed: {tile}')
    response = requests.post(searchURL, json=payload)
    response.raise_for_status()
    if not response.json().get('numberReturned'):
        raise ValueError(f'Scene could not be found: {tile}')
    return response.json()['features']

def get_scene_info_for_l1c_direct_from_esa(scene):
    """
    takes an l1c scene id (e.g. "S2B_7VEG_20210417_0_L1C") and looks in l2a STAC returns, 
    pulls ESA tileinfo.json for L1C based on L2A data to get full id
    Entirely made to work around STAC catalog that isn't being updated (L2A is, L1C is not)
    """
    l1c_tileinfoURL = l2a_scenedict[scene.replace("L1C", "L2A")]["assets"]["info"]["href"].replace("l2a", "l1c")
    r = requests.get(l1c_tileinfoURL)
    if r.status_code != 200:
# this section was a workaround for ESA paths when there were multiple granules for a tile on the same day - some ended up in different subdirectories (numbered 0,1,...) in ESA's archive than they should have - but this only effects a few tens of scenes we care about if I remember correctly (writing comment later), so I chose to ignore the problem - the patch commented out below didn't fix those few odd cases consistently anyway, but I leave it here just to remember the issue
#         if ("/1/tileInfo.json" in l1c_tileinfoURL) and (scene.split('_')[2] > '20200401'):
#             print(f'tweeking {l1c_tileinfoURL} <<<<<<<<<<<')
#             l1c_tileinfoURL = l1c_tileinfoURL.replace(
#                 "/1/tileInfo.json", "/0/tileInfo.json"
#             )
#             r = requests.get(l1c_tileinfoURL)
#         else:
#             print(
#                 f">>>>>>>>>>>>>>>>>>>>tile {tile} can't get {l1c_tileinfoURL}, quiting",
#                 flush=True
#             )
        print(
            f">>>>>>>>>>>>>>>>>>>>tile {tile} can't get {l1c_tileinfoURL}, quiting",
            flush=True
        )


    if r.status_code == 200:
        l1c_tileinfo = json.loads(r.text)
        return l1c_tileinfo
    else:
        return None
    

def make_scene_dict_entry(scene_key):
    """
    takes an l1c scene id (e.g. "S2B_7VEG_20210417_0_L1C") and looks for it in l1c STAC returns - if not there (because catalog not updated)
    looks in l2a STAC returns, pulls ESA tileinfo.json to get full id, builds necessary info for its_live processing for this scene
    Entirely made to work around STAC catalog that isn't being updated (L2A is, L1C is not)
    """
    if scene_key in l1c_scenedict: # original scene - already have STAC response for L1C
        scene_dict = {
                        'assets': {'B08':{'href': l1c_scenedict[scene_key]['assets']['B08']['href']}},
                        'bbox':  l1c_scenedict[scene_key]["bbox"],
                        'id': scene_key,
                        'properties': {
                                        'sentinel:product_id':  l1c_scenedict[scene_key]['properties']['sentinel:product_id'], 
                                        'datetime': l1c_scenedict[scene_key]["properties"]["datetime"]
                                        },
                        }
    elif scene_key.replace('L1C','L2A') in l2a_scenedict: # need to get json dict from esa to fill out some fields
        scene_tileinfo_dict = get_scene_info_for_l1c_direct_from_esa(scene_key)
        if scene_tileinfo_dict:
            scene_dict = {
                            'assets': {'B08':{'href':"s3://sentinel-s2-l1c/" + scene_tileinfo_dict["path"] + "/B08.jp2"}},
                            'bbox': l2a_scenedict[scene_key.replace('L1C','L2A')]["bbox"],
                            'id': scene_key,
                            'properties': {
                                            'sentinel:product_id': scene_tileinfo_dict["productName"], 
                                            'datetime': l2a_scenedict[scene_key.replace('L1C','L2A')]["properties"]["datetime"]
                                            },
                            }
        else:
            print(f'>>>>>>>>>>>>>>> failed to find information for {scene_key} from l2a_scenedict')
            scene_dict = None
    else:
        print(f'>>>>>>>>>>>>>>> failed to find information for {scene_key} in either l1c_scenedict or l2a_scenedict')
        scene_dict = None
    
    return scene_dict


parser = argparse.ArgumentParser(
     prog='Stac_search_S2.py',
     formatter_class=argparse.RawDescriptionHelpFormatter,
     description='''\
         find available S2 pairs for tile; option to submit job to Hyp3
         ''')
parser.add_argument('tile', help='S2 tile designator (e.g. 12CVB)')
# parser.add_argument('-run_jobs', action='store_true', help='run image pairs found [False]')
# parser.add_argument('-output_pairlist_json', action='store_true', help='write output pairs in text file with date [False]')
parser.add_argument('-min_days', type=int, help='minimum image pair separation in days [%(default)d]', default=5)
parser.add_argument('-max_days', type=int, help='minimum image pair separation in days [%(default)d]', default=90)
parser.add_argument('-min_data_coverage', type=int, help='minimum data coverage in tile (percent)[%(default)d]', default=70)
parser.add_argument('-max_cloud', type=int, help='maximum cloud cover (percent)[%(default)d]', default=70)
parser.add_argument('-start_date', help='starting date (yyyy-MM-ddT00:00:00Z) [None]', default=None)
parser.add_argument('-stop_date', help='stop date (yyyy-MM-ddT00:00:00Z) [None]', default=None)
parser.add_argument('-batch_name_start', help='start of Hyp3 batch process name (tile and days added [%(default)s])',default='S2_Job')
parser.add_argument('-decimation_factor', type=int, help='process only every nth pair [%(default)d]', default=1)
parser.add_argument('-quiet', action='store_true', help='limit output text [False]')

args = parser.parse_args()

tile = args.tile

# tile = '12CVB'

min_delt = args.min_days
max_delt = args.max_days
if args.decimation_factor==1:
    batch_name = f'{args.batch_name_start}_{tile}_{min_delt}_{max_delt}'
else:
    batch_name = f'{args.batch_name_start}_{tile}_{min_delt}_{max_delt}_{args.decimation_factor}'



a = get_s2_scenes_by_tile(  
                            tile,
                            S2_SEARCH_URL_L1C,
                            cloud_max=args.max_cloud, 
                            data_coverage_min=args.min_data_coverage,
                            min_date = args.start_date,
                            max_date = args.stop_date
                        )
print(f'{len(a)} l1c scenes found for tile {tile}')

l1c=[(x['id'].split('_')[2],dt.datetime.strptime(x['id'].split('_')[2],"%Y%m%d"),x['id'],x['properties']['eo:cloud_cover'],x['properties']['sentinel:product_id'].split('_')[4],x['properties']['sentinel:data_coverage']) for x in a]

l1c_scenedict = {x['id']:x for x in a}

relOrbs = set([x[4] for x in l1c])
scenes_by_relOrb = { x:[] for x in relOrbs }
for tup in l1c:
    scenes_by_relOrb[tup[4]].append(tup)


outlist_l1c = []
for orb in relOrbs:  
    orbstack = scenes_by_relOrb[orb]
    for i,(dstr,dtime,id,cc,ro,cov) in enumerate(orbstack[:-1]):
        for dstr1,dtime1,id1,cc1,ro1,cov1 in orbstack[i+1:]:
            deltime = dtime - dtime1
            if deltime >= dt.timedelta(days=min_delt) and deltime <= dt.timedelta(days=max_delt):
                outlist_l1c.append((id1,id))

if not(args.quiet):
    print(f'found original {len(outlist_l1c)} l1c pairs between {min_delt} and {max_delt} days for S2 tile {tile}')
    
b = get_s2_scenes_by_tile(  
                            tile,
                            S2_SEARCH_URL_L2A,
                            cloud_max=args.max_cloud, 
                            data_coverage_min=args.min_data_coverage,
                            min_date = args.start_date,
                            max_date = args.stop_date
                        )
print(f'{len(b)} l2a scenes found for tile {tile}')

l2a=[(x['id'].split('_')[2],dt.datetime.strptime(x['id'].split('_')[2],"%Y%m%d"),x['id'],x['properties']['eo:cloud_cover'],x['properties']['sentinel:product_id'].split('_')[4],x['properties']['sentinel:data_coverage']) for x in b]

l2a_scenedict = {x['id']:x for x in b}

relOrbs = set([x[4] for x in l2a])
scenes_by_relOrb = { x:[] for x in relOrbs }
for tup in l2a:
    scenes_by_relOrb[tup[4]].append(tup)


outlist_l2a = []
for orb in relOrbs:  
    orbstack = scenes_by_relOrb[orb]
    for i,(dstr,dtime,id,cc,ro,cov) in enumerate(orbstack[:-1]):
        for dstr1,dtime1,id1,cc1,ro1,cov1 in orbstack[i+1:]:
            deltime = dtime - dtime1
            if deltime >= dt.timedelta(days=min_delt) and deltime <= dt.timedelta(days=max_delt):
                outlist_l2a.append((id1,id))

if not(args.quiet):
    print(f'found full record {len(outlist_l2a)} l2a pairs between {min_delt} and {max_delt} days for S2 tile {tile}')


new_outlist = list()
needed_scene_dict = {}  # scenes  we need to get from GC storage (haven't used them yet)
existing_scene_dict = {} # scenes we should already have in us-west-2 (used them in the first full processing round, copied from eu region)
# extra_list = list()
count_in_outlist_l1c_already = 0
# in section below, nn contains L2A version names; mm contains L1C version names
for nn in outlist_l2a:
    # mm is L1C version of L2A scene name
    # if scene is in L1C list, we would already have the image copied to us-west-2 (since L1C stac catalog is static now...)
    mm = (nn[0].replace("L2A", "L1C"), nn[1].replace("L2A", "L1C")) 
    if mm not in outlist_l1c:

        got_pair_data = True
        # do we have this scene, or do we need to copy it from eu-2?
        if mm[0] in l1c_scenedict:
            if mm[0] not in existing_scene_dict:
                temp_scene_dict_entry = make_scene_dict_entry(mm[0])
                if temp_scene_dict_entry:
                    existing_scene_dict[mm[0]] = temp_scene_dict_entry
                else:
                    got_pair_data = False
        elif mm[0] not in needed_scene_dict:
            temp_scene_dict_entry = make_scene_dict_entry(nn[0])
            if temp_scene_dict_entry:
                needed_scene_dict[mm[0]] = temp_scene_dict_entry
            else:
                got_pair_data = False
                
        if mm[1] in l1c_scenedict:
            if mm[1] not in existing_scene_dict:
                temp_scene_dict_entry = make_scene_dict_entry(mm[1])
                if temp_scene_dict_entry:
                    existing_scene_dict[mm[1]] = temp_scene_dict_entry
                else:
                    got_pair_data = False        
        elif mm[1] not in needed_scene_dict:
            temp_scene_dict_entry = make_scene_dict_entry(nn[1])
            if temp_scene_dict_entry:
                needed_scene_dict[mm[1]] = temp_scene_dict_entry
            else:
                got_pair_data = False
        
        if got_pair_data:
            # this pair isn't already set to be processed by earlier search
            new_outlist.append(mm)
        else:
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>could not get needed info for pair {mm}, skipping")

    else:
        count_in_outlist_l1c_already += 1


print(f'adding {len(new_outlist)} NEW PAIRS to be processed; more than {len(outlist_l2a) - len(outlist_l1c)} because of ? cloud coverage estimate difference between l2a and l1c, so old scenes added in ?')
print(f'count_in_outlist_l1c_already = {count_in_outlist_l1c_already}')



# now make dictionary that has URLs for scenes that need to be copied, all scenes, and the pairs of these scenes that need to be processed
processList = new_outlist



all_scenes_dict = needed_scene_dict | existing_scene_dict
scenes_to_copy_dict = {x:needed_scene_dict[x]['assets']['B08']['href'] for x in needed_scene_dict}

# if args.decimation_factor > 1:
#     processList = outlist[0:len(outlist):args.decimation_factor]
#     print(f'But -decimation_factor = {args.decimation_factor} so would process only {len(processList)} pairs')
# else:
#     processList = outlist

print(f'>>>>>> make update_dict and save to json',flush=True)
update_dict = {
                'scenes_to_copy_dict':scenes_to_copy_dict,
                'all_scenes_dict':all_scenes_dict,
                'processList':processList
                }
# if args.output_pairlist_json:
outlist_filename = f"update_for_S2_pairlist_{tile}_{len(processList)}_pairs_dtmin_{min_delt}_dtmax_{max_delt}_maxcloud_{args.max_cloud}_{dt.datetime.now().strftime('%Y%m%d')}.json"
with open(outlist_filename,'w') as outjsonfile:
    json.dump(update_dict, outjsonfile)
print(f'wrote {len(processList)} pairs to {outlist_filename}',flush=True)





