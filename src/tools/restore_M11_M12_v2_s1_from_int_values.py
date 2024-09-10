#!/usr/bin/env python
"""
Restore M11 and M12 values of the Sentinel-1 V2 data that was already restored from the int type of the values.
This script handles 34 V2 Sentinel-1 granules that we were not able to restore M11/M12 based on input parameters due
to the "Error with orbit interpolation." error:

s3://its-live-data/velocity_image_pair/sentinel1/v02/N70W080/S1A_IW_SLC__1SSH_20180308T225954_20180308T230022_020930_023EA4_8B3C_X_S1A_IW_SLC__1SSH_20180320T225955_20180320T230022_021105_024431_40F4_G0120V02_P097.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/N70W080/S1B_IW_SLC__1SSH_20170130T225957_20170130T230024_004084_0070FB_BE53_X_S1A_IW_SLC__1SSH_20170205T230039_20170205T230106_015155_018CB5_862E_G0120V02_P094.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/N70W080/S1B_IW_SLC__1SSH_20170130T225957_20170130T230024_004084_0070FB_BE53_X_S1B_IW_SLC__1SSH_20170211T225957_20170211T230024_004259_007631_CB57_G0120V02_P093.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/N70W080/S1B_IW_SLC__1SSH_20190201T225946_20190201T230013_014759_01B867_88D2_X_S1B_IW_SLC__1SSH_20190213T225945_20190213T230012_014934_01BE24_21D9_G0120V02_P098.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/N70W080/S1B_IW_SLC__1SSH_20190213T225945_20190213T230012_014934_01BE24_21D9_X_S1B_IW_SLC__1SSH_20190225T225945_20190225T230012_015109_01C3E4_0D83_G0120V02_P098.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20200920T225942_20200920T230009_034448_0401FB_F58C_X_S1A_IW_SLC__1SDV_20201002T225942_20201002T230009_034623_040820_EBDA_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20200920T225942_20200920T230009_034448_0401FB_F58C_X_S1B_IW_SLC__1SDV_20200926T225901_20200926T225928_023552_02CBFC_ABD4_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20201002T225942_20201002T230009_034623_040820_EBDA_X_S1A_IW_SLC__1SDV_20201014T225943_20201014T230010_034798_040E37_6B24_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20201014T225943_20201014T230010_034798_040E37_6B24_X_S1A_IW_SLC__1SDV_20201026T225943_20201026T230010_034973_04143B_EFDB_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20201014T225943_20201014T230010_034798_040E37_6B24_X_S1B_IW_SLC__1SDV_20201020T225901_20201020T225928_023902_02D6E1_5BFF_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20201107T225942_20201107T230009_035148_041A45_027E_X_S1A_IW_SLC__1SDV_20201119T225942_20201119T230009_035323_04205E_22DE_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20201107T225942_20201107T230009_035148_041A45_027E_X_S1B_IW_SLC__1SDV_20201113T225901_20201113T225928_024252_02E1D3_86F3_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20210611T225943_20210611T230010_038298_048507_8619_X_S1B_IW_SLC__1SDV_20210617T225902_20210617T225929_027402_0345DF_3EDD_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20210623T225943_20210623T230010_038473_048A3C_F430_X_S1B_IW_SLC__1SDV_20210629T225903_20210629T225930_027577_034AC9_1793_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20211208T225947_20211208T230014_040923_04DC1F_7301_X_S1B_IW_SLC__1SDV_20211214T225906_20211214T225933_030027_0395D9_1A9D_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20220125T225945_20220125T230012_041623_04F394_9EA6_X_S1A_IW_SLC__1SDV_20220206T225944_20220206T230011_041798_04F997_B745_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20190827T225943_20190827T230010_017777_02174C_60E2_X_S1B_IW_SLC__1SDV_20190908T225944_20190908T230011_017952_021CC0_6EE4_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20190920T225945_20190920T230011_018127_02222D_A779_X_S1B_IW_SLC__1SDV_20191002T225945_20191002T230012_018302_02279F_AB5A_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20191002T225945_20191002T230012_018302_02279F_AB5A_X_S1B_IW_SLC__1SDV_20191014T225945_20191014T230012_018477_022D0E_4525_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20191026T225945_20191026T230012_018652_02326B_816C_X_S1B_IW_SLC__1SDV_20191107T225945_20191107T230012_018827_02380D_6652_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20191201T225944_20191201T230011_019177_024345_6E33_X_S1B_IW_SLC__1SDV_20191213T225944_20191213T230011_019352_0248D8_F809_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20191213T225944_20191213T230011_019352_0248D8_F809_X_S1B_IW_SLC__1SDV_20191225T225943_20191225T230010_019527_024E68_A604_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20200118T225942_20200118T230009_019877_02598C_8B1C_X_S1B_IW_SLC__1SDV_20200130T225942_20200130T230009_020052_025F29_5AC0_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20200130T225942_20200130T230009_020052_025F29_5AC0_X_S1B_IW_SLC__1SDV_20200211T225942_20200211T230009_020227_0264D9_A3F6_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20200423T225943_20200423T230010_021277_02862D_99EA_X_S1B_IW_SLC__1SDV_20200505T225943_20200505T230010_021452_028BBA_6831_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20200505T225943_20200505T230010_021452_028BBA_6831_X_S1B_IW_SLC__1SDV_20200517T225944_20200517T230011_021627_0290F4_6F1C_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20200610T225946_20200610T230012_021977_029B65_CA44_X_S1B_IW_SLC__1SDV_20200622T225946_20200622T230013_022152_02A0BB_C015_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20200622T225946_20200622T230013_022152_02A0BB_C015_X_S1B_IW_SLC__1SDV_20200704T225947_20200704T230014_022327_02A611_6414_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20200704T225947_20200704T230014_022327_02A611_6414_X_S1B_IW_SLC__1SDV_20200716T225947_20200716T230014_022502_02AB5E_DEC4_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20200716T225947_20200716T230014_022502_02AB5E_DEC4_X_S1B_IW_SLC__1SDV_20200728T225948_20200728T230015_022677_02B0B6_1EB4_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20200728T225948_20200728T230015_022677_02B0B6_1EB4_X_S1B_IW_SLC__1SDV_20200809T225949_20200809T230016_022852_02B616_8F3B_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20201219T225950_20201219T230017_024777_02F299_DEB2_X_S1B_IW_SLC__1SDV_20201231T225949_20201231T230016_024952_02F83B_73FD_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20210217T225947_20210217T230014_025652_030ECE_6F0B_X_S1B_IW_SLC__1SDV_20210301T225947_20210301T230014_025827_031484_13EA_G0120V02_P099.nc
s3://its-live-data/velocity_image_pair/sentinel1/v02/S60E060/S1A_IW_SLC__1SSH_20160722T225952_20160722T230020_012267_013117_AF1B_X_S1A_IW_SLC__1SSH_20160803T225953_20160803T230021_012442_0136F0_D794_G0120V02_P043.nc

We will use original M11/M12 values that were stored as compressed "int" dtype as "float32" dtype for these granules.

ATTN: This script should run from AWS EC2 instance to have fast access to the S3
bucket. It takes 2 seconds to upload the file to the S3 bucket from EC2 instance
vs. 1.5 minutes to upload the file from laptop to the S3 bucket.

Authors: Masha Liukis, Alex Gardner, Joe Kennedy
"""
import argparse
import boto3
from datetime import datetime
from botocore.exceptions import ClientError
import dask
from dask.diagnostics import ProgressBar
import json
import logging
import numpy as np
import os
import pandas as pd
import s3fs
import xarray as xr

from itscube_types import DataVars, Coords, Output
from mission_info import Encoding

granules_to_correct = """
its-live-data/velocity_image_pair/sentinel1/v02/N70W080/S1A_IW_SLC__1SSH_20180308T225954_20180308T230022_020930_023EA4_8B3C_X_S1A_IW_SLC__1SSH_20180320T225955_20180320T230022_021105_024431_40F4_G0120V02_P097.nc
its-live-data/velocity_image_pair/sentinel1/v02/N70W080/S1B_IW_SLC__1SSH_20170130T225957_20170130T230024_004084_0070FB_BE53_X_S1A_IW_SLC__1SSH_20170205T230039_20170205T230106_015155_018CB5_862E_G0120V02_P094.nc
its-live-data/velocity_image_pair/sentinel1/v02/N70W080/S1B_IW_SLC__1SSH_20170130T225957_20170130T230024_004084_0070FB_BE53_X_S1B_IW_SLC__1SSH_20170211T225957_20170211T230024_004259_007631_CB57_G0120V02_P093.nc
its-live-data/velocity_image_pair/sentinel1/v02/N70W080/S1B_IW_SLC__1SSH_20190201T225946_20190201T230013_014759_01B867_88D2_X_S1B_IW_SLC__1SSH_20190213T225945_20190213T230012_014934_01BE24_21D9_G0120V02_P098.nc
its-live-data/velocity_image_pair/sentinel1/v02/N70W080/S1B_IW_SLC__1SSH_20190213T225945_20190213T230012_014934_01BE24_21D9_X_S1B_IW_SLC__1SSH_20190225T225945_20190225T230012_015109_01C3E4_0D83_G0120V02_P098.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20200920T225942_20200920T230009_034448_0401FB_F58C_X_S1A_IW_SLC__1SDV_20201002T225942_20201002T230009_034623_040820_EBDA_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20200920T225942_20200920T230009_034448_0401FB_F58C_X_S1B_IW_SLC__1SDV_20200926T225901_20200926T225928_023552_02CBFC_ABD4_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20201002T225942_20201002T230009_034623_040820_EBDA_X_S1A_IW_SLC__1SDV_20201014T225943_20201014T230010_034798_040E37_6B24_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20201014T225943_20201014T230010_034798_040E37_6B24_X_S1A_IW_SLC__1SDV_20201026T225943_20201026T230010_034973_04143B_EFDB_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20201014T225943_20201014T230010_034798_040E37_6B24_X_S1B_IW_SLC__1SDV_20201020T225901_20201020T225928_023902_02D6E1_5BFF_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20201107T225942_20201107T230009_035148_041A45_027E_X_S1A_IW_SLC__1SDV_20201119T225942_20201119T230009_035323_04205E_22DE_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20201107T225942_20201107T230009_035148_041A45_027E_X_S1B_IW_SLC__1SDV_20201113T225901_20201113T225928_024252_02E1D3_86F3_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20210611T225943_20210611T230010_038298_048507_8619_X_S1B_IW_SLC__1SDV_20210617T225902_20210617T225929_027402_0345DF_3EDD_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20210623T225943_20210623T230010_038473_048A3C_F430_X_S1B_IW_SLC__1SDV_20210629T225903_20210629T225930_027577_034AC9_1793_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20211208T225947_20211208T230014_040923_04DC1F_7301_X_S1B_IW_SLC__1SDV_20211214T225906_20211214T225933_030027_0395D9_1A9D_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1A_IW_SLC__1SDV_20220125T225945_20220125T230012_041623_04F394_9EA6_X_S1A_IW_SLC__1SDV_20220206T225944_20220206T230011_041798_04F997_B745_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20190827T225943_20190827T230010_017777_02174C_60E2_X_S1B_IW_SLC__1SDV_20190908T225944_20190908T230011_017952_021CC0_6EE4_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20190920T225945_20190920T230011_018127_02222D_A779_X_S1B_IW_SLC__1SDV_20191002T225945_20191002T230012_018302_02279F_AB5A_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20191002T225945_20191002T230012_018302_02279F_AB5A_X_S1B_IW_SLC__1SDV_20191014T225945_20191014T230012_018477_022D0E_4525_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20191026T225945_20191026T230012_018652_02326B_816C_X_S1B_IW_SLC__1SDV_20191107T225945_20191107T230012_018827_02380D_6652_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20191201T225944_20191201T230011_019177_024345_6E33_X_S1B_IW_SLC__1SDV_20191213T225944_20191213T230011_019352_0248D8_F809_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20191213T225944_20191213T230011_019352_0248D8_F809_X_S1B_IW_SLC__1SDV_20191225T225943_20191225T230010_019527_024E68_A604_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20200118T225942_20200118T230009_019877_02598C_8B1C_X_S1B_IW_SLC__1SDV_20200130T225942_20200130T230009_020052_025F29_5AC0_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20200130T225942_20200130T230009_020052_025F29_5AC0_X_S1B_IW_SLC__1SDV_20200211T225942_20200211T230009_020227_0264D9_A3F6_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20200423T225943_20200423T230010_021277_02862D_99EA_X_S1B_IW_SLC__1SDV_20200505T225943_20200505T230010_021452_028BBA_6831_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20200505T225943_20200505T230010_021452_028BBA_6831_X_S1B_IW_SLC__1SDV_20200517T225944_20200517T230011_021627_0290F4_6F1C_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20200610T225946_20200610T230012_021977_029B65_CA44_X_S1B_IW_SLC__1SDV_20200622T225946_20200622T230013_022152_02A0BB_C015_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20200622T225946_20200622T230013_022152_02A0BB_C015_X_S1B_IW_SLC__1SDV_20200704T225947_20200704T230014_022327_02A611_6414_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20200704T225947_20200704T230014_022327_02A611_6414_X_S1B_IW_SLC__1SDV_20200716T225947_20200716T230014_022502_02AB5E_DEC4_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20200716T225947_20200716T230014_022502_02AB5E_DEC4_X_S1B_IW_SLC__1SDV_20200728T225948_20200728T230015_022677_02B0B6_1EB4_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20200728T225948_20200728T230015_022677_02B0B6_1EB4_X_S1B_IW_SLC__1SDV_20200809T225949_20200809T230016_022852_02B616_8F3B_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20201219T225950_20201219T230017_024777_02F299_DEB2_X_S1B_IW_SLC__1SDV_20201231T225949_20201231T230016_024952_02F83B_73FD_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S10W060/S1B_IW_SLC__1SDV_20210217T225947_20210217T230014_025652_030ECE_6F0B_X_S1B_IW_SLC__1SDV_20210301T225947_20210301T230014_025827_031484_13EA_G0120V02_P099.nc
its-live-data/velocity_image_pair/sentinel1/v02/S60E060/S1A_IW_SLC__1SSH_20160722T225952_20160722T230020_012267_013117_AF1B_X_S1A_IW_SLC__1SSH_20160803T225953_20160803T230021_012442_0136F0_D794_G0120V02_P043.nc
"""


class RestoreM11M12Values:
    """
    Restore original M11 and M12 values for existing V2 Sentinel1 granules as "float32" dtype,
    save it to local file with M11 and M12 values being of "float32" dtype,
    copy corrected granule and corresponding PNG files to the target location in AWS S3 bucket.
    """
    NC_ENGINE = 'h5netcdf'

    # S3 bucket with granules
    BUCKET = 'its-live-data'

    # Source S3 bucket directory
    SOURCE_DIR = 'velocity_image_pair/sentinel1/v02'

    # Target S3 bucket directory
    TARGET_DIR = None

    # Local directory to store corrected granules before copying them to the S3 bucket
    LOCAL_DIR = 'sandbox-correct-S1'

    # Number of granules to process in parallel
    CHUNK_SIZE = 100

    # Number of Dask workers for parallel processing
    DASK_WORKERS = 8

    DRYRUN = False

    def __init__(self):
        """
        Initialize object.

        Inputs:
        =======
        granule_table: File that stores information for granule correction.
        """
        self.s3 = s3fs.S3FileSystem()

        self.all_original_granules = granules_to_correct.split()
        logging.info(f'{len(self.all_original_granules)} granules to restore')

    def __call__(self, start_index: int = 0, stop_index: int = 0):
        """
        Restore M11 and M12 from original granules stored as compressed "int" dtype to "float32" dtype in the target S3 destination directory.
        """
        num_to_fix = len(self.all_original_granules)

        if stop_index > 0:
            num_to_fix = stop_index

        num_to_fix -= start_index

        # For debugging only
        # num_to_fix = 3

        start = start_index
        logging.info(f"{num_to_fix} granules to restore...")

        if num_to_fix <= 0:
            logging.info("Nothing to restore, exiting.")
            return

        if not os.path.exists(RestoreM11M12Values.LOCAL_DIR):
            os.mkdir(RestoreM11M12Values.LOCAL_DIR)

        while num_to_fix > 0:
            num_tasks = RestoreM11M12Values.CHUNK_SIZE if num_to_fix > RestoreM11M12Values.CHUNK_SIZE else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")

            tasks = [
                dask.delayed(RestoreM11M12Values.correct)(
                    each,     # granule to restore N11/M12 for
                    self.s3
                ) for each in self.all_original_granules[start:start+num_tasks]
            ]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(
                    tasks,
                    scheduler="processes",
                    num_workers=RestoreM11M12Values.DASK_WORKERS
                )

            for each_result in results[0]:
                logging.info("-->".join(each_result))

            num_to_fix -= num_tasks
            start += num_tasks

    @staticmethod
    def correct(
        granule_url: str,
        s3: s3fs.S3FileSystem,
    ):
        """
        Correct S1 data for the granule residing in S3 bucket. Copy corrected granule along with corresponding PNG files
        to the new S3 location.

        Inputs:
        =======
        granule_url: Granule path within S3 bucket.
        s3: s3fs.S3FileSystem object to access original ITS_LIVE granules for correction.
        """
        msgs = [f'Processing {granule_url}']

        granule_basename = os.path.basename(granule_url)

        # Read granule to restore the values from
        found_granule_url = granule_url.replace(
            'its-live-data/velocity_image_pair/sentinel1/v02',
            'its-live-data/velocity_image_pair/sentinel1-restoredM/v02'
        )

        ds = None

        # Local copy of fixed granule
        fixed_file = os.path.join(RestoreM11M12Values.LOCAL_DIR, granule_basename)

        # Read the granule for correction in
        with s3.open(found_granule_url, mode='rb') as fhandle:
            with xr.open_dataset(fhandle, engine=RestoreM11M12Values.NC_ENGINE) as ds:
                ds = ds.load()

                # Save the granule with M11/M12 as "float32" dtype

                for each_var in [DataVars.M11, DataVars.M12]:
                    if Output.SCALE_FACTOR in ds[each_var].encoding:
                        # Remove both compression encoding attributes if present as they are not relevant anymore
                        del ds[each_var].encoding[Output.SCALE_FACTOR]
                        del ds[each_var].encoding[Output.ADD_OFFSET]

                # Add date when granule was updated
                ds.attrs['date_updated'] = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

            # Save updated granule to the local file

            # Set chunking for 2D data variables
            dims = ds.dims
            num_x = dims[Coords.X]
            num_y = dims[Coords.Y]

            # Compute chunking like AutoRIFT does:
            # https://github.com/ASFHyP3/hyp3-autorift/blob/develop/hyp3_autorift/vend/netcdf_output.py#L410-L411
            chunk_lines = np.min([np.ceil(8192/num_y)*128, num_y])
            two_dim_chunks_settings = (chunk_lines, num_x)

            granule_encoding = Encoding.SENTINEL1.copy()

            for each_var, each_var_settings in granule_encoding.items():
                if each_var_settings[Output.FILL_VALUE_ATTR] is not None:
                    each_var_settings[Output.CHUNKSIZES_ATTR] = two_dim_chunks_settings

            # for each_var in [DataVars.M11, DataVars.M12]:
            #     granule_encoding[each_var][Output.FILL_VALUE_ATTR] = DataVars.MISSING_VALUE

            # Save to local file
            ds.to_netcdf(fixed_file, engine='h5netcdf', encoding=granule_encoding)

        # Upload corrected granule to the bucket - format sub-directory based on new cropped values
        if not RestoreM11M12Values.DRYRUN:
            s3_client = boto3.client('s3')
            try:
                # Upload granule to the target directory in the bucket:
                # remove bucket name from the target path, replace with target directory for corrected granule
                target = granule_url.replace(
                    RestoreM11M12Values.BUCKET + '/' + RestoreM11M12Values.SOURCE_DIR,
                    RestoreM11M12Values.TARGET_DIR
                )

                # msgs.append(f"Uploading to {target}")
                msgs.append(f"Uploading {fixed_file} to {RestoreM11M12Values.BUCKET}: {target}")
                s3_client.upload_file(fixed_file, RestoreM11M12Values.BUCKET, target)

                # msgs.append(f"Removing local {fixed_file}")
                os.unlink(fixed_file)

                # There are corresponding browse and thumbprint images to transfer
                bucket = boto3.resource('s3').Bucket(RestoreM11M12Values.BUCKET)

                for target_ext in ['.png', '_thumb.png']:
                    target_key = target.replace('.nc', target_ext)

                    # Path to original PNG file in the S3 bucket - just copy to new location in s3
                    source_key = target_key.replace(RestoreM11M12Values.TARGET_DIR, RestoreM11M12Values.SOURCE_DIR).replace('.nc', target_ext)

                    # msgs.append(f"Uploading {source_key} to {RestoreM11M12Values.BUCKET}: {target_key}")
                    source_dict = {
                        'Bucket': RestoreM11M12Values.BUCKET,
                        'Key': source_key
                    }

                    bucket.copy(source_dict, target_key)
                    msgs.append(f'Copying {target_ext} to s3')

            except ClientError as exc:
                msgs.append(f"ERROR: {exc}")

        return msgs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-c', '--chunk_size',
        type=int,
        default=100,
        help='Number of granules to process in parallel [%(default)d]'
    )
    parser.add_argument(
        '-t', '--target_bucket_dir',
        type=str,
        default='velocity_image_pair/sentinel1-restoredM-float-dtype/v02',
        help='AWS S3 bucket and directory to store corrected granules'
    )
    parser.add_argument(
        '-l', '--local_dir',
        type=str,
        default='sandbox-sentinel1',
        help='Directory to store fixed granules before uploading them to the S3 bucket'
    )
    parser.add_argument(
        '-w', '--dask_workers',
        type=int,
        default=8,
        help='Number of Dask parallel workers [%(default)d]'
    )
    parser.add_argument(
        '-s', '--start_granule',
        type=int,
        default=0,
        help='Index for the start granule to process (if previous processing terminated) [%(default)d]'
    )
    parser.add_argument(
        '-e', '--stop_granule',
        type=int,
        default=0,
        help='Index for the last granule to process (if splitting processing across multiple EC2s) [%(default)d]'
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='Dry run, do not actually copy any data to AWS S3 bucket'
    )

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")

    RestoreM11M12Values.CHUNK_SIZE = args.chunk_size
    RestoreM11M12Values.DASK_WORKERS = args.dask_workers
    RestoreM11M12Values.TARGET_DIR = args.target_bucket_dir
    RestoreM11M12Values.LOCAL_DIR = args.local_dir
    RestoreM11M12Values.DRYRUN = args.dryrun

    process_granules = RestoreM11M12Values()
    process_granules(args.start_granule, args.stop_granule)


if __name__ == '__main__':
    main()

    logging.info("Done.")
