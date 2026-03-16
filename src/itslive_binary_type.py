"""Binary data type to support ITS_LIVE data products.
"""
from dataclasses import dataclass
import numpy as np

import shapefile
from itscube_types import Vars
from itslive_mosaics_types import CompositeVars


@dataclass(frozen=True)
class BinaryFlagAttrsInfo:
   """
   Class to represent attributes for the output format of the binary flag data.
   """
   values = 'flag_values'
   meanings = 'flag_meanings'


# Former BinaryFlag
@dataclass(frozen=True)
class BinaryFlagInfo:
   """
   Class to store output format attributes and their values for the
   binary masking.
   """
   # Standard attributes for the output format
   attrs: BinaryFlagAttrsInfo = BinaryFlagAttrsInfo()

   # Binary mask values
   values = np.array([0, 1], dtype=np.ubyte)

   # Binary mask meanings
   meanings = {
      Vars.interp_mask: 'measured interpolated',
      shapefile.LANDICE: 'non-ice ice',
      shapefile.FLOATINGICE: 'non-ice ice',
      CompositeVars.sensor_include: 'filter_not_applied filter_applied',
      Vars.ascending_img1: 'descending ascending',
      Vars.ascending_img2: 'descending ascending'
   }

BinaryFlag = BinaryFlagInfo()
