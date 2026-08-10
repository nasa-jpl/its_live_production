"""Binary data type to support ITS_LIVE data products.

Note: This module uses lazy imports to avoid forcing heavy dependencies
(shapefile, itscube_types, itslive_mosaics_types) on tools that only need
the BinaryFlag constant definitions.
"""
from dataclasses import dataclass
import numpy as np


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

   def _get_meanings(self):
      """Lazy-load meanings to avoid importing heavy dependencies at module load."""
      # Import only when meanings are accessed
      import shapefile
      from itscube_types import Vars
      from itslive_mosaics_types import CompositeVars

      return {
         Vars.interp_mask: 'measured interpolated',
         shapefile.LANDICE: 'non-ice ice',
         shapefile.FLOATINGICE: 'non-ice ice',
         CompositeVars.sensor_include: 'filter_not_applied filter_applied',
         Vars.ascending_img1: 'descending ascending',
         Vars.ascending_img2: 'descending ascending'
      }

   @property
   def meanings(self):
      """Binary mask meanings with lazy import."""
      if not hasattr(self, '_meanings_cache'):
         object.__setattr__(self, '_meanings_cache', self._get_meanings())
      return self._meanings_cache

BinaryFlag = BinaryFlagInfo()
