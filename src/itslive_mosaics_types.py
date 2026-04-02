"""Data types to support ItS_LIVE composites and mosaics
"""
from dataclasses import dataclass

from itscube_types import Vars
from utils import Coords, Units, OutputFormat


# Former CubeJson
@dataclass(frozen=True)
class GeoJsonVarsInfo:
   """
   Variables names within GeoJson cube catalog file.
   """
   features: str = 'features'
   properties: str = 'properties'
   data_epsg: str = 'data_epsg'
   epsg: str = 'epsg'
   geometry_epsg: str = 'geometry_epsg'
   coordinates: str = 'coordinates'
   roi_percent_coverage: str = 'roi_percent_coverage'
   epsg_separator: str = ':'
   epsg_prefix: str = 'EPSG'
   url: str = 'zarr_url'
   composite_url: str = 'composite_zarr_url'
   exist_flag: str = 'datacube_exist'
   granule_count: str = 'granule_count'
   region: str = 'region_id'
   region_id: str = 'M_ID'
   rgi_code: str = 'RGI_CODE'
   directory: str = 'directory'

GeoJsonVars = GeoJsonVarsInfo()


# Former CompOutput
@dataclass(frozen=True)
class CompositeAttrs:
   """
   Class to represent attributes for the output format of the composites data.
   """
   composites_software_version = 'composites_software_version'
   datacube_autorift_parameter_file = 'datacube_autoRIFT_parameter_file'
   sensors_labels = 'sensors_labels'

   datacube_created = 'datacube_created'
   datacube_updated = 'datacube_updated'
   datacube_s3 = 'datacube_s3'
   datacube_url = 'datacube_url'

   values = {
      OutputFormat.title: 'ITS_LIVE composites of image pair velocities'
   }

   # As of zarr v2, "_FillValue" is no longer supported as encoding parameter,
   # but instead it is set as an attribute of the data variable.
   # To guarantee conversion of fill_value to NaN when reading the data back in,
   # the "_FillValue" attribute should be set to the desired fill value as
   # well.
   fill_value_attr = 'fill_value'


# Former CompDataVars
@dataclass(frozen=True)
class CompositeVarsInfo:
   """
   Data variables and their descriptions to write annual ands static
   composites to Zarr or NetCDF output store.
   """
   attrs: CompositeAttrs = CompositeAttrs()

   # Variables introducted for composites and mosaics data products
   vx_error = 'vx_error'
   vy_error = 'vy_error'
   v_error = 'v_error'
   vx_amp_error = 'vx_amp_error'
   vy_amp_error = 'vy_amp_error'
   v_amp_error = 'v_amp_error'
   vx_amp = 'vx_amp'
   vy_amp = 'vy_amp'
   v_amp = 'v_amp'
   vx_phase = 'vx_phase'
   vy_phase = 'vy_phase'
   v_phase = 'v_phase'
   count = 'count'
   max_dt = 'dt_max'
   outlier_frac = 'outlier_percent'
   sensor_include = 'sensor_filter_applied'
   vx0 = 'vx0'
   vy0 = 'vy0'
   v0 = 'v0'
   count0 = 'count0'
   vx0_error = 'vx0_error'
   vy0_error = 'vy0_error'
   v0_error = 'v0_error'
   slope_vx = 'dvx_dt'
   slope_vy = 'dvy_dt'
   slope_v = 'dv_dt'

   # Former STD_NAME
   name = {
      Vars.vx: 'land_ice_surface_x_velocity',
      Vars.vy: 'land_ice_surface_y_velocity',
      Vars.v: 'mean annual velocity',
      vx_error: 'vx error',
      vy_error: 'vy error',
      v_error: 'v error',
      vx_amp_error: 'vx_amplitude_error',
      vy_amp_error: 'vy_amplitude_error',
      v_amp_error: 'v_amp error',
      vx_amp: 'vx_amplitude',
      vy_amp: 'vy_amplitude',
      v_amp: 'climatological [%i-%i] mean seasonal amplitude',
      vx_phase: 'vx_phase',
      vy_phase: 'vy_phase',
      v_phase: 'v_phase',
      count: 'number_of_observations',
      max_dt: 'dt_maximum',
      sensor_include: 'sensor_filter_applied',
      outlier_frac: 'outlier_percent',
      vx0: 'climatological_x_velocity',
      vy0: 'climatological_y_velocity',
      v0: f'climatological [%i-%i] velocity',
      count0: 'count0',
      vx0_error: 'vx0_velocity_error',
      vy0_error: 'vy0_velocity_error',
      v0_error: 'v0_velocity_error',
      slope_vx: 'dvx_dt',
      slope_vy: 'dvy_dt',
      slope_v: 'dv_dt'
   }

   # Former DESCRIPTION
   description = {
      Vars.vx: 'mean annual velocity of sinusoidal fit to vx',
      Vars.vy: 'mean annual velocity of sinusoidal fit to vy',
      Vars.v: 'mean annual velocity determined by taking the hypotenuse of vx and vy',
      vx_error: 'error weighted error for vx',
      vy_error: 'error weighted error for vy',
      v_error: 'error weighted error for v',
      vx_amp_error: 'error for vx_amp',
      vy_amp_error: 'error for vy_amp',
      v_amp_error: 'error for v_amp',
      vx_amp: f'climatological [%i-%i] mean seasonal amplitude of sinusoidal fit to vx',
      vy_amp: f'climatological [%i-%i] mean seasonal amplitude in sinusoidal fit in vy',
      v_amp: f'climatological [%i-%i] mean seasonal amplitude in the direction of mean flow as defined by vx0 and vy0',
      vx_phase: f'climatological [%i-%i] day of seasonal maximum velocity of sinusoidal fit to vx; Values represent numerical day of the year.',
      vy_phase: f'climatological [%i-%i] day of seasonal maximum velocity of sinusoidal fit to vy; Values represent numerical day of the year.',
      v_phase: f'day of maximum climatological [%i-%i] seasonal velocity determined from sinusoidal fit to vx and vy; Values represent numerical day of the year.',
      count: 'number of image pairs used in error weighted least squares fit',
      max_dt: 'maximum allowable time separation between image pair acquisitions included in error weighted least squares fit',
      sensor_include: 'flag = 0 if sensor filter is not applied, flag = 1 if sensor (see sensor variable) filter is applied',
      outlier_frac: f'percentage of data identified as outliers and excluded from the climatological [%i-%i] error weighted least squares fit',
      vx0: f'climatological [%i-%i] vx determined by a weighted least squares line fit, described by an offset and slope, to mean annual vx values. The climatology uses a time-intercept of January 1, %i.',
      vy0: f'climatological [%i-%i] vy determined by a weighted least squares line fit, described by an offset and slope, to mean annual vy values. The climatology uses a time-intercept of January 1, %i.',
      v0: f'determined by taking the hypotenuse of vx0 and vy0. The climatology uses a time-intercept of January 1, %i.',
      count0: f'number of image pairs used for climatological [%i-%i] means',
      vx0_error: 'error for vx0',
      vy0_error: 'error for vy0',
      v0_error: 'error for v0',
      slope_vx: f'trend [%i-%i] in vx determined by a weighted least squares line fit, described by an offset and slope, to mean annual vx values',
      slope_vy: f'trend [%i-%i] in vy determined by a weighted least squares line fit, described by an offset and slope, to mean annual vy values',
      slope_v: f'trend [%i-%i] in v determined by projecting dvx_dt and dvy_dt onto the unit flow vector defined by vx0 and vy0'
   }

CompositeVars = CompositeVarsInfo()


# Define attributes for coordinates of composites and annual mosaics
# ATTN: this is done to set coordinates attributes of the xr.Dataset before
# saving it to the file - adding some data variables to the xr.Dataaset wipes
# out coordinates attributes (xarray bug?)
TIME_ATTRS = {
   Vars.attrs.std_name: Coords.STD_NAME[Coords.TIME],
   Vars.attrs.description: Coords.DESCRIPTION[Coords.TIME]
}
SENSORS_ATTRS = {
   Vars.attrs.std_name: Coords.STD_NAME[Coords.SENSORS],
   Vars.attrs.description: Coords.DESCRIPTION[Coords.SENSORS]
}
X_ATTRS = {
   Vars.attrs.std_name: Coords.STD_NAME[Coords.X],
   Vars.attrs.description: Coords.DESCRIPTION[Coords.X],
   Units.name: Units.m
}
Y_ATTRS = {
   Vars.attrs.std_name: Coords.STD_NAME[Coords.Y],
   Vars.attrs.description: Coords.DESCRIPTION[Coords.Y],
   Units.name: Units.m
}
