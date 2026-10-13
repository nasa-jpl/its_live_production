"""
Unit tests for utils/compare_zarr_cubes.py using pytest.

All tests exercise the pure comparison logic (_compare_datasets() and its
helpers) directly against small, in-memory synthetic xr.Datasets -- no S3,
no real zarr stores, matching this repo's convention of not testing the
I/O code path itself (see test_deep_copy_cube.py's synthetic-fixture
tests).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# Add src/ (for utils.py/itscube_types.py) and src/utils/ (for
# compare_zarr_cubes.py itself) to the import path. src/utils.py (a
# module) and src/utils/ (a plain directory of standalone scripts, no
# __init__.py) coexist -- 'import utils' always resolves to the former
# regardless of search order, since a directory with no __init__.py is
# only ever considered as a last-resort namespace package.
_SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_SRC_DIR / 'utils'))

from itscube_types import CubeFormat, Vars
import compare_zarr_cubes as czc


def _build_dataset():
   """Baseline synthetic deep-copy-cube-shaped dataset: one 3D (time,y,x)
   int16 var with a missing_value sentinel + sharding + a compressor, one
   3D (time,y,x) float32 var with a NaN fill (no sharding), and one static
   2D var -- enough to exercise every check in compare_zarr_cubes.py.
   old_cube/new_cube each call this fresh rather than sharing/copying one
   instance, so per-test mutations never leak across tests.
   """
   time = np.array(
      ['2020-01-01', '2020-01-02', '2020-01-03',
       '2020-01-04', '2020-01-05', '2020-01-06'],
      dtype='datetime64[ns]'
   )
   y = np.arange(4, dtype='float64')
   x = np.arange(5, dtype='float64')

   vx = np.zeros((6, 4, 5), dtype='int16')
   m11 = np.zeros((6, 4, 5), dtype='float32')
   m11[0, 0, 0] = np.nan

   ds = xr.Dataset(
      data_vars={
         Vars.vx: (('time', 'y', 'x'), vx),
         Vars.m11: (('time', 'y', 'x'), m11),
         'landice': (('y', 'x'), np.ones((4, 5), dtype='uint8')),
      },
      coords={'time': time, 'y': y, 'x': x},
      attrs={
         CubeFormat.date_created: '2026-01-01',
         'title': 'ITS_LIVE datacube of image pair velocities',
      }
   )

   ds[Vars.vx].attrs['missing_value'] = -32767
   ds[Vars.vx].encoding['chunks'] = (2, 4, 5)
   ds[Vars.vx].encoding['shards'] = (4, 4, 5)
   ds[Vars.vx].encoding['compressors'] = [{'cname': 'lz4', 'clevel': 5}]

   # A real numpy scalar (not a plain Python float) -- matches what
   # xarray/zarr actually hand back for a variable's own fill value when
   # read off a real store, and is what exposed the _values_equal bug
   # (isinstance(x, float) is False for np.float32 -- see TestValuesEqualHelper).
   ds[Vars.m11].attrs['_FillValue'] = np.float32(np.nan)
   ds[Vars.m11].encoding['chunks'] = (2, 4, 5)
   ds[Vars.m11].encoding['compressors'] = [{'cname': 'lz4', 'clevel': 5}]

   return ds


@pytest.fixture
def old_cube():
   return _build_dataset()


@pytest.fixture
def new_cube():
   return _build_dataset()


def _errors(findings):
   return [f for f in findings if f['level'] == 'ERROR']


def _warnings(findings):
   return [f for f in findings if f['level'] == 'WARNING']


def _for_var(findings, var_name):
   return [f for f in findings if f['variable'] == var_name]


class TestIdenticalCubes:
   def test_no_errors_or_warnings(self, old_cube, new_cube):
      findings = czc._compare_datasets(old_cube, new_cube)
      assert _errors(findings) == []
      assert _warnings(findings) == []

   def test_info_match_per_variable(self, old_cube, new_cube):
      findings = czc._compare_datasets(old_cube, new_cube)
      messages = {f['variable']: f['message'] for f in findings if f['level'] == 'INFO'}
      assert 'match exactly' in messages[Vars.vx]
      assert 'match exactly' in messages[Vars.m11]

   def test_nan_vs_nan_is_not_a_mismatch(self, old_cube, new_cube):
      # Baseline already has m11[0, 0, 0] = NaN on both sides.
      findings = czc._compare_datasets(old_cube, new_cube)
      assert _for_var(_errors(findings), Vars.m11) == []


class TestValueMismatches:
   def test_nan_vs_real_number_is_an_error(self, old_cube, new_cube):
      new_cube[Vars.m11].values[0, 0, 0] = 1.5
      findings = czc._compare_datasets(old_cube, new_cube)
      errors = _for_var(_errors(findings), Vars.m11)
      assert len(errors) == 1
      assert '1/120 elements differ' in errors[0]['message']
      assert 'index (0, 0, 0)' in errors[0]['message']

   def test_mismatch_reported_with_correct_global_index_across_batches(self, old_cube, new_cube):
      # 6 layers, batch_size=2 -> batches [0,2)/[2,4)/[4,6); place the
      # mismatch in the third batch (global layer 4) to verify the offset
      # math, not just a first-batch hit.
      new_cube[Vars.vx].values[4, 1, 2] = 5
      findings = czc._compare_datasets(old_cube, new_cube, batch_size=2)
      errors = _for_var(_errors(findings), Vars.vx)
      assert len(errors) == 1
      assert 'index (4, 1, 2)' in errors[0]['message']
      assert 'old=0' in errors[0]['message']
      assert 'new=5' in errors[0]['message']


class TestVariablePresence:
   def test_var_missing_from_new_is_an_error(self, old_cube, new_cube):
      new_cube = new_cube.drop_vars('landice')
      findings = czc._compare_datasets(old_cube, new_cube)
      errors = _for_var(_errors(findings), 'landice')
      assert len(errors) == 1
      assert 'missing from new' in errors[0]['message']

   def test_var_added_in_new_is_info_not_error(self, old_cube, new_cube):
      new_cube = new_cube.assign(floatingice=(('y', 'x'), np.zeros((4, 5), dtype='uint8')))
      findings = czc._compare_datasets(old_cube, new_cube)
      assert _for_var(_errors(findings), 'floatingice') == []
      info = _for_var([f for f in findings if f['level'] == 'INFO'], 'floatingice')
      assert len(info) == 1
      assert 'new addition' in info[0]['message']


class TestAttributes:
   def test_ignored_global_attr_difference_is_not_reported(self, old_cube, new_cube):
      new_cube.attrs[CubeFormat.date_created] = '2026-06-01'
      findings = czc._compare_datasets(old_cube, new_cube)
      assert _for_var(_errors(findings), '(global)') == []

   def test_non_ignored_global_attr_difference_is_an_error(self, old_cube, new_cube):
      new_cube.attrs['title'] = 'something else'
      findings = czc._compare_datasets(old_cube, new_cube)
      errors = _for_var(_errors(findings), '(global)')
      assert any('title' in f['message'] for f in errors)

   def test_variable_attr_difference_is_an_error(self, old_cube, new_cube):
      old_cube[Vars.vx].attrs['standard_name'] = 'land_ice_surface_velocity_x'
      findings = czc._compare_datasets(old_cube, new_cube)
      errors = _for_var(_errors(findings), Vars.vx)
      assert any('standard_name' in f['message'] for f in errors)


class TestEncoding:
   def test_differing_chunks_is_an_error(self, old_cube, new_cube):
      new_cube[Vars.vx].encoding['chunks'] = (3, 4, 5)
      findings = czc._compare_datasets(old_cube, new_cube)
      errors = _for_var(_errors(findings), Vars.vx)
      assert any('chunks differ' in f['message'] for f in errors)

   def test_differing_shards_is_an_error(self, old_cube, new_cube):
      new_cube[Vars.vx].encoding['shards'] = (2, 4, 5)
      findings = czc._compare_datasets(old_cube, new_cube)
      errors = _for_var(_errors(findings), Vars.vx)
      assert any('shards differ' in f['message'] for f in errors)

   def test_matching_none_shards_is_not_an_error(self, old_cube, new_cube):
      # m11 is unsharded on both sides (no 'shards' key set in the fixture).
      findings = czc._compare_datasets(old_cube, new_cube)
      errors = _for_var(_errors(findings), Vars.m11)
      assert not any('shards' in f['message'] for f in errors)

   def test_differing_compressor_family_is_an_error(self, old_cube, new_cube):
      new_cube[Vars.vx].encoding['compressors'] = [{'cname': 'zstd', 'clevel': 5}]
      findings = czc._compare_datasets(old_cube, new_cube)
      errors = _for_var(_errors(findings), Vars.vx)
      assert any('compressor family differs' in f['message'] for f in errors)

   def test_differing_clevel_same_cname_is_not_an_error(self, old_cube, new_cube):
      new_cube[Vars.vx].encoding['compressors'] = [{'cname': 'lz4', 'clevel': 9}]
      findings = czc._compare_datasets(old_cube, new_cube)
      errors = _for_var(_errors(findings), Vars.vx)
      assert not any('compressor' in f['message'] for f in errors)

   def test_fill_value_same_across_attrs_and_encoding_buckets_is_not_an_error(self, old_cube, new_cube):
      # old carries missing_value in .attrs; move it to .encoding on new,
      # same value -- must still be recognized as a match.
      del new_cube[Vars.vx].attrs['missing_value']
      new_cube[Vars.vx].encoding['missing_value'] = -32767
      findings = czc._compare_datasets(old_cube, new_cube)
      errors = _for_var(_errors(findings), Vars.vx)
      assert not any('missing_value' in f['message'] for f in errors)

   def test_differing_fill_value_is_an_error(self, old_cube, new_cube):
      new_cube[Vars.vx].attrs['missing_value'] = -1
      findings = czc._compare_datasets(old_cube, new_cube)
      errors = _for_var(_errors(findings), Vars.vx)
      assert any('missing_value' in f['message'] for f in errors)

   def test_nan_fill_value_on_both_sides_is_not_an_error(self, old_cube, new_cube):
      # Baseline already has _FillValue=NaN (as a real np.float32 scalar,
      # matching what a real store hands back) on both sides for m11.
      findings = czc._compare_datasets(old_cube, new_cube)
      errors = _for_var(_errors(findings), Vars.m11)
      assert not any('_FillValue' in f['message'] for f in errors)


class TestValuesEqualHelper:
   """Regression coverage for a real bug: raw zarr metadata hands back
   fill values as numpy scalars (e.g. np.float32(nan)), not plain Python
   floats -- isinstance(x, float) is False for those, so the original
   implementation fell through to `nan == nan` (always False) and
   reported identical NaN sentinels as ERRORs.
   """

   def test_python_nan_vs_python_nan(self):
      assert czc._values_equal(float('nan'), float('nan'))

   def test_numpy_float32_nan_vs_numpy_float32_nan(self):
      assert czc._values_equal(np.float32('nan'), np.float32('nan'))

   def test_numpy_float32_nan_vs_python_nan(self):
      assert czc._values_equal(np.float32('nan'), float('nan'))

   def test_real_values_still_compared_normally(self):
      assert czc._values_equal(1, 1)
      assert not czc._values_equal(1, 2)

   def test_non_numeric_values_fall_back_to_equality(self):
      assert czc._values_equal('lz4', 'lz4')
      assert not czc._values_equal('lz4', 'zstd')
      assert czc._values_equal(None, None)

   def test_array_valued_attribute_does_not_raise(self):
      # Second real bug: a multi-element array-valued attribute (e.g.
      # GeoTransform) raised "ValueError: The truth value of an array
      # with more than one element is ambiguous" under the old
      # np.isnan(x) and np.isnan(y) / bare `==` implementation.
      geo_transform = (-3345000.0, 60.0, 0.0, 345000.0, 0.0, -60.0)
      assert czc._values_equal(geo_transform, geo_transform)
      assert czc._values_equal(list(geo_transform), list(geo_transform))
      assert not czc._values_equal(geo_transform, (0.0,) + geo_transform[1:])

   def test_array_valued_attribute_with_nan_elements(self):
      assert czc._values_equal(
         np.array([1.0, np.nan, 3.0], dtype='float32'),
         np.array([1.0, np.nan, 3.0], dtype='float32'),
      )
      assert not czc._values_equal(
         np.array([1.0, np.nan, 3.0], dtype='float32'),
         np.array([1.0, np.nan, 4.0], dtype='float32'),
      )


class TestCoordinates:
   def test_mismatched_grid_skips_all_per_variable_checks(self, old_cube, new_cube):
      new_cube = new_cube.assign_coords(y=new_cube['y'].values + 100)
      findings = czc._compare_datasets(old_cube, new_cube)

      assert any(f['level'] == 'ERROR' and f['variable'] == 'y' for f in findings)
      assert any(f['variable'] == '(root)' for f in findings)
      # No per-variable checks should have run at all.
      for var_name in (Vars.vx, Vars.m11, 'landice'):
         assert _for_var(findings, var_name) == []

   def test_mismatched_time_length_warns_and_bounds_comparison_to_overlap(self, old_cube, new_cube):
      new_cube = new_cube.isel(time=slice(0, 4))
      findings = czc._compare_datasets(old_cube, new_cube)

      time_warnings = _for_var(_warnings(findings), 'time')
      assert len(time_warnings) == 1
      assert 'overlapping 4' in time_warnings[0]['message']

      # First 4 layers are identical, so no value-mismatch error despite
      # the differing layer counts.
      assert _for_var(_errors(findings), Vars.vx) == []


class TestVarNamesFilter:
   def test_restricts_to_requested_variables_and_warns_on_unknown(self, old_cube, new_cube):
      findings = czc._compare_datasets(old_cube, new_cube, var_names=['nonexistent_var', Vars.vx])

      unknown_warnings = _for_var(_warnings(findings), 'nonexistent_var')
      assert len(unknown_warnings) == 1
      assert 'not common to both cubes' in unknown_warnings[0]['message']

      # Vars.vx was requested and should have been checked...
      assert _for_var(findings, Vars.vx) != []
      # ...but m11/landice were not requested and should be entirely absent.
      assert _for_var(findings, Vars.m11) == []
      assert _for_var(findings, 'landice') == []
