"""
Resumability support for deep_copy_cube_per_var_chunk.py: S3 marker objects
record which (variable, chunk) work a prior attempt already finished, so a
retry (e.g. after a spot termination) can skip it instead of starting over.

Split out of deep_copy_cube_per_var_chunk.py so the marker/config-guard
mechanics can be read and tested independently of the copy logic. See
_Progress's docstring for the marker layout.
"""
import json
import logging
import os
import tempfile

import s3fs

import itslive_utils
import utils


# Markers are zero-byte S3 objects; only presence matters, never content.
SUCCESS_MARKER = '_SUCCESS'
RUN_CONFIG_NAME = 'run_config.json'

# Every write needs this ACL (see ~20 other scripts in this repo) since the
# job often runs under a non-bucket-owner account. Threaded through
# s3_additional_kwargs because markers go via s3fs's put_object, not the AWS
# CLI (verified: 'ACL' survives s3fs's _filter_kwargs whitelist).
BUCKET_OWNER_ACL = 'bucket-owner-full-control'

# Keys that define what a chunk marker MEANS -- see validate_config().
_CONFIG_SHAPE_KEYS = (
   'output_store',
   'time_chunk',
   'xy_chunk',
   'time_chunk_1d',
   'xy_shard_multiplier',
   'num_layers',
)


@itslive_utils.retry_decorator(max_retries=5)
def _marker_exists(s3, marker_path):
   """Check whether a marker exists.

   Args:
      s3 (s3fs.S3FileSystem): filesystem to check.
      marker_path (str): path to check.

   Returns:
      bool: True if the marker exists.
   """
   return s3.exists(marker_path)


@itslive_utils.retry_decorator(max_retries=5)
def _write_marker(s3, marker_path):
   """Touch (create empty) a marker.

   Args:
      s3 (s3fs.S3FileSystem): filesystem to write to.
      marker_path (str): path to touch.
   """
   s3.touch(marker_path)


@itslive_utils.retry_decorator(max_retries=5)
def _remove_prefix(s3, prefix):
   """Recursively delete a prefix, tolerating its absence.

   Args:
      s3 (s3fs.S3FileSystem): filesystem to delete from.
      prefix (str): prefix to delete.
   """
   if s3.exists(prefix):
      s3.rm(prefix, recursive=True)


@itslive_utils.retry_decorator(max_retries=5)
def _read_json(s3, path):
   """Read and parse a JSON object.

   Args:
      s3 (s3fs.S3FileSystem): filesystem to read from.
      path (str): path of the JSON object.

   Returns:
      dict or None: the parsed JSON, or None if `path` doesn't exist.
   """
   if not s3.exists(path):
      return None

   with s3.open(path, 'r') as fhandle:
      return json.load(fhandle)


def _s3_copy(local_path, s3_path, recursive=True):
   """Upload to S3 via the AWS CLI -- same retried mechanism as
   deep_copy_cube.upload_local_staging_dir(), but scoped to one path so it
   can run per chunk without waiting for the whole run to finish.

   Args:
      local_path (str): local file or directory to upload.
      s3_path (str): destination s3:// path.
      recursive (bool): True for a directory copy, False for a single file.
   """
   command_line = ["aws", "s3", "cp"]
   if recursive:
      command_line.append("--recursive")
   command_line += [local_path, s3_path, "--acl", "bucket-owner-full-control"]
   itslive_utils.s3_copy_using_subprocess(command_line, os.environ.copy())


class _Progress:
   """Tracks which (variable, chunk) work is already done, keyed under
   --progress-dir (deliberately outside the cube's own zarr store -- a
   published data product shouldn't carry pipeline bookkeeping). `base` is
   namespaced by the output store's name so one --progress-dir can serve a
   whole batch; validate_config() cross-checks output_store too, so a
   collision fails loudly instead of silently skipping work.

   Three marker levels, checked cheapest-first: {base}/_SUCCESS (whole run
   done), {base}/{var}/_SUCCESS (one variable done -- skips its per-chunk
   checks), {base}/{var}/{chunk}.done (one chunk done, including a
   legitimate radar-skip). Plus {base}/run_config.json (see
   validate_config()).

   Completion is always an explicit marker, never inferred from which chunk
   objects exist on S3: a killed sharded-chunk upload can leave some shards
   written and others missing, and zarr fills a missing shard with
   fill_value rather than erroring -- inferring "done" from partial state
   would ship corrupt data. A missing marker only costs a safe (idempotent)
   redo; a false "done" would be dangerous.

   No locking: concurrent attempts against the same --progress-dir are
   benign (last writer wins) but must not share --local-staging-dir.
   """

   def __init__(self, s3, base):
      """Args:
         s3 (s3fs.S3FileSystem): filesystem handle to check/write markers on.
         base (str): this instance's root path (see create()).
      """
      self.s3 = s3
      self.base = base

   @classmethod
   def create(cls, progress_dir, output_store):
      """Build a _Progress using an S3FileSystem that applies
      BUCKET_OWNER_ACL to every marker written.

      Args:
         progress_dir (str): s3:// directory root for progress markers.
         output_store (str): the cube store this progress tracks; its
            basename namespaces the root.

      Returns:
         _Progress: rooted at `progress_dir`/{basename of `output_store`}.
      """
      s3 = s3fs.S3FileSystem(s3_additional_kwargs={'ACL': BUCKET_OWNER_ACL})
      store_name = os.path.basename(output_store.rstrip('/'))
      return cls(s3, f'{progress_dir.rstrip("/")}/{store_name}')

   def _success_path(self):
      return f'{self.base}/{SUCCESS_MARKER}'

   def _var_success_path(self, var_name):
      """Args:
         var_name (str): variable name.

      Returns:
         str: path of `var_name`'s own done marker.
      """
      return f'{self.base}/{var_name}/{SUCCESS_MARKER}'

   def _chunk_path(self, var_name, chunk_index):
      """Args:
         var_name (str): variable name.
         chunk_index (int): index of the chunk along 'time'.

      Returns:
         str: path of `var_name`'s `chunk_index` marker.
      """
      return f'{self.base}/{var_name}/{chunk_index}.done'

   def _config_path(self):
      return f'{self.base}/{RUN_CONFIG_NAME}'

   def is_complete(self):
      return _marker_exists(self.s3, self._success_path())

   def mark_complete(self):
      _write_marker(self.s3, self._success_path())

   def var_is_done(self, var_name):
      """Args:
         var_name (str): variable name.

      Returns:
         bool: True if every chunk of `var_name` is already marked done.
      """
      return _marker_exists(self.s3, self._var_success_path(var_name))

   def mark_var_done(self, var_name):
      """Marks a variable as fully done.

      Args:
         var_name (str): variable name.
      """
      _write_marker(self.s3, self._var_success_path(var_name))

   def chunk_is_done(self, var_name, chunk_index):
      """Args:
         var_name (str): variable name.
         chunk_index (int): index of the chunk along 'time'.

      Returns:
         bool: True if this (var_name, chunk_index) pair is already marked
            done.
      """
      return _marker_exists(self.s3, self._chunk_path(var_name, chunk_index))

   def mark_chunk_done(self, var_name, chunk_index):
      """Marks one chunk as done.

      Args:
         var_name (str): variable name.
         chunk_index (int): index of the chunk along 'time'.
      """
      _write_marker(self.s3, self._chunk_path(var_name, chunk_index))

   def read_config(self):
      return _read_json(self.s3, self._config_path())

   def write_config(self, config):
      """Persists a run config via _s3_copy (AWS CLI), like every other real
      payload object this pipeline writes -- not through s3fs.

      Args:
         config (dict): run configuration, see build_config().
      """
      with tempfile.TemporaryDirectory() as tmp_dir:
         local_path = os.path.join(tmp_dir, RUN_CONFIG_NAME)
         with open(local_path, 'w') as fhandle:
            json.dump(config, fhandle, indent=3, sort_keys=True)

         _s3_copy(local_path, self._config_path(), recursive=False)

      logging.info(f'Recorded run configuration at {self._config_path()}')

   def validate_config(self, recorded, current, cube_total_layers, time_values):
      """Reconcile this attempt's config against the recorded one.

      Chunk markers are keyed only on (var_name, chunk_index), meaningless
      without the parameters that define what a chunk_index covers -- so
      any _CONFIG_SHAPE_KEYS mismatch is a hard failure (otherwise the
      skeleton re-upload would silently pair old chunks with a new grid).

      A grown input cube (more granules ingested since the first attempt)
      isn't an operator mistake, so it's clamped to the recorded
      total_layers (logged) instead of failed; a shrunk cube hard-fails
      (the declared shape could never be filled). Recorded first/last
      'time' are also re-checked: granules are ordered by mid_date, so an
      *inserted* (not appended) granule would shift chunks without
      changing the count, which clamping alone wouldn't catch.

      Args:
         recorded (dict): the first attempt's config, from read_config().
         current (dict): this attempt's config, from build_config().
         cube_total_layers (int): input cube's resolved layer count.
         time_values (np.ndarray): input cube's 'time' coordinate, used for
            the input-cube-changed checks below.

      Returns:
         int: how many layers this attempt must actually process.
      """
      mismatched = [
         (key, recorded.get(key), current.get(key))
         for key in _CONFIG_SHAPE_KEYS
         if recorded.get(key) != current.get(key)
      ]
      if mismatched:
         details = '\n'.join(
            f'  {key}: recorded {recorded!r}, this attempt {now!r}'
            for key, recorded, now in mismatched
         )
         raise RuntimeError(
            f'This attempt\'s parameters disagree with those recorded at '
            f'{self._config_path()} by the attempt that started this '
            f'store:\n{details}\n'
            f'Progress markers are keyed on (variable, chunk index), which '
            f'only means anything under the original parameters -- resuming '
            f'with different ones would leave chunks from the old chunk '
            f'grid in place and silently corrupt the store. Re-run with the '
            f'original parameters, or delete both {self.base} and the '
            f'output store to start over.'
         )

      recorded_layers = recorded['total_layers']
      if cube_total_layers < recorded_layers:
         raise RuntimeError(
            f'The input cube now resolves to {cube_total_layers} layer(s), '
            f'fewer than the {recorded_layers} this store was already '
            f'templated for; the declared shape could never be filled. '
            f'Delete both {self.base} and the output store to rebuild from '
            f'the cube as it stands now.'
         )

      for label, index in (('first', 0), ('last', recorded_layers - 1)):
         recorded_value = recorded[f'{label}_time']
         current_value = str(time_values[index])
         if recorded_value != current_value:
            raise RuntimeError(
               f'The input cube\'s {label} layer within the recorded range '
               f'changed (layer {index}: recorded {recorded_value}, now '
               f'{current_value}). Granules are ordered by mid_date, so this '
               f'means a granule was inserted into (not just appended after) '
               f'the already-written range, and every chunk written so far '
               f'now covers different granules than its marker claims. '
               f'Delete both {self.base} and the output store to rebuild.'
            )

      if cube_total_layers > recorded_layers:
         logging.warning(
            f'The input cube grew from {recorded_layers} to '
            f'{cube_total_layers} layer(s) since this store was templated. '
            f'Processing only the recorded {recorded_layers} -- the output '
            f'store\'s shape is already frozen, so the {cube_total_layers - recorded_layers} '
            f'newly appended layer(s) CANNOT be added by resuming. Rebuild '
            f'from scratch (delete {self.base} and the output store) if you '
            f'need them.'
         )

      return recorded_layers

   @staticmethod
   def build_config(
      output_store, params, total_layers, time_values=None
   ):
      """Assembles the run-configuration record write_config() persists.

      Args:
         output_store (str): recorded as-is.
         params (dict): must supply time_chunk, xy_chunk, time_chunk_1d,
            xy_shard_multiplier, num_layers.
         total_layers (int): recorded as-is.
         time_values (np.ndarray, optional): if given, contributes the
            first/last 'time' values bounding `total_layers` (omitted where
            unavailable to the caller).
      """
      config = {
         'output_store': output_store,
         'time_chunk': params['time_chunk'],
         'xy_chunk': params['xy_chunk'],
         'time_chunk_1d': params['time_chunk_1d'],
         'xy_shard_multiplier': params['xy_shard_multiplier'],
         'num_layers': params['num_layers'],
         'total_layers': total_layers,
      }
      if time_values is not None:
         config['first_time'] = str(time_values[0])
         config['last_time'] = str(time_values[total_layers - 1])

      return config

   def prune_var_markers(self, var_names):
      """Delete every named variable's per-chunk/_SUCCESS markers, keeping
      {base}/_SUCCESS and run_config.json -- called only after
      mark_complete(), so an interrupted prune still leaves the run
      recognizable as finished.

      Args:
         var_names (list of str): variables to prune.
      """
      for var_name in var_names:
         _remove_prefix(self.s3, f'{self.base}/{var_name}')

      logging.info(
         f'Pruned per-variable/per-chunk markers under {self.base} '
         f'(kept {SUCCESS_MARKER} and {RUN_CONFIG_NAME})'
      )

   def remove_entirely(self, var_names):
      """Delete this progress base in full. Unlike prune_var_markers()
      alone, this also removes run_config.json and _SUCCESS, but only ever
      removes _SUCCESS LAST, so an attempt killed mid-cleanup still reads as
      complete (is_complete() checks only _SUCCESS) instead of triggering a
      pointless redo of already-correct data.

      Args:
         var_names (list of str): passed through to prune_var_markers() for
            the per-variable part.

      Only safe for a base never consulted again once complete (e.g. an
      update transition, where "up to date" is decided by the store's live
      shape, not a marker) -- NOT creation's top-level base, which relies
      on prune_var_markers() keeping _SUCCESS forever so a rerun can
      short-circuit.
      """
      self.prune_var_markers(var_names)
      _remove_prefix(self.s3, self._config_path())
      _remove_prefix(self.s3, self._success_path())
      logging.info(f'Removed {self.base} entirely')


def _log_resume_or_fresh(s3, output_store):
   """Log fresh-vs-resuming, based on whether the output store already has
   content.

   Args:
      s3 (s3fs.S3FileSystem): filesystem to check.
      output_store (str): the cube store to check.

   Used instead of deep_copy_cube.resolve_output_store()'s strict S3 guard
   when resumability is on: after a spot termination, Batch resubmits the
   same command, so existing partial content is expected, not a mistake.
   """
   s3_path = output_store.replace(utils.S3_PREFIX, '', 1)
   if _marker_exists(s3, s3_path):
      logging.info(
         f'{output_store} already has content but no {SUCCESS_MARKER} '
         'marker -- treating this as a resume of a previously interrupted '
         'run; already-done (variable, chunk) work will be skipped.'
      )
   else:
      logging.info(f'{output_store} does not exist yet; starting a fresh run')
