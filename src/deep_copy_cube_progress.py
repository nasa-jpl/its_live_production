"""
Resumability support for deep_copy_cube_per_var_chunk.py: tracks, via tiny
marker objects on S3, which (variable, chunk) work a prior attempt already
finished, so a retried run (e.g. after an AWS Batch spot termination) can
skip it instead of rebuilding the whole cube from scratch.

Split out of deep_copy_cube_per_var_chunk.py into its own module so the
marker/config-guard mechanics can be read (and tested) independently of the
actual copy logic. See _Progress's own docstring for the marker layout, and
deep_copy_cube_per_var_chunk.py's module docstring for how it's used in
context.
"""
import json
import logging
import os
import tempfile

import s3fs

import itslive_utils
import utils


# Progress-marker filenames. Markers are plain zero-byte S3 objects; only
# their presence matters, never their content. See _Progress for the layout
# and the three levels they form.
SUCCESS_MARKER = '_SUCCESS'
RUN_CONFIG_NAME = 'run_config.json'

# Every object this pipeline writes to its-live-data carries this ACL (see
# _s3_copy()'s --acl, and the same flag in ~20 other scripts in this repo):
# the job commonly runs under an account that isn't the bucket owner, and
# without it the bucket owner can't manage or delete what we wrote. The
# marker objects below are written via s3fs's put_object rather than the AWS
# CLI, so they need it threaded in through s3_additional_kwargs (verified:
# s3fs's _call_s3 merges s3_additional_kwargs into the put_object call, and
# 'ACL' survives its _filter_kwargs whitelist).
BUCKET_OWNER_ACL = 'bucket-owner-full-control'

# Keys of the run parameters that define what a chunk marker MEANS -- see
# _Progress.validate_config() for why a mismatch on any of them has to be a
# hard failure rather than something to work around.
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
   return s3.exists(marker_path)


@itslive_utils.retry_decorator(max_retries=5)
def _write_marker(s3, marker_path):
   s3.touch(marker_path)


@itslive_utils.retry_decorator(max_retries=5)
def _remove_prefix(s3, prefix):
   """Recursively delete `prefix`, tolerating its absence."""
   if s3.exists(prefix):
      s3.rm(prefix, recursive=True)


@itslive_utils.retry_decorator(max_retries=5)
def _read_json(s3, path):
   """Read and parse a small JSON object from S3, or return None if it isn't
   there."""
   if not s3.exists(path):
      return None

   with s3.open(path, 'r') as fhandle:
      return json.load(fhandle)


def _s3_copy(local_path, s3_path, recursive=True):
   """Copy a local file or directory to S3 via the AWS CLI, matching
   deep_copy_cube.upload_local_staging_dir()'s mechanism (retried, subprocess-
   based) but scoped to a single path instead of a whole store, so it can be
   called once per chunk without waiting for the whole run to finish.

   Parameters
   ----------
   local_path : str
      Local file or directory to copy.
   s3_path : str
      Destination s3:// URL.
   recursive : bool
      True (default) for a directory copy, False for a single file.
   """
   command_line = ["aws", "s3", "cp"]
   if recursive:
      command_line.append("--recursive")
   command_line += [local_path, s3_path, "--acl", "bucket-owner-full-control"]
   itslive_utils.s3_copy_using_subprocess(command_line, os.environ.copy())


class _Progress:
   """Tracks which (variable, chunk) work a prior attempt already finished,
   so a retried run (e.g. after an AWS Batch spot termination) can skip it
   instead of rebuilding the whole cube from scratch.

   Markers live under --progress-dir, deliberately OUTSIDE the cube's own
   zarr store: the store is a published data product and has no business
   carrying this pipeline's bookkeeping. `base` appends the output store's
   own name to --progress-dir, so one shared --progress-dir can safely serve
   every cube in a batch without two cubes' markers colliding (and
   validate_config() cross-checks the recorded output_store on top of that,
   so a collision fails loudly rather than silently skipping real work).

   Three marker levels, checked cheapest-first:
   - {base}/_SUCCESS: every variable's every chunk is done. Checked once,
      before anything else in deep_copy_cube_per_var_chunk() -- if present,
      the whole run is a no-op.
   - {base}/{var_name}/_SUCCESS: this one variable's every chunk is done.
      Checked once per variable, before its chunk loop -- lets a resumed run
      skip an already-finished variable with a single S3 exists() call
      instead of one per chunk.
   - {base}/{var_name}/{chunk_index}.done: this one (variable, chunk) pair
      is done -- written+uploaded, or legitimately radar-skipped (see
      RADAR_ONLY_VARS; a skip is just as "resolved" as an upload). Only
      consulted for a variable not already marked fully done.

   Plus {base}/run_config.json, which records the parameters that give
   chunk_index its meaning -- see validate_config().

   Completion is always an EXPLICIT signal, never inferred from which chunk
   objects happen to exist at the output store: `aws s3 cp --recursive` of a
   sharded chunk isn't atomic, so a kill mid-copy can leave some shard
   objects written and others missing, and zarr answers any missing shard
   with fill_value rather than an error. Inferring "done" from partial state
   would therefore ship a permanently corrupt chunk. Marker absence, by
   contrast, only ever costs a redo: every write here is idempotent (a
   region/array write overwrites cleanly, `aws s3 cp` overwrites its
   destination, and a re-uploaded chunk rewrites the identical key set), so
   a false negative is always safe and only a false positive is dangerous.

   No locking: two attempts running concurrently against the same
   --progress-dir are benign on the S3 side (identical bytes, last writer
   wins) but must NOT share a --local-staging-dir, since each attempt's
   resolve_output_store() would rmtree the other's staging directory
   mid-run.
   """

   def __init__(self, s3, base):
      self.s3 = s3
      self.base = base

   @classmethod
   def create(cls, progress_dir, output_store):
      """Build a _Progress rooted at `progress_dir`/{output store name}.

      The filesystem is constructed with BUCKET_OWNER_ACL so every marker
      object it writes matches the rest of this pipeline's output.
      """
      s3 = s3fs.S3FileSystem(s3_additional_kwargs={'ACL': BUCKET_OWNER_ACL})
      store_name = os.path.basename(output_store.rstrip('/'))
      return cls(s3, f'{progress_dir.rstrip("/")}/{store_name}')

   def _success_path(self):
      return f'{self.base}/{SUCCESS_MARKER}'

   def _var_success_path(self, var_name):
      return f'{self.base}/{var_name}/{SUCCESS_MARKER}'

   def _chunk_path(self, var_name, chunk_index):
      return f'{self.base}/{var_name}/{chunk_index}.done'

   def _config_path(self):
      return f'{self.base}/{RUN_CONFIG_NAME}'

   def is_complete(self):
      return _marker_exists(self.s3, self._success_path())

   def mark_complete(self):
      _write_marker(self.s3, self._success_path())

   def var_is_done(self, var_name):
      return _marker_exists(self.s3, self._var_success_path(var_name))

   def mark_var_done(self, var_name):
      _write_marker(self.s3, self._var_success_path(var_name))

   def chunk_is_done(self, var_name, chunk_index):
      return _marker_exists(self.s3, self._chunk_path(var_name, chunk_index))

   def mark_chunk_done(self, var_name, chunk_index):
      _write_marker(self.s3, self._chunk_path(var_name, chunk_index))

   def read_config(self):
      return _read_json(self.s3, self._config_path())

   def write_config(self, config):
      """Persist this run's parameters, written the same way as every other
      object in this pipeline (a local file + _s3_copy's retried, ACL-
      setting AWS CLI invocation) rather than through s3fs, so there's one
      mechanism to reason about for real payload objects."""
      with tempfile.TemporaryDirectory() as tmp_dir:
         local_path = os.path.join(tmp_dir, RUN_CONFIG_NAME)
         with open(local_path, 'w') as fhandle:
            json.dump(config, fhandle, indent=3, sort_keys=True)

         _s3_copy(local_path, self._config_path(), recursive=False)

      logging.info(f'Recorded run configuration at {self._config_path()}')

   def validate_config(self, recorded, current, cube_total_layers, time_values):
      """Reconcile this attempt's parameters against the recorded ones and
      return the number of layers this attempt must actually process.

      Chunk markers are keyed only on (var_name, chunk_index), which is
      meaningless without the parameters that decide what a chunk_index
      COVERS. If those changed, the skeleton re-upload would overwrite the
      output store's metadata with a new chunk grid while chunk objects
      written under the old grid stayed in place (`aws s3 cp` never
      deletes), and the markers would suppress rewriting them -- yielding a
      store that opens cleanly and reads garbage. So any mismatch in
      _CONFIG_SHAPE_KEYS is a hard failure, not something to paper over.

      A CHANGED INPUT CUBE is handled differently, because it doesn't
      require an operator mistake -- a byte-identical command line can see
      a different layer count if the virtual cube's icechunk repo gained
      granules between attempts. The output store's shape was frozen by the
      first attempt's template, so this attempt processes exactly the
      recorded total_layers and ignores anything appended since (logged
      loudly). It cannot do the same if the cube SHRANK below that -- the
      declared shape could never be filled -- so that's a hard failure.

      Clamping alone would be a false sense of safety, though: granules are
      ordered by mid_date, so a granule inserted mid-sequence (rather than
      appended) shifts every later layer, and already-written chunks would
      silently no longer correspond to the same granules. The recorded
      first/last 'time' values are re-checked to catch exactly that -- free,
      since 'time' is an eagerly-loaded index coordinate.

      Parameters
      ----------
      recorded : dict
         The configuration the first attempt persisted, as returned by
         read_config().
      current : dict
         This attempt's configuration, from build_config().
      cube_total_layers : int
         Layer count this attempt resolved from the input cube (after
         --num-layers).
      time_values : np.ndarray
         The input cube's full 'time' coordinate.

      Returns
      -------
      int
         Number of layers this attempt must process.

      Raises
      ------
      RuntimeError
         On any mismatch that makes the recorded markers unsafe to trust.
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
      """Assemble the run-configuration record. `params` supplies the
      chunking parameters (any mapping with the _CONFIG_SHAPE_KEYS-relevant
      entries); `time_values`, when given, contributes the first/last 'time'
      values bounding the processed range."""
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
      """Delete the per-variable and per-chunk markers, keeping only
      {base}/_SUCCESS and {base}/run_config.json.

      Called only after mark_complete(), so an interrupted prune still
      leaves the store recognizable as finished on the next attempt.
      """
      for var_name in var_names:
         _remove_prefix(self.s3, f'{self.base}/{var_name}')

      logging.info(
         f'Pruned per-variable/per-chunk markers under {self.base} '
         f'(kept {SUCCESS_MARKER} and {RUN_CONFIG_NAME})'
      )


def _log_resume_or_fresh(s3, output_store):
   """Log whether this run is starting fresh or resuming a previously
   interrupted attempt, based on what's already at `output_store`.

   Used in place of deep_copy_cube.resolve_output_store()'s S3 branch when
   resumability is enabled -- unlike that one, this never raises: after a
   spot termination AWS Batch resubmits the identical command, so
   `output_store` already holding partial content from the killed attempt
   is the expected, resumable case, not a mistake to refuse. Without
   --progress-dir there's no resume to speak of and the strict guard is
   used instead, so that safety net is only given up where it would
   actively break the feature.

   Parameters
   ----------
   s3 : s3fs.S3FileSystem
   output_store : str
      Final s3:// destination.
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
