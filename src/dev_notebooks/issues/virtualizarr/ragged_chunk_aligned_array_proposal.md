# Proposal: Support for Chunk-Aligned Ragged Arrays in VirtualiZarr

## Executive Summary

This proposal addresses the limitation that VirtualiZarr cannot handle chunk-aligned ragged arrays (arrays with missing chunks). The good news is that **VirtualiZarr already has 70% of the infrastructure needed** - it just needs to be properly connected.

### The Problem

VirtualiZarr currently cannot handle chunk-aligned ragged arrays (arrays with missing chunks that should be filled with `fill_value`). This affects:
- **ITSCube objects** with chunk-aligned ragged layers ([issue #883](https://github.com/zarr-developers/VirtualiZarr/issues/883))
- **NASA SWOT satellite data** with variable-shaped files ([issue #22](https://github.com/zarr-developers/VirtualiZarr/issues/22))
- Any datasets where some chunks are missing
- Concatenating files with different shapes

### What Already Exists

VirtualiZarr already has the core infrastructure:
- ✅ `MISSING_CHUNK_PATH = ""` sentinel for missing chunks
- ✅ ChunkManifest can store and track missing chunks  
- ✅ Zarr v3 metadata includes `fill_value`
- ✅ Methods like `get_entry()` return `None` for missing chunks

### What Needs to Be Done

Three areas need updates to connect the existing infrastructure:

1. **ManifestStore** - Return `fill_value` when reading missing chunks (instead of trying to read from non-existent file)
2. **Concatenation** - Preserve missing chunks when combining arrays
3. **Indexing** - Handle missing chunks correctly during slicing operations

### Implementation Phases

| Phase | Description |
|-------|-------------|
| Phase 1 | Core reading support |
| Phase 2 | Concatenation support |
| Phase 3 | Public API helpers |

### Key Benefits

- ✅ **Backwards compatible** - No breaking changes to existing code
- ✅ **Memory efficient** - Chunk-aligned ragged arrays use less memory than dense arrays
- ✅ **Fast I/O** - No network requests for missing chunks (instant fill_value generation)
- ✅ **Natural API** - Fits existing VirtualiZarr patterns and conventions
- ✅ **Solves real use cases** - ITSCube, SWOT, and other sparse datasets become virtualizable

### Recommended Next Steps

1. Review this proposal with VirtualiZarr maintainers
2. Get feedback on the approach and priorities
3. Implement Phase 1 (core reading) first as a proof of concept
4. Add Phase 2 (concatenation) to enable multi-file use cases
5. Add Phase 3 (API helpers) for better user experience

---

## Problem Statement

VirtualiZarr currently cannot handle chunk-aligned ragged arrays - arrays where some chunks are missing and should be filled with the Zarr `fill_value` when accessed. This is a critical limitation for several use cases:

1. **ITSCube objects** - Have sparse layers where not all chunks contain data
2. **Satellite swath data** (NASA SWOT) - Different files produce slightly different shaped data each day
3. **Irregular time series** - Where some time steps may be missing for some spatial regions
4. **Padded concatenation** - Combining arrays of different shapes requires padding

Related issues:
- [#883](https://github.com/zarr-developers/VirtualiZarr/issues/883): ITSCube chunk-aligned ragged array problem
- [#22](https://github.com/zarr-developers/VirtualiZarr/issues/22): Original request for padding arrays with NaN/fill_value

## Current State Analysis

### What Already Exists

VirtualiZarr already has **partial** support for chunk-aligned ragged arrays:

1. **`MISSING_CHUNK_PATH = ""`** sentinel in `virtualizarr/manifests/manifest.py:33`
2. **ChunkManifest initialization** (line 234) treats empty paths as missing chunks
3. **`dict()` method** (line 453-474) omits missing chunks from dictionary output
4. **`get_entry()` method** (line 492-499) returns `None` for missing chunks
5. **Zarr fill_value** in ArrayV3Metadata already stores the fill value to use

### What's Missing

The infrastructure exists but missing chunks are not handled during:

1. **Array operations** - Concatenation, stacking, indexing
2. **Reading data** - ManifestStore needs to return fill_value for missing chunks
3. **Validation** - Some operations may fail when encountering missing chunks
4. **Creation** - No easy API to create sparse virtual datasets

## Proposed Solution

### Phase 1: Make Sparse Arrays Readable (Essential)

#### 1.1 Update ManifestStore to Handle Missing Chunks

**File:** `virtualizarr/manifests/store.py`

When `ManifestStore` encounters a chunk with `path == MISSING_CHUNK_PATH`, it should:
- Return a chunk filled with the array's `fill_value` instead of trying to read from a file
- Match the expected chunk shape and dtype

```python
# In ManifestStore._get_chunk_key() or similar method
def _get_chunk_key(self, key: str) -> bytes:
    entry = self.manifest.get_entry(parse_key_to_indices(key))
    
    if entry is None or entry['path'] == MISSING_CHUNK_PATH:
        # Return fill_value chunk
        chunk_shape = self._calculate_chunk_shape(key)
        fill_array = np.full(
            shape=chunk_shape,
            fill_value=self.metadata.fill_value,
            dtype=self.metadata.data_type
        )
        # Apply codecs to match expected format
        return self._encode_chunk(fill_array)
    
    # Normal path: read from file
    return self._read_from_file(entry)
```

#### 1.2 Update Indexing to Preserve Missing Chunks

**File:** `virtualizarr/manifests/indexing.py`

When indexing into a ManifestArray with missing chunks:
- Missing chunks should remain missing after slicing
- Don't try to read or validate missing chunks during indexing

```python
# In apply_indexer() or related functions
def apply_indexer(marr: ManifestArray, indexer: tuple) -> ManifestArray:
    # ... existing logic ...
    
    # When slicing the manifest, preserve missing chunks
    new_manifest = marr.manifest[chunk_grid_selector]  # This should work
    
    # ... rest of logic ...
```

#### 1.3 Fix Concatenation with Sparse Arrays

**File:** `virtualizarr/manifests/manifest.py` (likely needs a `concat()` method)

When concatenating arrays:
- Allow arrays with different numbers of chunks
- Fill gaps with missing chunk entries
- Validate that overlapping chunks have consistent data

```python
def concat_manifests(
    manifests: list[ChunkManifest],
    axis: int,
    fill_value: Any = None
) -> ChunkManifest:
    """
    Concatenate multiple chunk manifests along a given axis.
    
    Missing chunks in source manifests will be preserved as missing
    in the output manifest.
    """
    # Calculate output grid shape
    output_shape = list(manifests[0].shape_chunk_grid)
    output_shape[axis] = sum(m.shape_chunk_grid[axis] for m in manifests)
    
    # Initialize output arrays with MISSING_CHUNK_PATH
    paths = np.full(output_shape, MISSING_CHUNK_PATH, dtype=np.dtypes.StringDType())
    offsets = np.zeros(output_shape, dtype=np.uint64)
    lengths = np.zeros(output_shape, dtype=np.uint64)
    
    # Copy chunks from each manifest, offsetting along concat axis
    offset_along_axis = 0
    for manifest in manifests:
        # Build slice for this manifest's position
        slices = [slice(None)] * len(output_shape)
        slices[axis] = slice(
            offset_along_axis,
            offset_along_axis + manifest.shape_chunk_grid[axis]
        )
        
        # Copy non-missing chunks
        for idx, entry in manifest.iter_refs():
            output_idx = list(idx)
            output_idx[axis] += offset_along_axis
            output_idx = tuple(output_idx)
            
            paths[output_idx] = entry['path']
            offsets[output_idx] = entry['offset']
            lengths[output_idx] = entry['length']
        
        offset_along_axis += manifest.shape_chunk_grid[axis]
    
    return ChunkManifest.from_arrays(
        paths=paths,
        offsets=offsets,
        lengths=lengths,
        validate_paths=False  # Already validated
    )
```

### Phase 2: API for Creating Sparse Virtual Datasets

#### 2.1 Add Helper Function for Padding Arrays

**File:** `virtualizarr/xarray.py` (new public function)

```python
def pad_virtual_dataset(
    vds: xr.Dataset,
    target_shape: dict[str, int],
    fill_value: dict[str, Any] | None = None,
) -> xr.Dataset:
    """
    Pad a virtual dataset with missing chunks to match target shape.
    
    Parameters
    ----------
    vds : xr.Dataset
        Virtual dataset to pad
    target_shape : dict[str, int]
        Target size for each dimension, e.g. {'x': 1024, 'y': 1024}
    fill_value : dict[str, Any], optional
        Fill value for each variable. If not provided, uses variable's
        existing fill_value or dtype default.
    
    Returns
    -------
    xr.Dataset
        Padded virtual dataset with missing chunks in padded regions
    
    Examples
    --------
    >>> # Pad dataset to make all x/y dimensions 1024
    >>> vds_padded = pad_virtual_dataset(
    ...     vds,
    ...     target_shape={'x': 1024, 'y': 1024}
    ... )
    """
    # Implementation that modifies ManifestArrays to have larger
    # chunk grids with missing chunks in the padding region
    pass
```

#### 2.2 Enhance `open_virtual_mfdataset` to Handle Variable Shapes

**File:** `virtualizarr/xarray.py`

Update `open_virtual_mfdataset` to automatically pad when combining files of different shapes:

```python
def open_virtual_mfdataset(
    filepaths: list[str],
    ...,
    allow_padding: bool = False,
    padding_fill_value: dict[str, Any] | None = None,
) -> xr.Dataset:
    """
    ...
    
    Parameters
    ----------
    ...
    allow_padding : bool, default False
        If True, automatically pad arrays to consistent shapes when
        combining files. Missing chunks will be filled with fill_value.
    padding_fill_value : dict[str, Any], optional
        Override fill values for specific variables when padding.
    
    ...
    """
    # When allow_padding=True, detect shape mismatches and pad
    # automatically before concatenation
    pass
```

### Phase 3: Advanced Features (Optional)

#### 3.1 Sparse Array Detection and Optimization

Add utilities to:
- Detect which arrays are sparse (have missing chunks)
- Calculate storage efficiency metrics
- Optimize chunk layouts for chunk-aligned ragged arrays

#### 3.2 Partial Chunk Support

For truly irregular data, support chunks that are only partially filled:
- Chunk entry includes both the data region and the chunk region
- More complex but enables more use cases

## Implementation Plan

### Step 1: Core Reading Support
- [ ] Update `ManifestStore._get_chunk_key()` to return fill_value for missing chunks
- [ ] Add tests for reading chunk-aligned ragged arrays via ManifestStore
- [ ] Update indexing logic to preserve missing chunks

**Files modified:** `store.py`, `indexing.py`, tests

### Step 2: Concatenation Support
- [ ] Implement `concat_manifests()` function
- [ ] Update xarray concatenation operations to use new logic
- [ ] Add tests for concatenating chunk-aligned ragged arrays
- [ ] Handle coordinate variables specially (typically should not be sparse)

**Files modified:** `manifest.py`, `xarray.py`, tests

### Step 3: Public API
- [ ] Implement `pad_virtual_dataset()` helper function
- [ ] Update `open_virtual_mfdataset()` with padding options
- [ ] Documentation and examples
- [ ] Add "Working with Chunk-Aligned Ragged Arrays" section to docs

**Files modified:** `xarray.py`, docs

### Step 4: Validation and Edge Cases
- [ ] Add validation for chunk-aligned ragged array operations
- [ ] Test with real ITSCube data
- [ ] Test with NASA SWOT data
- [ ] Performance testing with large chunk-aligned ragged arrays

**Files modified:** various, tests

## Testing Strategy

### Unit Tests

```python
def test_missing_chunk_returns_fill_value():
    """Test that reading a missing chunk returns fill_value."""
    manifest = ChunkManifest(
        entries={
            "0.0": {"path": "s3://bucket/data.nc", "offset": 0, "length": 100},
            # "0.1" is missing - no entry
        },
        shape=(1, 2)
    )
    metadata = create_v3_array_metadata(
        shape=(10, 20),
        chunk_shape=(10, 10),
        data_type=np.dtype('float32'),
        fill_value=np.nan,
    )
    marr = ManifestArray(metadata=metadata, chunkmanifest=manifest)
    
    # Reading from missing chunk should return all NaN
    store = ManifestStore(marr, registry=test_registry)
    chunk_data = store["0.1"]  # Missing chunk
    assert np.all(np.isnan(decode_chunk(chunk_data)))

def test_concat_sparse_arrays():
    """Test concatenating arrays with missing chunks."""
    arr1 = create_sparse_manifest_array(shape=(10, 10), chunks=(10, 5))
    arr2 = create_sparse_manifest_array(shape=(10, 10), chunks=(10, 5))
    
    result = concat([arr1, arr2], axis=1)
    
    assert result.shape == (10, 20)
    # Verify missing chunks are preserved in output
    # ...

def test_pad_dataset_to_consistent_shape():
    """Test padding datasets to consistent shapes."""
    vds1 = create_test_virtual_dataset(shape=(100, 200))
    vds2 = create_test_virtual_dataset(shape=(100, 250))
    
    # Pad to common shape
    vds1_padded = pad_virtual_dataset(vds1, target_shape={'x': 100, 'y': 250})
    
    # Now can concatenate
    combined = xr.concat([vds1_padded, vds2], dim='time')
    assert combined.sizes == {'time': 2, 'x': 100, 'y': 250}
```

### Integration Tests

- Test with actual ITSCube NetCDF files (if available)
- Test with NASA SWOT sample data
- Test with various fill_value types (NaN, 0, custom values)
- Test roundtrip: create sparse -> write to Kerchunk -> read back

## Backwards Compatibility

This proposal maintains full backwards compatibility:

1. **Existing behavior unchanged** - Dense arrays work exactly as before
2. **Opt-in padding** - `allow_padding` parameter defaults to False
3. **No API breaks** - All existing functions keep same signatures
4. **Internal optimization** - Missing chunk handling is internal implementation detail

## Documentation Updates

Add new documentation sections:

1. **Usage Guide: "Working with Sparse Arrays"**
   - What are chunk-aligned ragged arrays
   - When to use them
   - Performance considerations
   
2. **FAQ: "Can I virtualize datasets with missing data?"**
   - Explain chunk-aligned ragged array support
   - Examples with satellite swath data
   
3. **API Reference**
   - Document new `pad_virtual_dataset()` function
   - Update `open_virtual_mfdataset()` docs with padding params

## Performance Considerations

### Memory

- Missing chunks store only 3 values (empty path, 0 offset, 0 length) = ~24 bytes
- Dense chunk storage: path string + 2 uint64s = ~100+ bytes
- **Chunk-aligned ragged arrays with many missing chunks use significantly less memory**

### I/O

- Reading missing chunks is instant (fill array in memory)
- No network requests or disk I/O for missing chunks
- **Chunk-aligned ragged arrays with many missing chunks are faster to access**

### Concatenation

- Need to iterate over existing chunks, not entire grid
- Use `iter_refs()` which skips missing chunks
- **Performance scales with number of present chunks, not grid size**

## Alternative Approaches Considered

### Alternative 1: Load Sparse Variables into Memory

**Pros:** Simple, works today  
**Cons:** Defeats purpose of virtualization, requires loading all data

### Alternative 2: Create Separate Virtual Datasets Per Region

**Pros:** Works with existing code  
**Cons:** Complex for users, doesn't match mental model of single datacube

### Alternative 3: Require Padding Files Before Virtualization

**Pros:** No VirtualiZarr changes needed  
**Cons:** Wasteful storage, defeats purpose of virtual references

**Recommendation:** The proposed solution is superior because it maintains the virtual/lazy nature while properly supporting the chunk-aligned ragged array use case.

## Success Criteria

1. ✅ ITSCube chunk-aligned ragged arrays can be virtualized and read correctly
2. ✅ NASA SWOT variable-shape files can be combined
3. ✅ Missing chunks return fill_value when accessed
4. ✅ Concatenation works with chunk-aligned ragged arrays
5. ✅ No regression in performance for dense arrays
6. ✅ Comprehensive test coverage (>90%)
7. ✅ Documentation includes examples and best practices

## References

- Issue #22: https://github.com/zarr-developers/VirtualiZarr/issues/22
- Issue #883: https://github.com/zarr-developers/VirtualiZarr/issues/883
- Zarr fill_value spec: https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#fill-value
- Kerchunk chunk-aligned ragged arrays: (if applicable)

## Questions for Maintainers

1. **Should missing chunks be allowed in coordinate variables?**
   - Probably not - coordinates define the array structure
   - But what about 2D coordinates that vary by file?

2. **How to handle fill_value differences across files?**
   - Use first file's fill_value?
   - Require all files have same fill_value?
   - Make it configurable?

3. **Should we support "truly sparse" encoding?**
   - Where we only store non-missing chunks in manifests?
   - Would save more memory but complicates indexing

4. **Integration with Zarr v3 chunk-aligned ragged array extension?**
   - Is there ongoing work on chunk-aligned ragged arrays in Zarr spec?
   - Should we coordinate with that effort?
