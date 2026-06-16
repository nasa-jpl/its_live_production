"""
ITS_LIVE Test Suite
===================

Unit tests for ITS_LIVE datacube generation and processing pipeline.

Run with verbose output:
   pytest src/tests/ -v

Run specific test file:
   pytest src/tests/test_itscube_generation.py -v

Run specific test function:
   pytest src/tests/test_itscube_generation.py::test_01_datacube_generation_via_cli -v

Run specific test method of the class;
   pytest test_itscube_generation.py::TestDatacubeGeneration::test_verify_datacube_structure -vv -l

Run with detailed output and show local variables on failure:
   pytest src/tests/ -vv -l

Test Data
=========

Tests use small regions to minimize runtime and data transfer:
- Malaspina region (EPSG:3413): 100km x 100km area in Alaska
- Polygon: [[-3300000, 200000], [-3200000, 200000], [-3200000, 300000], [-3300000, 300000], [-3300000, 200000]]
- Grid cell size: 120m
- Limited to 200 granules for fast testing

Test outputs are written to `src/tests/test_output/` and automatically cleaned up after tests complete.
"""
