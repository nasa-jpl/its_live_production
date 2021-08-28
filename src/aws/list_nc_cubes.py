import s3fs

aws_prefix = 'its-live-data.jpl.nasa.gov'
url_prefix = 'http://its-live-data.jpl.nasa.gov.s3.amazonaws.com'


s3_out = s3fs.S3FileSystem(anon=True)

all_datacubes = []
for each in s3_out.ls('its-live-data.jpl.nasa.gov/datacubes/v01/'):
    cubes = s3_out.ls(each)
    cubes = [each_cube for each_cube in cubes if each_cube.endswith('.nc')]
    all_datacubes.extend(cubes)

all_datacubes_str = "\n".join(all_datacubes)
print(f'Datacube in NetCDF format: {all_datacubes_str}')

alldata_str = '\n'.join(all_datacubes)
alldata_urls = alldata_str.replace(aws_prefix, url_prefix)


with open('datacubes_with_fixed_EPSG3413_nc_urls.txt', 'w') as fhandle:
    _ = fhandle.write(alldata_urls+'\n')

print(f'Number of datacubes: {len(all_datacubes)}')
