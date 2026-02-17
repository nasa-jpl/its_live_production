"""
Utility variables and functions for ITS_LIVE processing.
"""

S3_PREFIX = 's3://'
HTTP_PREFIX = 'https://'

PATH_URL = ".s3.amazonaws.com"

class Coords:
   """
   Coordinates for the ITS_LIVE datasets.
   """
   # For original datacube
   MID_DATE = 'mid_date'
   X = 'x'
   Y = 'y'

   STD_NAME = {
      MID_DATE: "image_pair_center_date_with_time_separation",
      X: "projection_x_coordinate",
      Y: "projection_y_coordinate"
   }

   DESCRIPTION = {
      MID_DATE:   "midpoint of image 1 and image 2 acquisition date and time "
                  "with granule's centroid longitude and latitude as "
                  "microseconds",
      X:          "x coordinate of projection",
      Y:          "y coordinate of projection"
   }
