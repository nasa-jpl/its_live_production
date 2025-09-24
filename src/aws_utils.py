"""
AWS utilities.
"""
import requests


def get_instance_type():
   """Get the EC2 instance type using Instance Metadata Service (IMDSv2).

   This works only if running on an EC2 instance. It's useful for debugging
   AWS Batch jobs when some issues, like convergence problems, may be related
   to the instance type.

   If running on AWS Fargate, there is no instance metadata available.
   """
   instance_type = ''

   # Get IMDSv2 token
   token_url = "http://169.254.169.254/latest/api/token"

   try:
      token_response = requests.put(
         token_url,
         headers={'X-aws-ec2-metadata-token-ttl-seconds': '21600'},
         timeout=2
      )
      token = token_response.text

      # Get instance type
      metadata_url = "http://169.254.169.254/latest/meta-data/instance-type"
      response = requests.get(
         metadata_url,
         headers={'X-aws-ec2-metadata-token': token},
         timeout=2
      )

      instance_type = response.text
   except:
      # If running outside of EC2 (e.g. on local machine or AWS Fargate)
      pass

   return instance_type


if __name__ == "__main__":
   print(f'Got instance type: {get_instance_type()}')
