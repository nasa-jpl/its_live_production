#!/usr/bin/env python3
"""CDK app: Spot compute environments, job queues, and job definitions
for the ITS_LIVE deep-copy cubes Batch workload. Two size tiers: the
default 32 GB-class one, and a paired 64Gb queue/job definition for
cubes whose source icechunk repo has more layers than
run_deep_copy_cube_batch.py's NUM_LAYERS_64GB_THRESHOLD. Plus a separate
32Gb compute environment/queue and job definition for virtual cubes.

Deploy with:  cdk diff  ->  cdk deploy
"""
import aws_cdk as cdk
from aws_cdk import (
   Duration,
   Size,
   Stack,
   aws_batch as batch,
   aws_ec2 as ec2,
   aws_ecs as ecs,
   aws_iam as iam,
)
from constructs import Construct

# ---------------------------------------------------------------------------
# Copy these from the existing console-created environment
# (awsv2 batch describe-compute-environments --region us-west-2)
# ---------------------------------------------------------------------------
ACCOUNT = "849259517355"
REGION = "us-west-2"
VPC_ID = "vpc-8f40e5e9"  # TODO: aws ec2 describe-subnets --subnet-ids subnet-91cdc4d8
SUBNET_IDS = ["subnet-91cdc4d8", "subnet-e225d284", "subnet-c7d15d9c", "subnet-18e2f430"]
SECURITY_GROUP_ID = "sg-056471276aa2ce97f"
INSTANCE_ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/ecsInstanceRole"
# Existing instance profile wrapping the instance role (the "Instance role"
# shown in the existing compute environment is actually this profile ARN).
INSTANCE_PROFILE_ARN = f"arn:aws:iam::{ACCOUNT}:instance-profile/ecsInstanceRole"
JOB_ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/its-live-s3-access"
# Existing ECS task execution role (ECR pull + CloudWatch Logs).
# Set to None to let CDK create one (needs IAM create permissions).
EXECUTION_ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/its-live-s3-access"
# Root volume size for scratch space. None = AMI default (~30 GiB, no launch
# template). Set e.g. 100 to create a small launch template with an
# encrypted gp3 root volume (the KH9 template used 1000 GiB).
ROOT_VOLUME_GIB = None
# Reuse only after checking its user data works on AL2023 (NVMe mount etc.)
LAUNCH_TEMPLATE_ID = None  # e.g. "lt-0594d0c41f2a4d924"
LAUNCH_TEMPLATE_VERSION = "1"  # pin so changes to the template don't leak in

IMAGE_URI = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/its-live-deep-copy-datacube:v1.0"  # TODO: set tag
VIRTUAL_CUBE_IMAGE_URI = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/its-live-icechunk-datacube:v1.0"

# Per-tier settings: the default 32 GB-class tier, and a paired 64Gb tier for
# cubes whose source icechunk repo has more layers than
# run_deep_copy_cube_batch.py's NUM_LAYERS_64GB_THRESHOLD. "64gb"'s instance
# types are the 2xlarge (8 vCPU / 64 GB) siblings of "32gb"'s xlarge
# (4 vCPU / 32 GB) ones -- same families, double the size. Both the queue
# and the job definition must change together per tier: a job definition's
# container memory limit is enforced regardless of which instance the queue
# places it on, so pairing a bigger queue with the 30 GiB job definition
# would place the job on a 64 GB instance but still cap it at 30 GiB.
TIER_32GB = "32gb"
TIER_64GB = "64gb"
TIER_CONFIG = {
   TIER_32GB: {
      "name_prefix": "its-live-deep-copy-cube-32Gb",
      # Construct-ID suffix used when building each resource below. Both
      # tiers follow the identical pattern -- no more special-cased
      # "default tier" with a bare construct ID.
      "id_suffix": "32Gb",
      "instance_types": ["i4i.xlarge", "r5b.xlarge", "r5dn.xlarge"],
      "max_vcpus": 688,     # AWS GET 5500 / 32 = 172; 172 * 4 = 688
      "job_vcpus": 4,
      "job_memory_gib": 30,  # = 30720 MiB, same as virtual-datacube-32Gb; fits 32 GB instances
   },
   TIER_64GB: {
      "name_prefix": "its-live-deep-copy-cube-64Gb",
      "id_suffix": "64Gb",
      "instance_types": ["r5b.2xlarge", "r5n.2xlarge", "r5dn.2xlarge"],
      "max_vcpus": 344,
      "job_vcpus": 8,
      "job_memory_gib": 60,  # = 61440 MiB; fits 64 GB instances
   },
}
# Arguments passed to the image's ENTRYPOINT. "Ref::name" is filled from job
# parameters: defaults below, overridable per job with submit-job --parameters.
JOB_COMMAND = [
   "--input-store", "Ref::inputStore",
   "--output-store", "Ref::outputStore",
   "--local-staging-dir", "Ref::localStagingDir",
   "--progress-dir", "Ref::progressDir",
   "--num-load-workers", "Ref::numLoadWorkers",
]
# Defaults for the Ref:: placeholders (copy from the existing job definition's
# "parameters" field). Any placeholder with no default must be supplied at
# submit time, otherwise the literal text "Ref::name" is passed to your code.
JOB_PARAMETERS: dict[str, str] = {}

# Virtual-cube workload (its-live-icechunk-datacube image). Gets its own
# queue + compute environment, identical to the 32gb tier's except for
# max_vcpus, so it scales independently of the deep-copy jobs.
VIRTUAL_CUBE_NAME_PREFIX = "its-live-virtual-cube-32Gb"
VIRTUAL_CUBE_MAX_VCPUS = 1400

# Arguments for the virtual-cube job definition.
VIRTUAL_CUBE_JOB_COMMAND = [
   "--output-store", "Ref::outputStore",
   "--projection", "Ref::projection",
   "--polygon", "Ref::polygon",
   "--batch-size", "Ref::batchSize",
   "--threads", "Ref::threads",
   "--use-searchAPI",
]


def use_existing_instance_profile(
   ce: batch.ManagedEc2EcsComputeEnvironment, profile_arn: str
) -> None:
   """Point a compute environment at an existing instance profile and drop
   the AWS::IAM::InstanceProfile that the CDK construct always creates."""
   cfn_ce = ce.node.default_child
   cfn_ce.add_property_override("ComputeResources.InstanceRole", profile_arn)
   ce.node.try_remove_child("InstanceProfile")


class DatacubesBatchStack(Stack):
   def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
      super().__init__(scope, construct_id, **kwargs)

      # --- Existing infrastructure, referenced rather than created ---------
      vpc = ec2.Vpc.from_lookup(self, "Vpc", vpc_id=VPC_ID)
      subnets = [
         ec2.Subnet.from_subnet_id(self, f"Subnet{i}", sid)
         for i, sid in enumerate(SUBNET_IDS)
      ]
      sg = ec2.SecurityGroup.from_security_group_id(self, "Sg", SECURITY_GROUP_ID)

      # mutable=False: CDK never tries to attach policies to roles
      # owned by the IAM admin group.
      instance_role = iam.Role.from_role_arn(
         self, "InstanceRole", INSTANCE_ROLE_ARN, mutable=False
      )
      job_role = iam.Role.from_role_arn(self, "JobRole", JOB_ROLE_ARN, mutable=False)
      execution_role = (
         iam.Role.from_role_arn(self, "ExecRole", EXECUTION_ROLE_ARN, mutable=False)
         if EXECUTION_ROLE_ARN
         else None
      )

      launch_template = (
         ec2.LaunchTemplate.from_launch_template_attributes(
               self,
               "LaunchTemplate",
               launch_template_id=LAUNCH_TEMPLATE_ID,
               version_number=LAUNCH_TEMPLATE_VERSION,
         )
         if LAUNCH_TEMPLATE_ID
         else None
      )

      if ROOT_VOLUME_GIB and not launch_template:
         launch_template = ec2.LaunchTemplate(
               self,
               "RootVolumeTemplate",
               launch_template_name=f"{TIER_CONFIG[TIER_32GB]['name_prefix']}-root-volume",
               block_devices=[
                  ec2.BlockDevice(
                     device_name="/dev/xvda",  # root device on Amazon Linux
                     volume=ec2.BlockDeviceVolume.ebs(
                           ROOT_VOLUME_GIB,
                           volume_type=ec2.EbsDeviceVolumeType.GP3,
                           encrypted=True,
                           delete_on_termination=True,
                     ),
                  )
               ],
         )

      # --- Settings shared by every tier's compute environment -------------
      common = dict(
         vpc=vpc,
         vpc_subnets=ec2.SubnetSelection(subnets=subnets),
         security_groups=[sg],
         spot=True,
         allocation_strategy=batch.AllocationStrategy.SPOT_PRICE_CAPACITY_OPTIMIZED,
         minv_cpus=0,
         instance_role=instance_role,
         launch_template=launch_template,
         images=[batch.EcsMachineImage(image_type=batch.EcsMachineImageType.ECS_AL2023)],
         # CDK adds the "optimal" category unless told not to
         use_optimal_instance_classes=False,
         # Update in place instead of replacing the environment
         replace_compute_environment=False,
         terminate_on_update=False,
         update_timeout=Duration.minutes(60),
      )

      # One compute environment + queue + job definition per tier. Construct
      # IDs come from each tier's id_suffix (see TIER_CONFIG).
      for tier in TIER_CONFIG.values():
         suffix = tier["id_suffix"]
         ce_id = f"SpotX86_{suffix}"

         ce = batch.ManagedEc2EcsComputeEnvironment(
               self,
               ce_id,
               compute_environment_name=f"{tier['name_prefix']}-spot",
               instance_types=[ec2.InstanceType(t) for t in tier["instance_types"]],
               maxv_cpus=tier["max_vcpus"],
               **common,
         )
         # Use the existing instance profile instead of the one CDK creates.
         use_existing_instance_profile(ce, INSTANCE_PROFILE_ARN)

         queue = batch.JobQueue(
               self,
               f"Queue{suffix}",
               job_queue_name=f"{tier['name_prefix']}-queue",
               priority=10,
               job_state_time_limit_actions=[
                  # Cancel jobs that can never be placed instead of letting
                  # them sit in RUNNABLE forever.
                  batch.JobStateTimeLimitAction(
                     max_time=Duration.minutes(10),
                     reason=batch.JobStateTimeLimitActionsReason.JOB_RESOURCE_REQUIREMENT,
                  ),
               ],
         )
         queue.add_compute_environment(ce, 1)

         batch.EcsJobDefinition(
               self,
               f"JobDef{suffix}",
               job_definition_name=f"{tier['name_prefix']}-job",
               container=batch.EcsEc2ContainerDefinition(
                  self,
                  f"Container{suffix}",
                  image=ecs.ContainerImage.from_registry(IMAGE_URI),
                  cpu=tier["job_vcpus"],
                  memory=Size.gibibytes(tier["job_memory_gib"]),
                  command=JOB_COMMAND,
                  job_role=job_role,
                  execution_role=execution_role,
                  environment={"AWS_RETRY_MODE": "adaptive"},
               ),
               parameters=JOB_PARAMETERS or None,
               retry_attempts=3,
               retry_strategies=[
                  # Retry Spot reclamation ("Host EC2*"), exit on real failures
                  batch.RetryStrategy.of(batch.Action.RETRY, batch.Reason.SPOT_INSTANCE_RECLAIMED),
                  batch.RetryStrategy.of(batch.Action.EXIT, batch.Reason.custom(on_reason="*")),
               ],
               timeout=Duration.hours(12),
         )

         cdk.CfnOutput(self, f"Queue{suffix}Name", value=queue.job_queue_name)

      # --- Virtual-cube compute environment, queue, job definition --------
      # Compute environment/queue are the 32gb tier's settings with a larger
      # max_vcpus. The job definition differs from its-live-deep-copy-cube-32Gb-job
      # in image, command and retry behavior: 5 retry attempts so a job survives
      # up to 5 Spot reclamations before giving up.
      virtual_cube_tier = TIER_CONFIG[TIER_32GB]

      virtual_cube_ce = batch.ManagedEc2EcsComputeEnvironment(
         self,
         "VirtualCubeSpot32Gb",
         compute_environment_name=f"{VIRTUAL_CUBE_NAME_PREFIX}-spot",
         instance_types=[ec2.InstanceType(t) for t in virtual_cube_tier["instance_types"]],
         maxv_cpus=VIRTUAL_CUBE_MAX_VCPUS,
         **common,
      )
      use_existing_instance_profile(virtual_cube_ce, INSTANCE_PROFILE_ARN)

      virtual_cube_queue = batch.JobQueue(
         self,
         "VirtualCubeQueue32Gb",
         job_queue_name=f"{VIRTUAL_CUBE_NAME_PREFIX}-queue",
         priority=10,
         job_state_time_limit_actions=[
               # Cancel jobs that can never be placed instead of letting
               # them sit in RUNNABLE forever.
               batch.JobStateTimeLimitAction(
                  max_time=Duration.minutes(10),
                  reason=batch.JobStateTimeLimitActionsReason.JOB_RESOURCE_REQUIREMENT,
               ),
         ],
      )
      virtual_cube_queue.add_compute_environment(virtual_cube_ce, 1)

      cdk.CfnOutput(self, "VirtualCubeQueue32GbName", value=virtual_cube_queue.job_queue_name)

      batch.EcsJobDefinition(
         self,
         "VirtualCubeJobDef32Gb",
         job_definition_name=f"{VIRTUAL_CUBE_NAME_PREFIX}-job",
         container=batch.EcsEc2ContainerDefinition(
               self,
               "VirtualCubeContainer32Gb",
               image=ecs.ContainerImage.from_registry(VIRTUAL_CUBE_IMAGE_URI),
               cpu=virtual_cube_tier["job_vcpus"],
               memory=Size.gibibytes(virtual_cube_tier["job_memory_gib"]),
               command=VIRTUAL_CUBE_JOB_COMMAND,
               job_role=job_role,
               execution_role=execution_role,
               environment={"AWS_RETRY_MODE": "adaptive"},
         ),
         parameters=JOB_PARAMETERS or None,
         retry_attempts=5,
         retry_strategies=[
               # Retry Spot reclamation ("Host EC2*"), exit on real failures
               batch.RetryStrategy.of(batch.Action.RETRY, batch.Reason.SPOT_INSTANCE_RECLAIMED),
               batch.RetryStrategy.of(batch.Action.EXIT, batch.Reason.custom(on_reason="*")),
         ],
         timeout=Duration.hours(12),
      )

      # Known, harmless synth warnings:
      # - subnets imported by ID have no route table info; nothing here uses it
      # - the execution role (its-live-s3-access) already has ECR pull access
      cdk.Annotations.of(self).acknowledge_warning(
         "@aws-cdk/aws-ec2:noSubnetRouteTableId",
         "Imported subnets; route tables are not used by Batch",
      )
      cdk.Annotations.of(self).acknowledge_warning(
         "@aws-cdk/aws-ecs:ecrImageRequiresPolicy",
         "Execution role already grants ECR pull",
      )


app = cdk.App()
DatacubesBatchStack(
   app,
   "DatacubesBatchStack",
   env=cdk.Environment(account=ACCOUNT, region=REGION),  # required by Vpc.from_lookup
   # No bootstrap stack needed: this app has no file assets, and the
   # template is small enough to deploy directly with your own credentials.
   synthesizer=cdk.LegacyStackSynthesizer(),
)
cdk.Tags.of(app).add("project", "its-live")
app.synth()