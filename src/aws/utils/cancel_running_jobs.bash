#! /bin/bash
for i in $(aws batch list-jobs --job-queue datacube-ondemand-8vCPU-64GB --job-status running --output text --query jobSummaryList[*].[jobId])
do
echo "Deleting Job: $i"
aws batch terminate-job --job-id $i --reason "Terminating job."
echo "Job $i deleted"
done
