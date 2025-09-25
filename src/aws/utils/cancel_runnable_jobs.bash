#! /bin/bash
for i in $(aws --profile saml-pub batch list-jobs --job-queue datacube-ondemand-8vCPU-64GB --job-status runnable --output text --query jobSummaryList[*].[jobId])
do
  echo "Cancel Job: $i"
  echo "Job $i canceled"
  aws --profile saml-pub batch cancel-job --job-id $i --reason "Cancelling job."
done
