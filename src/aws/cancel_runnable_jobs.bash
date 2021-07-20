#! /bin/bash
for i in $(aws batch list-jobs --job-queue masha-dave-test --job-status runnable --output text --query jobSummaryList[*].[jobId])
do
  echo "Cancel Job: $i"
  aws batch cancel-job --job-id $i --reason "Cancelling job."
  echo "Job $i canceled"
done
