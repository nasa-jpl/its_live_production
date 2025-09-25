#! /bin/bash
# Call the script: pass AWS_PROFILE for the account to cancel running Batch jobs for
# ./cancel_running_jobs.bash saml-pub


for i in $(awsv2 --profile saml-pub batch list-jobs --profile $1 --job-queue datacube-spot-8vCPU-64GB --job-status running --output text --query jobSummaryList[*].[jobId])
do
echo "Deleting Job: $i"
awsv2 --profile saml-pub batch terminate-job --profile $1 --job-id $i --reason "Terminating job."
echo "Job $i deleted"
done
