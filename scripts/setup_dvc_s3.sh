#!/usr/bin/env bash
# Usage: ./scripts/setup_dvc_s3.sh my-bucket-name [region]
set -euo pipefail
BUCKET="${1:?usage: $0 <bucket-name> [region]}"
REGION="${2:-us-east-1}"

[ -d .dvc ] || dvc init
dvc remote add -d -f storage "s3://${BUCKET}/dvc-store"
dvc remote modify storage region "${REGION}"
# Credentials are NOT stored in the repo: DVC uses your AWS env vars / ~/.aws profile.
echo "Remote configured:"; dvc remote list
