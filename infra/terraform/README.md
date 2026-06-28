# Terraform — Multi-Agent RAG Cloud Deployment (GCP)

This is a minimal Cloud SQL + Cloud Run reference for Phase 13 of the plan. Adjust for production:

- Restrict `authorized_networks` to your VPC.
- Move `db_password` to Google Secret Manager.
- Set up Artifact Registry for the container image.
- Add IAM service account with least-privilege roles.

Usage:

```bash
cd infra/terraform
terraform init
terraform plan -var "project_id=YOUR_PROJECT" -var "db_password=YOUR_PASSWORD"
terraform apply -var "project_id=YOUR_PROJECT" -var "db_password=YOUR_PASSWORD"
```

After `apply`, the `backend_url` output gives the public Cloud Run URL.
