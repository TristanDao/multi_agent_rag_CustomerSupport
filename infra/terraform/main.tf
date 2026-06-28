terraform {
  required_version = ">= 1.6.0"
}

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "asia-southeast1"
}

variable "db_password" {
  description = "PostgreSQL password (use a secret, not literal)"
  type        = string
  sensitive   = true
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Cloud SQL (PostgreSQL) — small instance for MVP
resource "google_sql_database_instance" "rag_pg" {
  name             = "rag-pg"
  region           = var.region
  database_version = "POSTGRES_16"
  settings {
    tier = "db-f1-micro"
    ip_configuration {
      ipv4_enabled = true
      authorized_networks {
        name  = "all"
        value = "0.0.0.0/0"  # restrict in production
      }
    }
  }
  deletion_protection = false
}

resource "google_sql_database" "retail" {
  name     = "retail_db"
  instance = google_sql_database_instance.rag_pg.name
}

resource "google_sql_user" "retail" {
  name     = "retail"
  instance = google_sql_database_instance.rag_pg.name
  password = var.db_password
}

# Cloud Run service for the FastAPI backend
resource "google_cloud_run_service" "rag_backend" {
  name     = "rag-backend"
  location = var.region
  template {
    spec {
      containers {
        image = "asia-southeast1-docker.pkg.dev/${var.project_id}/rag/backend:latest"
        ports { container_port = 8000 }
        env {
          name  = "DATABASE_URL"
          value = "postgresql+psycopg2://retail:${var.db_password}@${google_sql_database_instance.rag_pg.public_ip_address}:5432/retail_db"
        }
        env {
          name  = "QDRANT_URL"
          value = "http://qdrant.internal:6333"
        }
      }
    }
  }
  traffic { percent = 100; latest_revision = true }
}

output "backend_url" {
  value = google_cloud_run_service.rag_backend.status[0].url
}
