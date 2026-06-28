variable "project_id" { type = string }
variable "region"    { type = string default = "asia-southeast1" }
variable "db_password" { type = string sensitive = true }
