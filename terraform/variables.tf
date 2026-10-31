variable "shared_network" {
  description = "Docker network that MLflow (Compose) runs on"
  type        = string
  default     = "mlops-loan-default_default"
}
