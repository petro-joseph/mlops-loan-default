# One environment of the loan-default service: the scoring API and a Prometheus
# that watches it. The Terraform workspace is the environment name, and each
# workspace has its own state, so staging can be created, changed and destroyed
# without touching anything else.
#
#   terraform workspace new staging
#   terraform plan     # what would change
#   terraform apply    # make it so
#   terraform destroy  # remove the environment
#
# MLflow (the shared registry) and production run under Docker Compose. Every
# environment talks to the same registry and serves its own alias, so the
# version tested here is exactly the one later promoted to production.

terraform {
  required_version = ">= 1.9"
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "3.0.2"
    }
  }
}

provider "docker" {}

locals {
  env  = terraform.workspace
  root = abspath("${path.module}/..")

  # Host ports per environment; the workspace must be one of these keys
  # ("Invalid index" means: terraform workspace select staging).
  # Production's 8000/9090 belong to Compose.
  ports = {
    staging = { api = 8100, prometheus = 9190 }
  }

  # Rebuild the image when the code that goes into it changes.
  source_hash = sha1(join("", [
    for f in sort(concat(["Dockerfile", "requirements.txt"], tolist(fileset(local.root, "src/*.py")))) :
    filesha1("${local.root}/${f}")
  ]))
}

# The network Compose created; MLflow is reachable on it as http://mlflow:5000.
data "docker_network" "shared" {
  name = var.shared_network
}

resource "docker_image" "api" {
  name = "loan-api:${local.env}"
  build {
    context = local.root
  }
  triggers = {
    source = local.source_hash
  }
}

resource "docker_container" "api" {
  name    = "loan-${local.env}-api"
  image   = docker_image.api.image_id
  command = ["uvicorn", "serve:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
  restart = "unless-stopped"

  env = [
    "MLFLOW_TRACKING_URI=http://mlflow:5000",
    "MODEL_ALIAS=${local.env}",
    "RELOAD_SECONDS=30",
  ]

  ports {
    internal = 8000
    external = local.ports[local.env].api
    ip       = "127.0.0.1"
  }

  networks_advanced {
    name = data.docker_network.shared.name
  }

  healthcheck {
    test     = ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
    interval = "15s"
    timeout  = "5s"
    retries  = 3
  }

  labels {
    label = "environment"
    value = local.env
  }

  # Docker reports network_mode "bridge" for containers on a user network,
  # which the provider reads as a change that forces replacement on every plan.
  lifecycle {
    ignore_changes = [network_mode]
  }
}

resource "docker_image" "prometheus" {
  name         = "prom/prometheus:v2.55.1"
  keep_locally = true # shared with production; destroy must not delete it
}

resource "docker_container" "prometheus" {
  name    = "loan-${local.env}-prometheus"
  image   = docker_image.prometheus.image_id
  restart = "unless-stopped"

  # Config is generated per environment and copied in, not bind-mounted.
  upload {
    file = "/etc/prometheus/prometheus.yml"
    content = templatefile("${path.module}/prometheus.yml.tftpl", {
      env    = local.env
      target = "${docker_container.api.name}:8000"
    })
  }
  upload {
    file    = "/etc/prometheus/alerts.yml"
    content = file("${local.root}/monitoring/alerts.yml")
  }

  ports {
    internal = 9090
    external = local.ports[local.env].prometheus
    ip       = "127.0.0.1"
  }

  networks_advanced {
    name = data.docker_network.shared.name
  }

  labels {
    label = "environment"
    value = local.env
  }

  # Docker reports network_mode "bridge" for containers on a user network,
  # which the provider reads as a change that forces replacement on every plan.
  lifecycle {
    ignore_changes = [network_mode]
  }
}
