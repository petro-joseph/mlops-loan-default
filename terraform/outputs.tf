output "api_url" {
  value = "http://127.0.0.1:${local.ports[local.env].api}"
}

output "prometheus_url" {
  value = "http://127.0.0.1:${local.ports[local.env].prometheus}"
}

output "serves_alias" {
  value = local.env
}
