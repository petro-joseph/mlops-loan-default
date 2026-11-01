#!/usr/bin/env bash
# One-command demo: start the platform, train a model, promote it to
# production, wait until the API serves it, and score a sample loan.
# Needs Docker with the compose plugin. Safe to run again: each run trains
# and promotes a new version.
set -euo pipefail
cd "$(dirname "$0")/.."

API_PORT=${API_PORT:-8000}
MLFLOW_PORT=${MLFLOW_PORT:-5000}
PROMETHEUS_PORT=${PROMETHEUS_PORT:-9090}
GRAFANA_PORT=${GRAFANA_PORT:-3001}
export API_PORT MLFLOW_PORT PROMETHEUS_PORT GRAFANA_PORT
export GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
RUN="docker compose run --rm trainer"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

wait_for() {  # wait_for <description> <seconds> <command...>
  local what=$1 limit=$2; shift 2
  for _ in $(seq "$limit"); do
    if "$@" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  echo "Timed out waiting for $what. Check: docker compose logs" >&2
  exit 1
}

step "1/5 Building images and starting MLflow, the API, Prometheus and Grafana"
docker compose build api trainer
docker compose up -d mlflow api prometheus grafana
wait_for "MLflow" 90 curl -sf "localhost:$MLFLOW_PORT/health"

step "2/5 Training a model (logistic regression on the German Credit data)"
out=$($RUN python src/train.py --model logreg --C 1.0 | tee /dev/stderr)
version=$(sed -n 's/.*registered loan-default v\([0-9]*\).*/\1/p' <<<"$out")
[ -n "$version" ] || { echo "Could not read the new version from the training output" >&2; exit 1; }

step "3/5 Promoting v$version to production (the quality gate checks it first)"
$RUN python src/registry.py promote "$version" || echo "(v$version not promoted; the API keeps serving the current version)"

step "4/5 Waiting for the API to load the production model (it checks every 30 s)"
want=$($RUN python -c "from mlflow import MlflowClient; print(MlflowClient().get_model_version_by_alias('loan-default','production').version)" | tail -1)
wait_for "the API to serve v$want" 90 sh -c "curl -sf localhost:$API_PORT/health | grep -q '\"version\":\"$want\"'"
curl -s "localhost:$API_PORT/health"; echo

step "5/5 Scoring a sample loan"
curl -s -X POST "localhost:$API_PORT/predict" -H 'Content-Type: application/json' -d @examples/loan.json; echo

cat <<EOF

Done. Open:
  MLflow (runs and registry)  http://localhost:$MLFLOW_PORT
  Grafana (dashboard)         http://localhost:$GRAFANA_PORT
  Prometheus (alerts)         http://localhost:$PROMETHEUS_PORT/alerts
  API docs (try /predict)     http://localhost:$API_PORT/docs

Next: make traffic   (normal traffic, watch Grafana)
      make drift     (shifted traffic; the drift alerts fire in a few minutes)
      make down      (stop everything)
EOF
