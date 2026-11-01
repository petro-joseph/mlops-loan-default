# MLOps: loan-default model lifecycle

A small, complete practice of the operational side of machine learning: tracking
experiments, versioning models in a registry, promoting through a quality gate,
and rolling back. The model is a loan-default classifier on the public
[German Credit](https://www.openml.org/d/31) dataset (1,000 loans), kept in
[`dataset/`](dataset/README.md) so training is reproducible offline.

The model itself is deliberately simple. The point is the controls around it.

## Try it in one command

Needs Docker with the Compose plugin (and `make`). Everything runs in containers;
nothing is installed on your machine.

```bash
git clone https://github.com/petro-joseph/mlops-loan-default
cd mlops-loan-default
make demo
```

`make demo` builds the images, starts MLflow, the API, Prometheus and Grafana,
trains a model, promotes it to production through the quality gate, waits until
the API serves it, scores a sample loan and prints the links:

| Open | URL |
|---|---|
| MLflow: runs and the model registry | http://localhost:5000 |
| Grafana: the service dashboard | http://localhost:3001 |
| Prometheus: alert rules | http://localhost:9090/alerts |
| API docs: try `/predict` in the browser | http://localhost:8000/docs |

Then `make traffic` (normal traffic), `make drift` (shifted traffic; the drift
alerts fire within a few minutes) and `make down` to stop. If a port is taken,
override it: `API_PORT=18000 GRAFANA_PORT=3002 make demo` (also `MLFLOW_PORT`,
`PROMETHEUS_PORT`).

## Stack

| Piece | Tool |
|---|---|
| Experiment tracking + model registry | MLflow 2.17 (SQLite backend, proxied artifact store) |
| Training | scikit-learn pipeline (scaling + one-hot + classifier) |
| Serving | FastAPI + Uvicorn, Prometheus metrics |
| Monitoring | Prometheus 2.55 (scrape + alert rules), Grafana 11 (provisioned dashboard), PSI drift check |
| Runtime | Docker Compose (registry + production); Terraform 1.9 with the Docker provider (staging); Python 3.12 inside the containers |

## Lifecycle

```
train ──> registered version (vN) ──promote --to staging──> @staging     (Terraform-built environment, :8100)
                                   ──promote (quality gate)─> @production  (Compose, :8000)
                                     <──────rollback─────────
```

- **Every run is traceable:** parameters, validation metrics (AUC, F1,
  accuracy), git commit, and the data it saw (dataset file SHA-256, row count).
- **Registering is not deploying.** Training only adds a new version.
- **Deployment = moving the `production` alias.** Consumers load
  `models:/loan-default@production`, so promotion and rollback copy no files.
- **Quality gate:** a version is promoted only if `val_auc >= 0.70` and it is
  no more than 0.01 below the current production version. `--force` overrides
  the gate, and the override is recorded on the version.
- **Audit trail:** each promotion and rollback is stamped on the model version
  (time, user, reason), and the alias history is kept on the registered model.
- **Rollback skips the gate** because it returns to a version that already
  passed it and ran in production.

## Serving

The `api` service loads `models:/loan-default@production` and re-checks the
alias every 30 s. When it moves, the new version is loaded and swapped in with
no restart; if loading fails, the version in memory keeps serving and
`loan_model_reloads_total{result="failed"}` goes up. Every response says which
model version produced it.

| Endpoint | |
|---|---|
| `POST /predict` | `{"loans": [{...20 applicant fields...}]}` -> probability of default and a decision (`approve` below 0.5, else `refer` to a credit officer). Missing or malformed fields -> 422. |
| `GET /health` | 200 with the serving version; 503 until a model is loaded |
| `GET /metrics` | Prometheus: request count and latency per endpoint and status, decisions, distribution of predicted default probability, serving version, reload results |

```bash
make serve        # build and start the API on 127.0.0.1:8000
make predict      # score examples/loan.json
make metrics
make promote V=2  # within 30 s, /predict answers with "model_version": "2"
make rollback     # and back to v1
```

## Monitoring

**Two kinds of problem, two owners.** Platform alerts (API down, errors,
latency, a promoted version that fails to load) belong to whoever runs the
service. Drift alerts belong to the model owner: when applicants stop looking
like the training data, the scores may no longer be trustworthy even though
nothing is broken.

**Drift (PSI).** Training saves `reference.json` with the run: the share of
training rows in each decile (numeric features) or category, plus the
distribution of validation scores. The API keeps the last 500 applications and
every 30 s publishes each feature's Population Stability Index against the
serving version's reference: `loan_feature_psi{feature}` and `loan_score_psi`.
Credit-risk rule of thumb: < 0.1 stable, 0.1-0.25 watch, > 0.25 shifted.

| Alert | Fires when | Owner |
|---|---|---|
| `LoanApiDown` | Prometheus can't scrape the API for 1 min | platform |
| `LoanApiErrors` | > 5% of predictions fail (5xx) for 5 min | platform |
| `LoanApiSlow` | p95 latency > 500 ms for 5 min | platform |
| `LoanModelReloadFailing` | a promoted version failed to load in the last 10 min | platform |
| `LoanInputDrift` | any feature PSI > 0.25 for 2 min | model |
| `LoanScoreDrift` | score PSI > 0.25 for 2 min | model |

Grafana (`127.0.0.1:3001`, opens on the dashboard) shows serving version,
traffic by status, p50/p95 latency, decisions and referral rate, and PSI per
feature. Alerts are visible in Prometheus (`127.0.0.1:9090/alerts`); routing
them to email or chat would add Alertmanager.

```bash
make up                # mlflow, api, prometheus, grafana
make traffic S=120     # 2 minutes of applicants like the training data
make drift S=150       # a shifted population: larger, longer, education loans
make alerts            # LoanInputDrift / LoanScoreDrift go PENDING, then FIRING
make traffic S=120     # normal traffic refills the window; alerts resolve
```

Measured on the AWS server: normal traffic kept every feature under 0.05. The drifted
run pushed `credit_amount` to 2.3, `duration` to 2.1 and `purpose` to 2.0
while untouched features (e.g. `age`) stayed at 0.03, the referral rate went
from 39% to 78%, and both drift alerts fired.

Versions trained before this step (v1-v3) have no `reference.json`; the API
still serves them, with drift metrics off.

## Environments (Terraform)

Production and the shared MLflow registry run under Docker Compose. Staging is
defined in `terraform/` and created with Terraform: its own API (serving the
`staging` alias) and its own Prometheus, on the same network as the registry.
Every environment reads the same registry, so the version tested in staging is
byte-for-byte the one promoted to production; only the alias differs.

The Terraform workspace is the environment, with its own state. Ports per
environment live in `locals.ports`; adding one (say `test`) is one line there.

```bash
make staging-up                    # terraform init + workspace staging + apply (asks for "yes")
make promote V=4 ARGS="--to staging"
make staging-predict               # staging API on 127.0.0.1:8100; Prometheus on :9190
make staging-plan                  # "No changes": what is running matches the code
make staging-down                  # terraform destroy; production is untouched
```

Checked on the AWS server: a second `plan` reports no changes; deleting the staging
Prometheus container by hand makes `plan` show exactly that one resource to
create, and `apply` restores it. Images are rebuilt only when the Dockerfile,
requirements or `src/*.py` change (`triggers.source`).

## CI (GitHub Actions)

Every push and pull request runs four jobs (`.github/workflows/ci.yml`):

| Job | What it checks |
|---|---|
| Lint and unit tests | `ruff`; `pytest` covers the quality gate rules, the PSI maths, and the API (with a stand-in model, no MLflow needed) |
| Model quality gate | trains the candidate (`train.py` defaults) against a throwaway registry, then `check_quality.py` fails the build if its `val_auc` is below 0.70 or more than 0.01 below `quality/baseline.json`; the comparison is written to the run summary |
| Terraform | `fmt -check` and `validate` |
| Image | the service image builds |

The gate rules live in one place, `src/gate.py`, used by both CI and
`registry.py promote`. CI never touches the real registry: it proves a change
doesn't make the model worse, and promotion stays a separate, recorded decision.
The baseline is a file in the repo, so accepting a better model (raising it) is
itself a reviewed change.

```bash
make test                  # lint + unit tests in a container, like CI
```

## Run it

```bash
make up                                             # MLflow :5000, API :8000, Prometheus :9090, Grafana :3001
make build
make train ARGS="--model logreg --C 1.0"            # -> v1
make train ARGS="--model rf --n-estimators 300 --max-depth 8"   # -> v2
make status
make promote V=1                                    # production (default)
make promote V=2
make rollback                                       # production back to v1
```

The MLflow UI listens on localhost only. From another machine:
`ssh -L 5000:localhost:5000 -L 3001:localhost:3001 -L 9090:localhost:9090 <server>`.

## Roadmap

- [x] Experiment tracking and model registry
- [x] Promotion with a quality gate, rollback, audit trail
- [x] Serving: FastAPI container that follows `@production`, with Prometheus `/metrics`
- [x] Monitoring: Prometheus alerts, Grafana dashboard, PSI drift check against training data
- [x] Infrastructure as Code: a staging environment defined in Terraform
- [x] CI: lint, unit tests, a model-quality gate against a committed baseline, Terraform checks, image build
