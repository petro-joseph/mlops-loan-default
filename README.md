# MLOps: loan-default model lifecycle

A small, complete practice of the operational side of machine learning: tracking
experiments, versioning models in a registry, promoting through a quality gate,
and rolling back. The model is a loan-default classifier on the public
[German Credit](https://www.openml.org/d/31) dataset (1,000 loans).

The model itself is deliberately simple. The point is the controls around it.

## Stack

| Piece | Tool |
|---|---|
| Experiment tracking + model registry | MLflow 2.17 (SQLite backend, proxied artifact store) |
| Training | scikit-learn pipeline (scaling + one-hot + classifier) |
| Runtime | Docker Compose; Python 3.12 inside the containers |

## Lifecycle

```
train ──> registered version (vN) ──promote (quality gate)──> @production
                                     <──────rollback─────────
```

- **Every run is traceable:** parameters, validation metrics (AUC, F1,
  accuracy), git commit, and the data it saw (source, row count, content hash).
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

## Run it

```bash
make up                                             # MLflow on 127.0.0.1:5000
make build
make train ARGS="--model logreg --C 1.0"            # -> v1
make train ARGS="--model rf --n-estimators 300 --max-depth 8"   # -> v2
make status
make promote V=1
make promote V=2
make rollback                                       # production back to v1
```

The MLflow UI listens on localhost only. From another machine:
`ssh -L 5000:localhost:5000 <server>` and open http://localhost:5000.

## Roadmap

- [x] Experiment tracking and model registry
- [x] Promotion with a quality gate, rollback, audit trail
- [ ] Serving: FastAPI container that loads `@production`, with Prometheus `/metrics`
- [ ] Monitoring: Prometheus scrape and an input-drift check against training data
- [ ] Infrastructure as Code: the stack defined in Terraform
- [ ] CI: tests, and a pipeline that blocks promotion on a quality regression
