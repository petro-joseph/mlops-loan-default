RUN = GIT_SHA=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown) docker compose run --rm trainer

up:        ; docker compose up -d mlflow api prometheus grafana
down:      ; docker compose down
build:     ; docker compose build trainer
train:     ; $(RUN) python src/train.py $(ARGS)
status:    ; $(RUN) python src/registry.py status
promote:   ; $(RUN) python src/registry.py promote $(V) $(ARGS)
rollback:  ; $(RUN) python src/registry.py rollback
serve:     ; docker compose up -d --build api
predict:   ; curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @examples/loan.json; echo
health:    ; curl -s localhost:8000/health; echo
metrics:   ; curl -sL localhost:8000/metrics | grep '^loan_'
monitor:   ; docker compose up -d prometheus grafana
traffic:   ; $(RUN) python src/traffic.py --seconds $(or $(S),120) --rps 5
drift:     ; $(RUN) python src/traffic.py --seconds $(or $(S),120) --rps 5 --drift
alerts:    ; curl -s localhost:9090/api/v1/alerts | python3 -c "import json,sys; [print(a['state'].upper(), a['labels']['alertname'], '-', a['annotations']['summary']) for a in json.load(sys.stdin)['data']['alerts']] or print('no alerts')"

.PHONY: up down build train status promote rollback serve predict health metrics monitor traffic drift alerts
