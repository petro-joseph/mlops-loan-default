RUN = GIT_SHA=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown) docker compose run --rm trainer

up:        ; docker compose up -d mlflow
down:      ; docker compose down
build:     ; docker compose build trainer
train:     ; $(RUN) python src/train.py $(ARGS)
status:    ; $(RUN) python src/registry.py status
promote:   ; $(RUN) python src/registry.py promote $(V) $(ARGS)
rollback:  ; $(RUN) python src/registry.py rollback

.PHONY: up down build train status promote rollback
