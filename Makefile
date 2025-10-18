.PHONY: init download train-% eval-% train-all eval-all aggregate plots compare gates validate test lint zip reproduce-all deploy

ENV_NAME ?= ppojax
MODELS   := fno pino bayes_deeponet divfree_fno cvae_fno
SEEDS    := 0 1 2 3 4
EPOCHS   ?= 200
N_SAMPLES ?= 16
SEED     ?= 42
TRAIN_ARGS ?=
EVAL_ARGS  ?=
DOWNLOAD_ARGS?=--dataset ns_incom --shards 512-0 --max-files 1 --pairs-per-file 1024

init:
	python -m pip install -e .

download:
	ppo-download --root ./data_cache $(DOWNLOAD_ARGS)

train-%:
	@echo "== Training $* (seed $(SEED)) =="
	ppo-train --config config.yaml --model $* --epochs $(EPOCHS) --seed $(SEED) $(TRAIN_ARGS)

eval-%:
	@echo "== Evaluating $* (seed $(SEED)) =="
	python -m src.eval --config config.yaml --model $* --seed $(SEED) --n_samples $(N_SAMPLES) $(EVAL_ARGS)

train-all:
	@for s in $(SEEDS); do \
		for m in $(MODELS); do \
			$(MAKE) train-$$m SEED=$$s; \
		done; \
	done

eval-all:
	@for s in $(SEEDS); do \
		python -m src.eval --config config.yaml --all-models --seed $$s --n_samples $(N_SAMPLES) --output results/comparison_metrics_seed$$s.json $(EVAL_ARGS); \
	done

aggregate:
	python -m analysis.compare --inputs results/comparison_metrics_seed*.json --out results/compare.md --csv results/compare.csv --bootstrap 1000

plots:
	python -m analysis.compare_plots --csv results/compare.csv --outdir results/figures

gates:
	python -m analysis.gates --csv results/compare.csv

compare: train-all eval-all aggregate plots gates

validate:
	python -m src.qa.validate_physics --results results

test:
	pytest -q

lint:
	pylint src models constraint_lib || true

zip:
	python -m src.packaging.make_zip --out final_solution.zip

reproduce-all: init download train-all eval-all aggregate plots gates zip

deploy:
	docker build -t ppojax:latest .
