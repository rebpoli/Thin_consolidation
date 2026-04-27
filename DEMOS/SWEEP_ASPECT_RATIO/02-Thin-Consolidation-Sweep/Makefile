TOOLS   := ../../../TOOLS
PYTHON  := python3
N_JOBS  ?= 6

.PHONY: gen run postproc clean all dry-run rerun-failed

## Generate per-case config.yaml files in runs/
gen:
	$(PYTHON) py/gen_configs.py

## Run up to N_JOBS cases in parallel (override with: make run N_JOBS=4)
run:
	$(TOOLS)/run_sweep.py --n-jobs $(N_JOBS)

## Generate plots
postproc:
	$(TOOLS)/postproc.py

## Preview cases without writing or running
dry-run:
	$(PYTHON) py/gen_configs.py --dry-run
	$(TOOLS)/run_sweep.py --dry-run

## Re-run only failed/missing cases
rerun-failed:
	$(TOOLS)/run_sweep.py --rerun-failed

## Remove all generated configs and outputs
clean:
	rm -rf runs/ log/ png/

## Full pipeline: generate → run → post-process
all: gen run postproc
