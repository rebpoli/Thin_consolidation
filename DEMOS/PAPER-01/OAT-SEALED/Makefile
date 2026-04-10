TOOLS   := ../../../TOOLS
PYTHON  := python3

.PHONY: gen run postproc clean all

## Generate per-case config.yaml files in runs/
gen:
	$(PYTHON) py/gen_configs.py

## Run all cases in parallel (requires gen to have been run)
run:
	$(TOOLS)/run_sweep.py

## Generate sensitivity plots
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
