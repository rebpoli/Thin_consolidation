TOOLS   := ../../../TOOLS
PYTHON  := python3

.PHONY: gen run postproc plot-pq plot-ucs clean all

## Generate per-case config.yaml files in runs/
gen:
	$(PYTHON) py/gen_configs.py

## Run all cases in parallel (requires gen to have been run)
run:
	$(TOOLS)/run_sweep.py

## Generate sensitivity plots
postproc:
	$(TOOLS)/postproc.py

## Generate P-Q stress-invariant plots
plot-pq:
	$(TOOLS)/postproc.py --script py/plot_pq.py

## Generate UCS failure envelope table
plot-ucs:
	$(TOOLS)/postproc.py --script py/plot_ucs.py

## Generate all post-processing outputs
postproc-all: postproc plot-pq plot-ucs

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
all: gen run postproc-all
