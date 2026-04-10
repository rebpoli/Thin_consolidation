TOOLS   := ../../TOOLS
PYTHON  := python3

.PHONY: gen run postproc compare clean all

## Generate configs for both sweeps
gen:
	$(MAKE) -C OAT-SEALED  gen
	$(MAKE) -C OAT-DRAINED gen

## Run both sweeps in sequence
run:
	$(MAKE) -C OAT-SEALED  run
	$(MAKE) -C OAT-DRAINED run

## Post-process each sweep individually
postproc:
	$(MAKE) -C OAT-SEALED  postproc
	$(MAKE) -C OAT-DRAINED postproc-all

## Compare sealed vs drained results
compare:
	$(PYTHON) py/compare.py

## Full pipeline for each sweep, then compare
all: gen run postproc compare

## Remove all outputs from both sweeps
clean:
	$(MAKE) -C OAT-SEALED  clean
	$(MAKE) -C OAT-DRAINED clean
	rm -rf png/
