TOOLS   := ../../../TOOLS
PYTHON  := python3

.PHONY: run postproc clean all

## Run the FEM simulation
run:
	$(PYTHON) run.py

## Generate post-processing plots (FEM vs analytical)
postproc:
	$(TOOLS)/postproc.py

## Remove simulation outputs
clean:
	rm -rf outputs/

## Run simulation then post-process
all: run postproc
