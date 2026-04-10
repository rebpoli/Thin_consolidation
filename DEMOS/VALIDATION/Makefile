.PHONY: all run postproc clean

SUBDIRS := 01-Traditional-Consolidation 02-Thin-Disc-Consolidation

## Run all validation simulations then post-process
all:
	$(MAKE) -C 01-Traditional-Consolidation all
	$(MAKE) -C 02-Thin-Disc-Consolidation  all

## Run simulations only
run:
	$(MAKE) -C 01-Traditional-Consolidation run
	$(MAKE) -C 02-Thin-Disc-Consolidation  run

## Post-process only (requires outputs to exist)
postproc:
	$(MAKE) -C 01-Traditional-Consolidation postproc
	$(MAKE) -C 02-Thin-Disc-Consolidation  postproc

## Remove all outputs
clean:
	$(MAKE) -C 01-Traditional-Consolidation clean
	$(MAKE) -C 02-Thin-Disc-Consolidation  clean
