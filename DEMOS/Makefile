.PHONY: tests validation paper-01 all clean unit-tests

## Run unit tests and all validation cases
tests:
	$(MAKE) -C UNIT-TESTS test
	$(MAKE) -C VALIDATION all

## Run unit tests only
unit-tests:
	$(MAKE) -C UNIT-TESTS test

## Run validation demos only
validation:
	$(MAKE) -C VALIDATION all

## Run Paper-01 full pipeline (gen → run → postproc → compare)
paper-01:
	$(MAKE) -C PAPER-01 all

## Run everything
all: tests paper-01

## Clean all demo outputs
clean:
	$(MAKE) -C UNIT-TESTS  clean
	$(MAKE) -C VALIDATION  clean
	$(MAKE) -C PAPER-01    clean
