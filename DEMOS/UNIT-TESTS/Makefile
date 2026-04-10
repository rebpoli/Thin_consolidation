TOOLS := ../../TOOLS
PYTHON := python3

.PHONY: test clean

## Run all unit tests
test:
	$(PYTHON) py/config.test.py
	$(PYTHON) py/analytical.test.py
	$(PYTHON) py/mesh_generator.test.py

## Remove generated mesh artefacts
clean:
	rm -f *.msh *.vtk *.geo_unrolled
