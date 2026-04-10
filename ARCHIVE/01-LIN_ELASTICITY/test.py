#!/usr/bin/env -S python

from dolfinx.fem import functionspace  # Click `functionspace`
from ufl import div, grad  # Click `div` and `grad`
import numpy as np  # Click `numpy`

print(functionspace, div, grad, np)
