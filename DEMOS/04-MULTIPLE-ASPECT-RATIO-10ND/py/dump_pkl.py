#!/usr/bin/env -S python -u 

import pickle
import sys

with open(sys.argv[1], 'rb') as f:
    df = pickle.load(f)
    print(df.T.to_string(header=False))
