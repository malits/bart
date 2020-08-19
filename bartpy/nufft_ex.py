#!/usr/bin/python3

import ctypes
from ctypes import *
import os
import sys

import numpy as np

BART_PATH = os.environ['TOOLBOX_PATH']
sys.path.insert(0, os.path.join(BART_PATH, 'python'))
import cfl

bartso = CDLL('./bart.so')

dims = (c_long * 16)(1)
dims[0] = 128
dims[1] = 128
dst = np.asfortranarray(np.empty((128, 128), dtype=np.complex64))
d3 = c_bool(False)
tstrs = (c_long * 16)(0)
samples = None

out = dst.ctypes.data_as(POINTER(2 * c_float))

bartso.calc_phantom(dims, out, d3, c_bool(False), tstrs, samples)