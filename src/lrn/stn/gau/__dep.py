import numpy as np
import sklearn as skl

from matplotlib import pyplot as plt
from numpy.random import sample, choice
from sklearn.gaussian_process import GaussianProcessRegressor as GPReg
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as ConKer
