import sys, os, pickle
print(os.path.dirname(sys.executable))
import time
import os
import argparse
import numpy as np
import pyemma
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import glob
import mdtraj as md
import imp
from pyemma import plots
matplotlib.rcParams.update({'font.size': 14})
print("pyemma version",pyemma.__version__)
import msmtools
import sklearn.preprocessing
from itertools import combinations
import matplotlib.gridspec as gridspec
import sklearn.preprocessing as pre
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.compat.v1.Session(config=config))
from hde import HDE, analysis
print(tf.__version__)


def select_restart_state(values, select_type, microstates, nparallel=1, parameters=None):
    if select_type == 'rand':
        #print(values,values.shape)
        len_v=values.shape[0]
        p = np.full(len_v,1.0/len_v)
        #print(microstates, p)
        return np.random.choice(microstates, p = p, size=nparallel)
    elif select_type == 'sto_inv_linear':
        inv_values = 1.0 / values
        p = inv_values / np.sum(inv_values)
        return np.random.choice(microstates, p = p, size=nparallel)
    else:
        print('ERROR: selected select_type in select_restart_state does not exist')



parser = argparse.ArgumentParser()
parser.add_argument("--Kconfig", type=str,dest="Kconfig",required=True)


args = parser.parse_args()
Kconfig = imp.load_source('Kconfig', args.Kconfig)
print(Kconfig.strategy)
