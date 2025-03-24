#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import matplotlib.pyplot as plt
import numpy as np

import os
import glob

from matplotlib import rc

# Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':10})

# Set the font used for MathJax
rc('mathtext',**{'default':'regular'})
rc('figure',**{'figsize':(8,6)})

# rc('text',**{'usetex':True})

# plt.rcParams['text.usetex'] = True # TeX rendering


# %%


def run_Crameri2012Case1(res, eletype):
    os.system(f'python3 Ex_Crameri2012Case1_FreeSurf.py {res} {eletype}')

test_res   = [128, 256]
test_eletype = [0, 1, 2, 3]

for res in test_res:
    for eletype in test_eletype:
        run_Crameri2012Case1(res,eletype)


# %%





# %%





# %%





# %%




