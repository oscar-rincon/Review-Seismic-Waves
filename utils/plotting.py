#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:11:57 2017

@author: mraissi
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#mpl.use('pgf')

# def figsize(scale, nplots = 1):
#     fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
#     inches_per_pt = 1.0/72.27                       # Convert pt to inch
#     golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
#     fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
#     fig_height = nplots*fig_width*golden_mean              # height in inches
#     fig_size = [fig_width,fig_height]
#     return fig_size
def figsize(width_scale=1, height_scale=1, nplots=1):
    fig_width_pt = 390.0  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * width_scale  # width in inches
    fig_height = nplots * fig_width * golden_mean * height_scale  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size

# Configuración de LaTeX para matplotlib
pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "xelatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "sans-serif",
    "font.serif": [],
    "font.sans-serif": ["DejaVu Sans"], # specify the sans-serif font
    "font.monospace": [],
    "axes.labelsize": 8,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": (3.15, 2.17),     # default fig size of 0.9 textwidth
    "pgf.preamble": r' \usepackage{amsmath},\usepackage{cmbright},\usepackage[utf8x]{inputenc},\usepackage[T1]{fontenc},\usepackage{amssymb},\usepackage{amsfonts},\usepackage{mathastext}',
        # plots will be generated using this preamble
    }
mpl.rcParams.update(pgf_with_latex)

# I make my own newfig and savefig functions
def newfig(width,height, nplots = 1):
    fig = plt.figure(figsize=figsize(width, height, nplots))
    ax = fig.add_subplot(111)
    return fig, ax

def savefig(filename, crop = True):
    if crop == True:
        plt.savefig('{}'.format(filename), bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig('{}'.format(filename))

## Simple plot
#fig, ax  = newfig(1.0)
#
#def ema(y, a):
#    s = []
#    s.append(y[0])
#    for t in range(1, len(y)):
#        s.append(a * y[t] + (1-a) * s[t-1])
#    return np.array(s)
#    
#y = [0]*200
#y.extend([20]*(1000-len(y)))
#s = ema(y, 0.01)
#
#ax.plot(s)
#ax.set_xlabel('X Label')
#ax.set_ylabel('EMA')
#
#savefig('ema')