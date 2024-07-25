#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:11:57 2017

@author: oscar-rincon
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter

# Configuration for LaTeX and Helvetica font
def figsize(width_scale=1, height_scale=1, nplots=1):
    fig_width_pt = 390.0  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * width_scale  # width in inches
    fig_height = nplots * fig_width * golden_mean * height_scale  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(1.0),     # default fig size of 0.9 textwidth
    "pgf.preamble": r'\usepackage[utf8x]{inputenc},\usepackage[T1]{fontenc},\usepackage{amssymb},\usepackage{amsfonts}',
        # plots will be generated using this preamble
    }
mpl.rcParams.update(pgf_with_latex)

# Plotting functions
def newfig(width, height, nplots=1):
    fig = plt.figure(figsize=figsize(width, height, nplots))
    ax = fig.add_subplot(111)
    return fig, ax

def savefig(filename, crop=True):
    if crop:
        plt.savefig('{}'.format(filename), bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig('{}'.format(filename))

# Data preparation
data = {
    'YEAR': [2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010],
    'WORKS': [5009, 4124, 2975, 2084, 1538, 1076, 856, 647, 531, 520, 478, 401, 384, 351]
}

new_data = {
    'YEAR': [2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010],
    'RELATIVE_WORKS': [4160202, 4106360, 4040051, 3773670, 3580432, 3351082, 3234051, 3117389, 2989939, 2970369, 2935032, 2800621, 2663823, 2505650]
}

df = pd.DataFrame(data)
df_new = pd.DataFrame(new_data)
df_new['RELATIVE_PERCENT'] = (df['WORKS'] / df_new['RELATIVE_WORKS'])

# Plotting
width_in_inches = 75 / 25.4
height_in_inches = 50 / 25.4

plt.figure(figsize=(width_in_inches, height_in_inches))
plt.step(df['YEAR'], df['WORKS'], linestyle='-', color='gray')

plt.ylabel('Absolute')
ax = plt.gca()
ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))  # Ensure 4 major ticks on y-axis

# Set scientific notation for y-axis
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

selected_years = [2010, 2013, 2016, 2019, 2022]
plt.xticks(selected_years)

plt.xlabel('Year')
ax2 = ax.twinx()
ax2.step(df_new['YEAR'], df_new['RELATIVE_PERCENT'], linestyle='--', color='blue', label='Relative')
ax2.set_ylabel('Relative', color='blue')
ax2.tick_params(axis='y', colors='blue')
ax2.spines['right'].set_color('blue')
ax2.yaxis.set_minor_locator(AutoMinorLocator(n=2))
ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Ensure 4 major ticks on secondary y-axis

# Set scientific notation for secondary y-axis
ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('publications_with_relative.pdf')
plt.show()