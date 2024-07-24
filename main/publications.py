import matplotlib.pyplot as plt
import pandas as pd

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:11:57 2017

@author: mraissi
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

 
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

# Datos originales
data = {
    'YEAR': [2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010],
    'WORKS': [5009, 4124, 2975, 2084, 1538, 1076, 856, 647, 531, 520, 478, 401, 384, 351]
}

# Nuevos datos
new_data = {
    'YEAR': [2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010],
    'RELATIVE_WORKS': [4160202, 4106360, 4040051, 3773670, 3580432, 3351082, 3234051, 3117389, 2989939, 2970369, 2935032, 2800621, 2663823, 2505650]
}

# Crear DataFrames
df = pd.DataFrame(data)
df_new = pd.DataFrame(new_data)

# Calcular la cantidad relativa (por ejemplo, como porcentaje del valor máximo)
df_new['RELATIVE_PERCENT'] = (df['WORKS'] / df_new['RELATIVE_WORKS']) 

# Convertir ancho de mm a pulgadas
width_in_inches = 90 / 25.4
height_in_inches = 50 / 25.4

# Graficar
plt.figure(figsize=(width_in_inches, height_in_inches))
plt.step(df['YEAR'], df['WORKS'], linestyle='-', color='gray')

# Configurar el eje Y izquierdo
plt.ylabel('Absolute')
ax = plt.gca()
#ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
# Ajustar los ticks del eje X para mostrar solo algunos años específicos
selected_years = [2010, 2013, 2016, 2019, 2022]  # Años seleccionados para mostrar
plt.xticks(selected_years)

# Crear un segundo eje Y para los valores relativos
ax2 = ax.twinx()
ax2.step(df_new['YEAR'], df_new['RELATIVE_PERCENT'], linestyle='--', color='blue', label='Relative')
ax2.set_ylabel('Relative', color='blue')  # Establece el color del texto del eje Y a azul
ax2.tick_params(axis='y', colors='blue')  # Cambia el color de los ticks del eje Y a azul


# Ajustes finales
plt.xlabel('Year')
#plt.xticks([df['YEAR'].iloc[0], df['YEAR'].iloc[-1]])
plt.tight_layout()
plt.savefig('publications_with_relative.png')
plt.show()