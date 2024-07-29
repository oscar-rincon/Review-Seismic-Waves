#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:11:57 2017

@author: oscar-rincon
"""

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter

# Configuración del directorio actual y del directorio de utilidades
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utils')
sys.path.insert(0, utilities_dir)
from plotting import *  # Importar utilidades de trazado personalizadas

# Preparación de datos
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

# Configuración de la figura
width_in_inches = 75 / 25.4
height_in_inches = 50 / 25.4
plt.figure(figsize=(width_in_inches, height_in_inches))

# Trazado de datos absolutos
plt.step(df['YEAR'], df['WORKS'], linestyle='-', color='gray')
plt.ylabel('Absolute')

# Configuración del eje principal
ax = plt.gca()
ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))  # Asegurar 4 marcas principales en el eje y
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

# Configuración de los años seleccionados en el eje x
selected_years = [2010, 2013, 2016, 2019, 2022]
plt.xticks(selected_years)
plt.xlabel('Year')

# Trazado de datos relativos en un eje secundario
ax2 = ax.twinx()
ax2.step(df_new['YEAR'], df_new['RELATIVE_PERCENT'], linestyle='--', color='blue', label='Relative')
ax2.set_ylabel('Relative', color='blue')
ax2.tick_params(axis='y', colors='blue')
ax2.spines['right'].set_color('blue')
ax2.yaxis.set_minor_locator(AutoMinorLocator(n=2))
ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Asegurar 4 marcas principales en el eje y secundario
ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

# Ocultar la espina superior en ambos ejes
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Ajustar el diseño y guardar la figura
plt.tight_layout()
plt.savefig('publications_with_relative.pdf')
plt.show()