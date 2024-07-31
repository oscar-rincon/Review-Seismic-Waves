#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 20:11:57 2024

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

# Leer los archivos CSV, especificando que el delimitador es una coma y saltar las primeras líneas
df_waves = pd.read_csv('Scopus-50-Analyze-Year-Waves.csv', skiprows=6, delimiter=',')
df_total = pd.read_csv('Scopus-50-Analyze-Year-Total.csv', skiprows=6, delimiter=',')

# Extraer los datos de las columnas
years_waves = df_waves.iloc[:, 0].tolist()
works_waves = df_waves.iloc[:, 1].tolist()
years_total = df_total.iloc[:, 0].tolist()
works_total = df_total.iloc[:, 1].tolist()

# Crear diccionarios de datos
data_waves = {
    'YEAR': years_waves,
    'WORKS': works_waves
}

data_total = {
    'YEAR': years_total,
    'WORKS': works_total
}

# Crear diccionario de datos relativos
data_relative = {
    'YEAR': data_waves['YEAR'],
    'WORKS': np.array(data_waves['WORKS']) / np.array(data_total['WORKS'])
}

# Configuración de la figura
width_in_inches = 75 / 25.4
height_in_inches = 50 / 25.4
plt.figure(figsize=(width_in_inches, height_in_inches))

# Trazado de datos absolutos
plt.plot(data_waves['YEAR'], data_waves['WORKS'], linestyle='-', color='gray')
plt.ylabel('Absolute')

# Configuración del eje principal
ax = plt.gca()
ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))  # Asegurar 4 marcas principales en el eje y
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

# Configuración de los años seleccionados en el eje x
selected_years = [2010, 2013, 2016, 2019, 2022]
plt.xticks(selected_years)
plt.xlabel('Year')

# Trazado de datos relativos en un eje secundario
ax2 = ax.twinx()
ax2.plot(data_relative['YEAR'], data_relative['WORKS'], linestyle='--', color='blue', label='Relative')
ax2.yaxis.set_minor_locator(AutoMinorLocator(n=2))
ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Asegurar 4 marcas principales en el eje y secundario
ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax2.set_ylabel('Relative', color='blue')
ax2.tick_params(axis='y', colors='blue')
ax2.tick_params(axis='y', which='minor', colors='blue')  # Configurar ticks menores en azul
ax2.spines['right'].set_color('blue')

# Ocultar la espina superior en ambos ejes
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Ajustar el diseño y guardar la figura
plt.tight_layout()
plt.savefig('publications_absolute_relative.pdf')
plt.show()