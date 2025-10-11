#!/bin/bash

# Compilar el archivo .tex con xelatex
xelatex main_review_seismic_waves.tex

# Ejecutar bibtex para procesar las referencias
bibtex main_review_seismic_waves.aux

# Compilar dos veces más con xelatex para asegurar que todas las referencias se resuelvan correctamente
xelatex main_review_seismic_waves.tex
xelatex main_review_seismic_waves.tex