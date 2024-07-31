#!/bin/bash

# Compilar el archivo .tex con xelatex
xelatex Review_Elastic_Waves.tex

# Ejecutar bibtex para procesar las referencias
bibtex Review_Elastic_Waves.aux

# Compilar dos veces más con xelatex para asegurar que todas las referencias se resuelvan correctamente
xelatex Review_Elastic_Waves.tex
xelatex Review_Elastic_Waves.tex