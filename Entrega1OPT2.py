import gurobipy as gp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pulp import *

# Cargar el conjunto de datos
data = pd.read_csv('BostonHousing.csv')

# Mostrar las primeras filas del conjunto de datos
print(data.head())

# Estadísticas descriptivas del conjunto de datos
print(data.describe())

# Correlación entre las variables
print(data.corr())


# Histograma para la columna 'crim'
plt.figure(figsize=(10, 6))
plt.hist(data['crim'], bins=30, color='blue', edgecolor='black')
plt.title('Histograma de la Tasa de Criminalidad (crim)')
plt.xlabel('Tasa de Criminalidad')
plt.ylabel('Frecuencia')
plt.grid(axis='y', alpha=0.75)

plt.show()

# Boxplot para la columna 'zn'
plt.figure(figsize=(8, 6))
plt.boxplot(data['zn'], vert=False, patch_artist=True)
plt.title('Boxplot de la Proporción de Suelo Residencial Urbanizado')
plt.xlabel('Proporción de suelo residencial urbanizado en lotes de más de 25,000 pies cuadrados (%)')
plt.grid(axis='x', alpha=0.75)

plt.show()

# Histograma para la columna 'indus'
plt.figure(figsize=(10, 6))
plt.hist(data['indus'], bins=30, color='green', edgecolor='black')
plt.title('Histograma de la Proporción de Acres Comerciales No Minoristas por Ciudad')
plt.xlabel('Proporción de Acres Comerciales No Minoristas por Ciudad')
plt.ylabel('Frecuencia')
plt.grid(axis='y', alpha=0.75)

plt.show()

# PLANTEAMIENTO 2 
#Queremos minimizar la suma de las desviaciones absolutas (norma ℓ 1) entre el valor real de las viviendas (medv) y el valor predicho por la recta de regresión lineal. La recta de regresión lineal se puede expresar como:

#medv_pred = b0 + b1 * crim + b2 * zn + b3 * indus + b4 * chas + b5 * nox + b6 * rm + b7 * age + b8 * dis + b9 * rad + b10 * tax + b11 * pt_ratio + b12 * lstat

#Donde b0, b1, ..., b12 son los coeficientes de la recta de regresión lineal. El problema de programación lineal se puede formular de la siguiente manera:

#Minimizar: | medv - medv_pred | Sujeto a:

#    b0, b1, ..., b12 >= 0
#    b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 + b11 + b12 = 1
