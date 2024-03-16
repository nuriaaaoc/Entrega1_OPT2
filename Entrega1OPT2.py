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

# Grafico de barras para la columna 'chas'
# Contando la cantidad de viviendas cerca y lejos del río
chas_counts = data['chas'].value_counts()
# Gráfico
plt.figure(figsize=(8, 6))
chas_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribución de Viviendas Cerca y Lejos del Río')
plt.xlabel('Cerca del Río (1 = Sí, 0 = No)')
plt.ylabel('Cantidad de Viviendas')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.75)
plt.show()

# Histograma para la columna 'nox'
plt.figure(figsize=(10, 6))
plt.hist(data['nox'], bins=30, color='purple', edgecolor='black')
plt.title('Histograma de la Concentración de Óxido Nítrico (nox)')
plt.xlabel('Concentración de Óxido Nítrico (partes por 10 millones)')
plt.ylabel('Frecuencia')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Histograma para la columna 'rm'
plt.figure(figsize=(10, 6))
plt.hist(data['rm'], bins=30, color='orange', edgecolor='black')
plt.title('Histograma del Número Medio de Habitaciones por Vivienda')
plt.xlabel('Número Medio de Habitaciones por Vivienda')
plt.ylabel('Frecuencia')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Histograma para la columna 'age'
plt.figure(figsize=(10, 6))
plt.hist(data['age'], bins=30, color='teal', edgecolor='black')
plt.title('Histograma de la Proporción de Unidades Construidas Antes de 1940')
plt.xlabel('Proporción de Unidades Construidas Antes de 1940 (%)')
plt.ylabel('Frecuencia')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Histograma para la columna 'dis'
plt.figure(figsize=(10, 6))
plt.hist(data['dis'], bins=30, color='brown', edgecolor='black')
plt.title('Histograma de las Distancias Ponderadas a Cinco Centros de Empleo de Boston')
plt.xlabel('Distancias Ponderadas a Cinco Centros de Empleo')
plt.ylabel('Frecuencia')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Gráfico de barras para la variable 'rad'
# Contando la frecuencia de cada índice de accesibilidad a autopistas radiales
rad_counts = data['rad'].value_counts().sort_index()
# Gráfico
plt.figure(figsize=(10, 6))
rad_counts.plot(kind='bar', color='darkcyan', edgecolor='black')
plt.title('Distribución del Índice de Accesibilidad a Autopistas Radiales')
plt.xlabel('Índice de Accesibilidad a Autopistas Radiales')
plt.ylabel('Cantidad de Viviendas')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.75)
plt.show()

# Histograma para la columna 'tax'
plt.figure(figsize=(10, 6))
plt.hist(data['tax'], bins=30, color='darkgreen', edgecolor='black')
plt.title('Histograma del Impuesto sobre los Bienes Inmuebles por $10,000')
plt.xlabel('Impuesto sobre los Bienes Inmuebles por $10,000')
plt.ylabel('Frecuencia')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Histograma para la columna 'ptratio'
plt.figure(figsize=(10, 6))
plt.hist(data['ptratio'], bins=30, color='darkred', edgecolor='black')
plt.title('Histograma de la Proporción Alumnos-Profesor por Ciudad')
plt.xlabel('Proporción Alumnos-Profesor')
plt.ylabel('Frecuencia')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Histograma para la columna 'lstat'
plt.figure(figsize=(10, 6))
plt.hist(data['lstat'], bins=30, color='maroon', edgecolor='black')
plt.title('Histograma del Porcentaje de Población de Bajo Estatus')
plt.xlabel('Porcentaje de Población de Bajo Estatus')
plt.ylabel('Frecuencia')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Histograma para la columna 'medv'
plt.figure(figsize=(10, 6))
plt.hist(data['medv'], bins=30, color='navy', edgecolor='black')
plt.title('Histograma del Valor Mediano de las Viviendas (medv)')
plt.xlabel('Valor Mediano de las Viviendas (miles de dólares)')
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
