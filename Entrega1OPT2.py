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

import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize

# Cargar los datos desde el archivo CSV
data = pd.read_csv("BostonHousing.csv")

# Número de filas y columnas en los datos
n, m = data.shape

# Crear el problema de programación lineal
prob = LpProblem("Recta_Regresion_L1", LpMinimize)

# Definir las variables
variables = [LpVariable("coef_" + str(i), lowBound=None) for i in range(1, m)]
intercepto = LpVariable("intercepto", lowBound=None)

# Nuevas variables para las desviaciones positivas y negativas
positivas = [LpVariable(f"positivas_{i}", lowBound=0) for i in range(n)]
negativas = [LpVariable(f"negativas_{i}", lowBound=0) for i in range(n)]

# Definir la función objetivo
prob += lpSum(positivas) + lpSum(negativas)

# Restricciones de las desviaciones
for i in range(n):
    prob += data['medv'][i] - lpSum(variables[j-1] * data.iloc[i, j] for j in range(1, m)) - intercepto <= positivas[i]
    prob += lpSum(variables[j-1] * data.iloc[i, j] for j in range(1, m)) + intercepto - data['medv'][i] <= negativas[i]

# Resolver el problema
prob.solve()

# Extraer los coeficientes y el intercepto
coeficientes = [v.varValue for v in variables]
intercepto_valor = intercepto.varValue

# Imprimir la ecuación de la recta resultante
print("Ecuación de la recta:")
print(f"medv = {intercepto_valor} + {' + '.join([f'{coeficientes[i]} * {data.columns[i+1]}' for i in range(len(coeficientes))])}")

# Calcular el error de la recta
error = sum(positivas[i].varValue + negativas[i].varValue for i in range(n))
print("Error de la recta:", error)



#REPRESENTACION RESULTADO ANTERIOR


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus

# Cargar los datos desde el archivo CSV
data = pd.read_csv("BostonHousing.csv")

# Número de filas y columnas en los datos
n, m = data.shape

# Crear el problema de programación lineal
prob = LpProblem("Recta_Regresion_L1", LpMinimize)

# Definir las variables
variables = [LpVariable("coef_" + str(i), lowBound=None) for i in range(1, m)]
intercepto = LpVariable("intercepto", lowBound=None)

# Nuevas variables para las desviaciones positivas y negativas
positivas = [LpVariable(f"positivas_{i}", lowBound=0) for i in range(n)]
negativas = [LpVariable(f"negativas_{i}", lowBound=0) for i in range(n)]

# Definir la función objetivo
prob += lpSum(positivas) + lpSum(negativas)

# Restricciones de las desviaciones
for i in range(n):
    prob += data['medv'][i] - lpSum(variables[j-1] * data.iloc[i, j] for j in range(1, m)) - intercepto <= positivas[i]
    prob += lpSum(variables[j-1] * data.iloc[i, j] for j in range(1, m)) + intercepto - data['medv'][i] <= negativas[i]

# Resolver el problema
prob.solve()

# Verificar si la solución es óptima
if LpStatus[prob.status] != 'Optimal':
    print("No se pudo encontrar una solución óptima.")
    exit()

# Extraer los coeficientes y el intercepto
coeficientes = [v.varValue for v in variables]
intercepto_valor = intercepto.varValue

# Calcular las predicciones de la recta de regresión
x_vals = data['lstat']
y_vals = intercepto_valor + np.dot(data.iloc[:, 1:].values, coeficientes)

# Crear la figura y los ejes
plt.figure(figsize=(10, 6))

# Visualizar los datos (puntos)
plt.scatter(data['lstat'], data['medv'], color='blue', label='Datos reales')

# Visualizar la recta de regresión
plt.plot(x_vals, y_vals, color='red', label='Recta de regresión')

# Títulos y etiquetas de los ejes
plt.title('Recta de Regresión y Datos', fontsize=16)
plt.xlabel('lstat', fontsize=14)
plt.ylabel('medv', fontsize=14)

# Mostrar la leyenda y la cuadrícula
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()


# PLANTEAMIENTO 3

#Dado un conjunto de datos (xi,yi)(xi​,yi​), donde xixi​ representa las variables explicativas y yiyi​ representa la variable objetivo (medv), buscamos encontrar los coeficientes m1,m2,...,mkm1​,m2​,...,mk​ y el término de intercepción bb de la recta y=m1x1+m2x2+...+mkxk+by=m1​x1​+m2​x2​+...+mk​xk​+b que minimiza la desviación absoluta máxima entre las predicciones y los valores reales.
#Podemos definir el problema de programación lineal de la siguiente manera:
#   Variables de Decisión:
#      m1,m2,...,mkm1​,m2​,...,mk​: Los coeficientes de las variables explicativas.
#        bb: Término de intercepción.
#        dd: Desviación absoluta máxima.

#    Función Objetivo:
#    Minimizar dd.

#    Restricciones:
#        Para cada observación (xi,yi)(xi​,yi​):
#        yi−(m1xi1+m2xi2+...+mkxik+b)≤d
#        yi​−(m1​xi1​+m2​xi2​+...+mk​xik​+b)≤d
#        (m1xi1+m2xi2+...+mkxik+b)−yi≤d
#        (m1​xi1​+m2​xi2​+...+mk​xik​+b)−yi​≤d


import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize

# Cargar los datos desde el archivo CSV
data = pd.read_csv("BostonHousing.csv")

# Número de filas y columnas en los datos
n, m = data.shape

# Crear el problema de programación lineal
prob = LpProblem("Recta_Regresion_L_inf", LpMinimize)

# Definir las variables
variables = [LpVariable("coef_" + str(i), lowBound=None) for i in range(1, m)]
intercepto = LpVariable("intercepto", lowBound=None)
desviacion_maxima = LpVariable("desviacion_maxima", lowBound=None)

# Definir la función objetivo
prob += desviacion_maxima

# Restricciones
for i in range(n):
    prob += data['medv'][i] - lpSum(variables[j-1] * data.iloc[i, j] for j in range(1, m)) - intercepto <= desviacion_maxima
    prob += lpSum(variables[j-1] * data.iloc[i, j] for j in range(1, m)) + intercepto - data['medv'][i] <= desviacion_maxima

# Resolver el problema
prob.solve()

# Extraer los coeficientes y el intercepto
coeficientes = [v.varValue for v in variables]
intercepto_valor = intercepto.varValue

# Imprimir la ecuación de la recta resultante
print("Ecuación de la recta:")
print(f"medv = {intercepto_valor} + {' + '.join([f'{coeficientes[i]} * {data.columns[i+1]}' for i in range(len(coeficientes))])}")

# Calcular el error de la recta
error = desviacion_maxima.varValue
print("Desviación absoluta máxima:", error)


#REPRESENTACION


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize

# Cargar los datos desde el archivo CSV
data = pd.read_csv("BostonHousing.csv")

# Número de filas y columnas en los datos
n, m = data.shape

# Crear el problema de programación lineal
prob = LpProblem("Recta_Regresion_L_inf", LpMinimize)

# Definir las variables
variables = [LpVariable("coef_" + str(i), lowBound=None) for i in range(1, m)]
intercepto = LpVariable("intercepto", lowBound=None)
desviacion_maxima = LpVariable("desviacion_maxima", lowBound=None)

# Definir la función objetivo
prob += desviacion_maxima

# Restricciones
for i in range(n):
    prob += data['medv'][i] - lpSum(variables[j-1] * data.iloc[i, j] for j in range(1, m)) - intercepto <= desviacion_maxima
    prob += lpSum(variables[j-1] * data.iloc[i, j] for j in range(1, m)) + intercepto - data['medv'][i] <= desviacion_maxima

# Resolver el problema
prob.solve()

# Extraer los coeficientes y el intercepto
coeficientes = [v.varValue for v in variables]
intercepto_valor = intercepto.varValue

# Crear la figura y los ejes
plt.figure(figsize=(10, 6))

# Visualizar los datos
plt.scatter(data['lstat'], data['medv'], color='blue', label='Datos reales', alpha=0.6)

# Calcular la variable predicha por la recta de regresión
x_vals = np.linspace(data['lstat'].min(), data['lstat'].max(), 100)
y_vals = intercepto_valor + coeficientes[0] * x_vals  # Suponiendo que solo hay una variable explicativa

# Visualizar la recta de regresión
plt.plot(x_vals, y_vals, color='red', label='Recta de regresión')

# Títulos y etiquetas de los ejes
plt.title('Recta de Regresión - Desviación Absoluta Máxima', fontsize=16)
plt.xlabel('lstat (Variable Explicativa)', fontsize=14)
plt.ylabel('medv (Variable Objetivo)', fontsize=14)

# Mostrar leyenda y cuadrícula
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()



