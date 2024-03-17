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

# Calcular la matriz de correlación entre las variables
correlation_matrix = data.corr()

correlation_matrix


# 2 
#MEDIANTE PROGRAMACION LINEAL MINIMIZANDO LA SUMA DE LAS DESVIACIONES ABSOLUTAS (norma l1)

import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpVariable, lpSum

data_path = '/home/alumnos/noviedo/OPT2/Entrega1/BostonHousing.csv'
datos = pd.read_csv(data_path)

# Separar variables explicativas y dependiente
X = datos.drop(columns=['medv'])  # Variables explicativas
y = datos['medv']                 # Variable dependiente

# Dimensiones del conjunto de datos
n, k = X.shape

# Creación del problema de optimización
prob = LpProblem("Minimize_L1_Norm", LpMinimize)

# Definir variables para coeficientes (incluyendo el término independiente)
beta = LpVariable.dicts("Beta", range(k + 1), lowBound=None)

# Variables de holgura para cada observación
d_plus = LpVariable.dicts("d_plus", range(n), lowBound=0)
d_minus = LpVariable.dicts("d_minus", range(n), lowBound=0)

# Función objetivo: minimizar la suma de las desviaciones absolutas
prob += lpSum(d_plus[i] + d_minus[i] for i in range(n))

# Restricciones
for i in range(n):
    xi = [1] + list(X.iloc[i, :])  # Añadir un 1 al inicio para el intercepto
    prob += lpSum(beta[j] * xi[j] for j in range(k + 1)) + d_minus[i] - d_plus[i] == y.iloc[i]

# Resolver
prob.solve()

# Coeficientes resultantes
coeficientes1 = [beta[j].varValue for j in range(k + 1)]

# Predicciones y error total
predicciones = [sum(coeficientes1[j] * xi if j > 0 else coeficientes1[j] for j, xi in enumerate([1] + list(X.iloc[i, :]))) for i in range(n)]
error_absoluto_total = sum(abs(y.iloc[i] - pred) for i, pred in enumerate(predicciones))
print("Error norma 1:", error_absoluto_total)

# Ecuación de la recta
ecuacion_recta = "y = " + " + ".join("{:.2f}{}".format(coef, ' * ' + X.columns[j-1] if j > 0 else '') for j, coef in enumerate(coeficientes1))
print("Ecuación de la recta:", ecuacion_recta)

# Dibujo
plt.figure(figsize=(10, 6))
plt.scatter(y, predicciones, color='blue', label='Predicciones vs. Reales')
plt.plot(y, y, color='red', linestyle='--', label='Línea de referencia')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs. Valores Reales')
plt.legend()
plt.grid(True)
plt.show()





# 3
# MEDIANTE PROGRAMACION LINEAL MINIMIZANDO LA DESVIACION ABSOLUTA MAXIMA (norma linfinito)

import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo CSV
datos = pd.read_csv('/home/alumnos/noviedo/OPT2/Entrega1/BostonHousing.csv')

# Separar las variables explicativas (X) y la variable dependiente (y)
X = datos.drop(columns=['medv'])
y = datos['medv']

# Preparar el problema de optimización
prob = LpProblem("Minimizar_Norma_Infinito", LpMinimize)

# Número de características + 1 para el intercepto
n_vars = X.shape[1] + 1

# Crear variables de decisión para los coeficientes y el intercepto
coeficientes = LpVariable.dicts("Coef", range(n_vars), cat='Continuous')

# Variable para la desviación máxima
max_dev = LpVariable("MaxDev", lowBound=0)

# Minimizar la máxima desviación
prob += max_dev

# Añadir las restricciones
for i in range(len(X)):
    xi = pd.Series([1]).append(X.iloc[i])
    prediction = lpSum([coeficientes[j] * xi.values[j] for j in range(n_vars)])
    prob += prediction - y.iloc[i] <= max_dev
    prob += y.iloc[i] - prediction <= max_dev

# Resolver el problema
prob.solve()

# Extraer resultados
coef_resultados = [coeficientes[j].varValue for j in range(n_vars)]
max_dev_resultado = max_dev.varValue

# Predicciones utilizando los coeficientes resultantes
X_with_intercept = pd.concat([pd.Series(1, index=X.index, name="Intercept"), X], axis=1)
predictions = X_with_intercept.dot(coef_resultados)

# Imprimir los resultados
print("Coeficientes de la regresión:", coef_resultados)
print("Máxima desviación absoluta:", max_dev_resultado)

# Dibujar los valores reales vs. los predichos
plt.figure(figsize=(10, 6))
plt.scatter(y, predictions, alpha=0.5)
plt.title("Valores reales vs. Valores predichos")
plt.xlabel("Valores reales de medv")
plt.ylabel("Valores predichos de medv")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()

# 4 
#MEDIANTE METODO MINIMOS CUADRADOS USANDO LA FUNCION LineearRegression() (norma l2)

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar los datos desde el archivo CSV
data = pd.read_csv("BostonHousing.csv")

# Separar las variables explicativas (X) y la variable objetivo (y)
X = data.drop(columns=['medv'])
y = data['medv']

# Inicializar el modelo de regresión lineal
model = LinearRegression()

# Ajustar el modelo a los datos
model.fit(X, y)

# Obtener los coeficientes de la recta de regresión
coeficientes = model.coef_
intercepto = model.intercept_

# Imprimir la ecuación de la recta resultante
print("Ecuación de la recta:")
print("medv =", intercepto, "+", " + ".join([f"{coeficientes[i]} * {X.columns[i]}" for i in range(len(coeficientes))]))

# Calcular el error de la recta (error cuadrático medio)
y_pred = model.predict(X)
error = mean_squared_error(y, y_pred)
print("Error cuadrático medio de la recta:", error)


#REPRESENTACION RESULTADOS

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo CSV
data = pd.read_csv("BostonHousing.csv")

# Separar las variables explicativas (X) y la variable objetivo (y)
X = data.drop(columns=['medv'])
y = data['medv']

# Inicializar el modelo de regresión lineal
model = LinearRegression()

# Ajustar el modelo a los datos
model.fit(X, y)

# Obtener las predicciones del modelo
y_pred = model.predict(X)

# Crear la figura y los ejes
plt.figure(figsize=(10, 6))

# Visualizar los datos (puntos)
plt.scatter(y, y_pred, color='blue', label='Datos reales vs. Predicciones')

# Visualizar la recta de regresión
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Recta de regresión')

# Títulos y etiquetas de los ejes
plt.title('Recta de Regresión y Predicciones vs. Datos Reales', fontsize=16)
plt.xlabel('Valor real de medv', fontsize=14)
plt.ylabel('Predicción de medv', fontsize=14)

# Mostrar la leyenda y la cuadrícula
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()



