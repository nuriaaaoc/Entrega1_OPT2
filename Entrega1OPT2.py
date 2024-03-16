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






