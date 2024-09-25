'''
Universidad del Valle de Guatemala
Métodos Matemáticos 1 para la Física
Investigación sobre funciones Gamma y Beta

Autor: Isaac Solórzano Quintana

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.optimize import curve_fit

# Cargar los datos del CSV
df = pd.read_csv('Hist_gamma.csv')

# Definir la función de la distribución gamma
def gamma_distribution(x, alpha, beta):
    return (beta**alpha * x**(alpha - 1) * np.exp(-beta * x)) / gamma(alpha)

# Función para calcular el RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# Generar los gráficos
for i in range(1, 5):
    plt.figure(figsize=(10, 6))
    
    # Datos del histograma experimental
    bin_edges = df['Bin_Edge'].values 
    histogram_values = df[f'Histogram_{i}'].values 

    x_data=bin_edges

    # Ajuste de la distribución gamma
    initial_guess = [1, 1]  # Estimaciones iniciales para alpha y beta
    popt, pcov = curve_fit(gamma_distribution, x_data, histogram_values, p0=initial_guess, maxfev=10000)

    # Extraer los parámetros ajustados
    alpha_fit, beta_fit = popt

    # Calcular los valores predichos por la distribución beta ajustada
    y_pred = gamma_distribution(x_data, alpha_fit, beta_fit)

    # Calcular el RMSE
    rmse = calculate_rmse(histogram_values, y_pred)
    print(f'RMSE for Histogram {i}: {rmse:.4f}')

    # Valores x para la distribución gamma
    x = np.linspace(0, bin_edges.max(), 100)
    
    # Calcular la distribución gamma ajustada
    y_fit = gamma_distribution(x, alpha_fit, beta_fit)
    
    # Graficar el histograma experimental como scatter y unir puntos con líneas
    plt.scatter(x_data, histogram_values, color='blue', label=f'Histograma {i} (Experimental)', zorder=2)
    plt.plot(x_data, histogram_values, color='blue', alpha=0.5)  # Unir puntos con líneas

    # Graficar la distribución gamma ajustada
    plt.plot(x, y_fit, label=f'Distribución Gamma (alpha={alpha_fit:.2f}, beta={beta_fit:.2f})', color='red', zorder=1)

    # Personalizar el gráfico
    plt.title(f'Ajuste de Distribución Gamma para el Histograma {i}')
    plt.xlabel('Bin Edge')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid()

    # Guardar el gráfico como archivo (opcional)
    plt.savefig(f'histogram_gamma_distribution_fit_{i}.png')

    # Mostrar el gráfico
    plt.show()