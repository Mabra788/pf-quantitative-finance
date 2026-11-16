import pandas as pd
import numpy as np
import yfinance as yf

# descargar precios desde yahoo finance
tickers = ['IVV', 'ILF', 'SOXX'] # SP 500, LATAM, Semiconductores (IA)
vector_w = [0.5, 0.3, 0.2]
vector_w = np.array(vector_w).reshape(-1,1)
df_prices = yf.download(tickers, start='2024-01-01', end= '2026-09-30')['Close']
df_prices = df_prices[tickers]


# calcular retornos diarios
df_ret = df_prices.pct_change().dropna()

# calcular matriz varianza covarianza
df_cov = df_ret.cov()
#print(df_cov)

# calcular varianza del portafolio
w = vector_w
w_t = w.T
sigma = df_cov

#print('w', w.shape)
#print('w_t', w_t.shape)
#print('sigma', sigma.shape)

varianza = np.dot(sigma,w)
varianza = np.dot(w_t,varianza)[0][0]
#print(varianza)

desv = varianza **(1/2)
volatilidad_anual = desv * (252**(1/2)) #anualizar volatilidad
print(f'La volatilidad anual del portafolio es: {volatilidad_anual*100:.04}%')

# calcular la media del portafolio 
vector_ret = df_ret.mean()
vector_ret = vector_ret.values
vector_ret = vector_ret.reshape(-1,1)
mu = np.dot(w_t, vector_ret)[0][0]
mu_anual = mu * 252
#print(f'La media anual del portafolio es: {mu_anual*100:.04}%')

# calcular value at risk (VaR) al 95% de confianza
vae_95 = mu_anual - 1.64*volatilidad_anual
print(f'La VaR anual del portafolio es: {vae_95*100:.04}%')

# FIRMA: lQ6w1EN1OpqkQGAAhffG9zCUtZ0MJOj5Jxn5mmZAoUHnPl6Cz4ViI26IHJGHj1d3q2HC9wt2Qm9qxDQ7BcuhHGBb8TmyVF3T
