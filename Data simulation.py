import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# drift coefficent
mu = 0.03
# number of steps
n = 5
# time in years
T = 5
# number of sims
M = 10000
# initial stock price
S0 = 1
# volatility
sigma = 0.075

# calc each time step
dt = T/n

# simulation using numpy arrays
St = np.exp(
    (mu - sigma ** 2 / 2) * dt
    + sigma * np.random.normal(0, np.sqrt(dt), size=(M,n)).T
)

# include array of 1's
St = np.vstack([np.ones(M), St])

# multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0).
St = S0 * St.cumprod(axis=0)
df_1=pd.DataFrame(St)
df_2=pd.DataFrame(St)






with pd.ExcelWriter(r'C:\Users\Magnus\Desktop\PFFE\asset1.xlsx') as writer:
    df_1.to_excel(writer,sheet_name='df_1')
with pd.ExcelWriter(r'C:\Users\Magnus\Desktop\PFFE\asset2.xlsx') as writer:
    df_2.to_excel(writer,sheet_name='df_2')