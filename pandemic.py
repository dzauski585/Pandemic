
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# SEIR model equations
def SEIR_model(y, t, N, beta, sigma, gamma, mu):
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I - mu * I
    dRdt = gamma * I
    dDdt = mu * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]


"""
Initial conditions (such as S0, I0, and R0)  
Initial condition are set such that S0 = 99%, which indicates the proportion of susceptible individuals when the simulation starts. 
I0 is set to 1%, which indicates proportion of infected individuals to be 1% when the simulation starts. 
R0 is set to 0% which is expected that there are are no recovered individuals when the simulations start.
"""
# Initial conditions
N = 8000000000 #world pop
E0 = 0 #exposed
I0 = 1 #infected
R0 = 0.00  # recovered
D0 = 0 #dead
S0 = (0.99 * N) - I0 - R0  #99% susceptible minus the infected and recovered
y0 = [S0, E0, I0, R0, D0]

# Parameters
beta = 8  #contact rate 
sigma = 0.1   #incubation
gamma = 0.05  #mean recovery rate
mu = 1 #death rate

# Time vector
t = np.linspace(0, 200, 200)  

# Solve the SEIR model equations
solution = odeint(SEIR_model, y0, t, args=(N, beta, sigma, gamma, mu))

# Extract results
S, E, I, R, D = solution.T

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, D, label='Dead')
plt.xlabel('Time (days)')
plt.ylabel('Population in Billions')
plt.title('SEIR Model Simulation')
plt.legend()
plt.grid(True)
plt.show()