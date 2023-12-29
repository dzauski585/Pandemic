
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
S0 = (0.9999 * N) - I0 - R0  #99% susceptible minus the infected and recovered
y0 = [S0, E0, I0, R0, D0]

# Parameters
beta = 1  #contact rate or infection rate each interaction will infect for example 0.25 is 1 in 4 interactions infect
# beta = c*p where c = average number of contacts a susceptible person makes per time and p = probability that susceptable people become infected with contact to infected person

'''
lambda = beta * (I / N) or the rate of movement from exposed to infected
'''
sigma = 0.5   #incubation
gamma = 0.4  #mean recovery rate in 1 over days
mu = 0.01 #death rate technically mu is normal death rate of population and disease mu is (mu / mu + beta)

Rnaught = beta / gamma

critical = 1 - (1 / Rnaught)

# Time vector
t = np.linspace(0, 200, 200)  

# Solve the SEIR model equations
solution = odeint(SEIR_model, y0, t, args=(N, beta, sigma, gamma, mu))

# Extract results
S, E, I, R, D = solution.T

# Print the Rnaught and critical threshhold
print(f"The Rnaught is {Rnaught}")
print(f"The critical vaccination or herd immunity threshold is {critical}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, D, label='Dead')
plt.xlabel('Time (days)')
plt.ylabel('Population in Billions')
plt.title('SEIRD Model Simulation')
plt.legend()
plt.grid(True)
plt.show()
