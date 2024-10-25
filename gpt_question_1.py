import numpy as np
import matplotlib.pyplot as plt

# Parameters
S_tot = 1000  # Total stock
delta = 1 / 1.05  # Discount factor
N = 501  # Number of discrete states
nA = 501  # Number of actions
step_size = 2

# State and action spaces
S = np.linspace(0, S_tot, N)  # State space from 0 to 1000 in steps of 2
A = np.linspace(0, S_tot, nA)  # Action space from 0 to 1000 in steps of 2

# Utility functions
def u1(y):
    return 2 * np.sqrt(y)

def u2(y):
    return 5 * y - 0.05 * y ** 2

# Construct S_mat and A_mat for vectorized operations
S_mat = np.tile(S.reshape(N, 1), (1, nA))  # N x nA
A_mat = np.tile(A.reshape(1, nA), (N, 1))  # N x nA

# Feasibility: A_mat <= S_mat
feasible_actions = A_mat <= S_mat

# Initialize utility matrices
U1 = np.where(feasible_actions, u1(A_mat), -np.inf)  # N x nA
U2 = np.where(feasible_actions, u2(A_mat), -np.inf)

# Compute next state indices
S_prime_mat = S_mat - A_mat  # N x nA
next_state_indices = np.zeros((N, nA), dtype=int)

next_state_indices[feasible_actions] = (S_prime_mat[feasible_actions] / step_size).astype(int)
# Infeasible actions remain at zero index (but are ignored due to -inf utility)

# Value function iteration for u1
V_u1 = np.zeros(N)
tol = 1e-8
max_iter = 1000
iteration = 0

while iteration < max_iter:
    iteration += 1
    Vprev = V_u1.copy()
    
    Vnext = V_u1[next_state_indices]  # N x nA
    Q = U1 + delta * Vnext  # N x nA
    
    V_u1_new = np.max(Q, axis=1)
    C_u1 = np.argmax(Q, axis=1)
    
    diff = np.max(np.abs(V_u1_new - V_u1))
    V_u1 = V_u1_new.copy()
    
    if diff < tol:
        print(f'Converged in {iteration} iterations for u1')
        break
else:
    print('Did not converge within max iterations for u1')

# Value function iteration for u2
V_u2 = np.zeros(N)
iteration = 0

while iteration < max_iter:
    iteration += 1
    Vprev = V_u2.copy()
    
    Vnext = V_u2[next_state_indices]  # N x nA
    Q = U2 + delta * Vnext  # N x nA
    
    V_u2_new = np.max(Q, axis=1)
    C_u2 = np.argmax(Q, axis=1)
    
    diff = np.max(np.abs(V_u2_new - V_u2))
    V_u2 = V_u2_new.copy()
    
    if diff < tol:
        print(f'Converged in {iteration} iterations for u2')
        break
else:
    print('Did not converge within max iterations for u2')

# Simulation for T=80 periods
T = 80  # Number of periods to simulate

# Simulation for u1
stock_u1 = np.zeros(T + 1)
extraction_u1 = np.zeros(T)
price_u1 = np.zeros(T)
state_index_u1 = np.zeros(T + 1, dtype=int)

# Initial state index for stock 1000
state_index_u1[0] = int(S_tot / step_size)
stock_u1[0] = S[state_index_u1[0]]

for t in range(T):
    current_state_index = state_index_u1[t]
    action_index = C_u1[current_state_index]
    extraction_u1[t] = A[action_index]
    stock_u1[t + 1] = stock_u1[t] - extraction_u1[t]
    next_state_index = next_state_indices[current_state_index, action_index]
    state_index_u1[t + 1] = next_state_index
    
    # Price as marginal utility
    if extraction_u1[t] > 0:
        price_u1[t] = 1 / np.sqrt(extraction_u1[t])
    else:
        price_u1[t] = np.inf  # Set to infinity if extraction is zero

# Simulation for u2
stock_u2 = np.zeros(T + 1)
extraction_u2 = np.zeros(T)
price_u2 = np.zeros(T)
state_index_u2 = np.zeros(T + 1, dtype=int)

state_index_u2[0] = int(S_tot / step_size)
stock_u2[0] = S[state_index_u2[0]]

for t in range(T):
    current_state_index = state_index_u2[t]
    action_index = C_u2[current_state_index]
    extraction_u2[t] = A[action_index]
    stock_u2[t + 1] = stock_u2[t] - extraction_u2[t]
    next_state_index = next_state_indices[current_state_index, action_index]
    state_index_u2[t + 1] = next_state_index
    
    # Price as marginal utility
    price_u2[t] = 5 - 0.1 * extraction_u2[t]  # u2'(y) = 5 - 0.1*y

# Plotting the results
time = np.arange(T)

plt.figure(figsize=(12, 5))

# Plot optimal extraction paths
plt.subplot(1, 2, 1)
plt.plot(time, extraction_u1, label='Utility Function 1')
plt.plot(time, extraction_u2, label='Utility Function 2')
plt.xlabel('Time')
plt.ylabel('Optimal Extraction')
plt.title('Optimal Extraction Path')
plt.legend()

# Plot price paths
plt.subplot(1, 2, 2)
plt.plot(time, price_u1, label='Utility Function 1')
plt.plot(time, price_u2, label='Utility Function 2')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Price Path')
plt.legend()

plt.tight_layout()
plt.show()
