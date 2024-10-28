# Author: Ozzy Houck
# Date: 2024-10-28
# Purpose: computational pset q1: implement value function iteration.

import numpy as np
from scipy.sparse import csr_matrix

# Set up starting values
stock = 1000

discount_rate = 0.05
delta = 1 / (1 + discount_rate)

N = 501 # number of states
nA = 501 # number of actions
step_size = stock / (N - 1)
# part a) and b)
# state space and action space are 0 to 1000 by 2
state_space = np.linspace(0, stock, N)
action_space = np.linspace(0, stock, nA)

# part (c)
def utility1(y: int) -> float:
    return 2 * np.sqrt(y)

def utility2(y: int) -> float:
    return 2 * y ** (1/2)

# both are N x nA
state_space_matrix = np.tile(state_space.reshape(N, 1), (1, nA)) 
action_space_matrix = np.tile(action_space.reshape(1, nA), (N, 1))

feasible_actions = action_space_matrix <= state_space_matrix

# part (d)
utility_matrix1 = np.where(feasible_actions, 
                            utility1(action_space_matrix), -np.inf)    
utility_matrix2 = np.where(feasible_actions,
                                utility2(action_space_matrix), -np.inf)

utility_matrix1_flat = utility_matrix1.flatten() # N * nA 1D array to use in Bellman with trnasition matrix
utility_matrix2_flat = utility_matrix2.flatten()

# get next state_indices
update_state_matrix = state_space_matrix - action_space_matrix
next_state_indices = np.zeros((N, nA), dtype=int)

# part (e)
next_state_indices[feasible_actions] = (update_state_matrix[feasible_actions] / step_size).astype(int)

# part (f)
# create  N x N * nA matrix transition matrix
row_indices = []
col_indices = []
data = []

for i in range(N):
    for j in range(nA):
        if feasible_actions[i, j]:
            next_state = next_state_indices[i, j]
            row_indices.append(i)
            col_indices.append(j * N + next_state)
            data.append(1)
transition_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(N, nA * N)) 

# part (g)
def bellman(v, U_flat, transition_matrix, delta, N, nA):
    '''Evaluated RHS of bellman before max
    - v: current value function (N, 1)
    - U_flat: flattened utility matrix (N * nA, ) 
    - transition_matrix: (N, N * nA) sparse matrix
    - delta: dicsount factor
    - N: number of states
    - nA: number of actions
    '''
    Vnext_flat = transition_matrix.dot(v)  # (N, 1)
    B_flat = U_flat + delta * Vnext_flat  # (N * nA, 1)
    B_sa = B_flat.reshape(N, nA)  # (N, nA) sa for state action
    return B_sa

# Value function iteration function
max_iterations = 1000
value_function = np.zeros(N)
utility_matrix = utility_matrix1
iteration = 0
tolerance = 1e-6
def value_function_iteration(U_flat, transition_matrix, delta, N, nA, tolerance, max_iterations):
    """
    Performs value function iteration using the Bellman function.

    Parameters:
    - U: Utility matrix (N x nA array)
    - next_state_indices: Next state indices matrix (N x nA array)
    - delta: Discount factor
    - N: Number of states
    - nA: Number of actions
    - tolerance: Convergence tolerance
    - max_iterations: Maximum number of iterations

    Returns:
    - v: Value function (N x 1 array)
    - policy: Optimal action indices for each state (N x 1 array)
    """
    v = np.zeros(N)
    for iteration in range(max_iterations):
        B_sa = bellman(v, U_flat, transition_matrix, delta, N, nA)
        v_new = np.max(B_sa, axis=1)
        policy = np.argmax(B_sa, axis=1)
        diff = np.max(np.abs(v_new - v))
        v = v_new
        if diff < tolerance:
            print(f'Converged in {iteration + 1} iterations')
            break
    else:
        print('Did not converge within max iterations')
    return v, policy

# Solve for utility function 1
print("Solving for utility function 1:")
v_u1, C_u1 = value_function_iteration(utility_matrix1, next_state_indices, delta, N, nA, tolerance, max_iterations)


# Part (h) Find optimal transition matrix
def get_optimal_transition_matrix(N, C, next_state_indices):
    """
    Constructs the optimal transition matrix Topt.

    Parameters:
    - N: Number of states
    - C: Optimal action indices for each state (array of size N)
    - next_state_indices: Next state indices matrix (N x nA array)

    Returns:
    - Topt: Optimal transition matrix (N x N sparse matrix)
    """
    row_indices = np.arange(N)
    col_indices = next_state_indices[np.arange(N), C]
    data = np.ones(N)
    Topt = csr_matrix((data, (row_indices, col_indices)), shape=(N, N))
    return Topt

Topt_u1 = get_optimal_transition_matrix(N, C_u1, next_state_indices)
