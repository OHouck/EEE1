import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# Define utility functions and their marginal utilities
def utility_func1(y):
    """Utility function 1: U(y) = 2 * sqrt(y)"""
    return 2 * np.sqrt(y)

def marginal_utility1(y):
    """Marginal utility of utility function 1"""
    return np.where(y > 0, y ** (-0.5), np.inf)

def utility_func2(y):
    """Utility function 2: U(y) = 5y - 0.05y^2"""
    return 5 * y - 0.05 * y ** 2

def marginal_utility2(y):
    """Marginal utility of utility function 2"""
    return 5 - 0.1 * y

# Initial parameters
total_stock = 1000
discount_rate = 0.05
discount_factor = 1 / (1 + discount_rate)

# Define state and action spaces
num_states = 501
num_actions = 501
stock_increment = total_stock / (num_states - 1)

state_values = np.linspace(0, total_stock, num_states)
action_values = np.linspace(0, total_stock, num_actions)

# Create matrices for all state-action pairs
state_matrix = np.repeat(state_values.reshape(num_states, 1), num_actions, axis=1)
action_matrix = np.repeat(action_values.reshape(1, num_actions), num_states, axis=0)

# Determine feasible actions (can't extract more than available stock)
feasible_actions = action_matrix <= state_matrix

# Compute utility matrices for both utility functions
utility_matrix1 = np.full((num_states, num_actions), -1e10)
utility_matrix2 = np.full((num_states, num_actions), -1e10)
utility_matrix1[feasible_actions] = utility_func1(action_matrix[feasible_actions])
utility_matrix2[feasible_actions] = utility_func2(action_matrix[feasible_actions])

# Flatten utility matrices for use in Bellman equation
utility_vector1 = utility_matrix1.flatten()
utility_vector2 = utility_matrix2.flatten()

# Compute next state indices after actions
next_states = state_matrix - action_matrix
next_state_indices = np.zeros((num_states, num_actions), dtype=int)
next_state_indices[feasible_actions] = (next_states[feasible_actions] / stock_increment).astype(int)

# Build the transition matrix for state-action pairs
row_idx = []
col_idx = []
data = []

for s in range(num_states):
    for a in range(num_actions):
        if feasible_actions[s, a]:
            next_s_idx = next_state_indices[s, a]
            flat_idx = s * num_actions + a
            row_idx.append(flat_idx)
            col_idx.append(next_s_idx)
            data.append(1)

transition_matrix = csr_matrix((data, (row_idx, col_idx)), shape=(num_states * num_actions, num_states))

# Bellman operator function
def bellman_operator(v, U_flat, trans_matrix, delta, N, nA):
    """Computes the Bellman operator for value function iteration."""
    v_next = trans_matrix.dot(v)
    B_flat = U_flat + delta * v_next
    return B_flat.reshape(N, nA)

# Value function iteration algorithm
def value_function_iteration(U_flat, trans_matrix, delta, N, nA, tol=1e-8, max_iter=1000):
    """Performs value function iteration to find optimal policy."""
    v = np.zeros(N)
    for iteration in range(max_iter):
        B_sa = bellman_operator(v, U_flat, trans_matrix, delta, N, nA)
        v_new = np.max(B_sa, axis=1)
        policy = np.argmax(B_sa, axis=1)
        if np.max(np.abs(v_new - v)) < tol:
            print(f'Converged after {iteration + 1} iterations.')
            return v_new, policy
        v = v_new
    print('Did not converge within the maximum number of iterations.')
    return v, policy

# Obtain optimal transition matrix based on policy
def get_optimal_transition_matrix(N, policy, next_state_indices):
    """Constructs the optimal transition matrix based on the policy."""
    row_indices = np.arange(N)
    col_indices = next_state_indices[np.arange(N), policy]
    data = np.ones(N)
    return csr_matrix((data, (row_indices, col_indices)), shape=(N, N))

# Solve for the first utility function
print("Solving for Utility Function 1...")
value_function1, policy1 = value_function_iteration(
    utility_vector1, transition_matrix, discount_factor, num_states, num_actions
)
optimal_transition_matrix1 = get_optimal_transition_matrix(num_states, policy1, next_state_indices)

# Solve for the second utility function
print("Solving for Utility Function 2...")
value_function2, policy2 = value_function_iteration(
    utility_vector2, transition_matrix, discount_factor, num_states, num_actions
)
optimal_transition_matrix2 = get_optimal_transition_matrix(num_states, policy2, next_state_indices)

# Simulation function for T periods
def simulate(T, initial_stock, marginal_utility_func, policy, action_values, stock_increment):
    """Simulates the extraction and price over T periods."""
    stock = initial_stock
    extraction_history = []
    stock_history = []
    price_history = []
    for t in range(T):
        stock_history.append(stock)
        state_idx = min(int(stock / stock_increment), num_states - 1)
        action_idx = policy[state_idx]
        extraction = action_values[action_idx]
        stock -= extraction
        if stock < 0:
            stock = 0
        extraction_history.append(extraction)
        price = marginal_utility_func(extraction)
        price_history.append(price)
        if stock <= 0:
            break
    return extraction_history, stock_history, price_history

# Plotting function using matplotlib
def plot_results(extraction, stock, price, title_suffix):
    """Plots the extraction, stock, and price over time."""
    periods = np.arange(1, len(extraction) + 1)
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    axs[0].plot(periods, extraction, label='Extraction', color='blue')
    axs[0].set_title(f'Extraction Over Time {title_suffix}')
    axs[0].set_xlabel('Period')
    axs[0].set_ylabel('Extraction')
    axs[0].legend()

    axs[1].plot(periods, stock[:len(periods)], label='Stock', color='green')
    axs[1].set_title(f'Stock Over Time {title_suffix}')
    axs[1].set_xlabel('Period')
    axs[1].set_ylabel('Stock')
    axs[1].legend()

    axs[2].plot(periods, price, label='Price', color='red')
    axs[2].set_title(f'Price Over Time {title_suffix}')
    axs[2].set_xlabel('Period')
    axs[2].set_ylabel('Price')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

# Simulate and plot results for utility function 1
T = 80
extraction1, stock1, price1 = simulate(
    T, total_stock, marginal_utility1, policy1, action_values, stock_increment
)
plot_results(extraction1, stock1, price1, '(Utility Function 1)')

# Simulate and plot results for utility function 2
extraction2, stock2, price2 = simulate(
    T, total_stock, marginal_utility2, policy2, action_values, stock_increment
)
plot_results(extraction2, stock2, price2, '(Utility Function 2)')
