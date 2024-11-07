import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# Define utility functions and their marginal utilities
def utility1(y):
    """Utility function U(y) = 2 * sqrt(y)"""
    return 2 * np.sqrt(y)

def marginal_utility1(y):
    """Marginal utility of utility function 1"""
    return np.where(y > 0, y ** (-0.5), np.inf)

def utility2(y):
    """Utility function U(y) = 5y - 0.05y^2"""
    return 5 * y - 0.05 * y ** 2

def marginal_utility2(y):
    """Marginal utility of utility function 2"""
    return 5 - 0.1 * y

# Set up initial parameters
stock = 1000  # Total stock available
discount_rate = 0.05
delta = 1 / (1 + discount_rate)  # Discount factor

N = 501  # Number of states
nA = 501  # Number of actions
step_size = stock / (N - 1)  # Step size between states

# Define action space
action_space = np.linspace(0, stock, nA)

# Create state space evenly over the square root of the action space
state_space = np.linspace(0, np.sqrt(stock), N) ** 2

# Create matrices for state-action pairs
state_space_matrix = np.repeat(state_space[:, np.newaxis], nA, axis=1)  # N x nA
action_space_matrix = np.repeat(action_space[np.newaxis, :], N, axis=0)  # N x nA

# Determine feasible actions (can't extract more than available stock)
feasible_actions = action_space_matrix <= state_space_matrix

# Compute utility matrices for both utility functions
utility_matrix1 = np.full((N, nA), -1e10)
utility_matrix1[feasible_actions] = utility1(action_space_matrix[feasible_actions])

utility_matrix2 = np.full((N, nA), -1e10)
utility_matrix2[feasible_actions] = utility2(action_space_matrix[feasible_actions])

# Flatten utility matrices for use in Bellman equation
utility_vector1 = utility_matrix1.flatten()
utility_vector2 = utility_matrix2.flatten()

# Compute updated state matrix after extraction
update_state_matrix = state_space_matrix - action_space_matrix

# Build the sparse transition matrix
row_indices = []
col_indices = []
data_values = []

for state_idx in range(N):
    for action_idx in range(nA):
        if feasible_actions[state_idx, action_idx]:
            current_stock = state_space[state_idx]
            extraction_amount = action_space[action_idx]
            next_stock = current_stock - extraction_amount
            row_index = state_idx * nA + action_idx

            # Find the indices of the next state using linear interpolation
            if next_stock <= state_space[0]:
                next_state_idx = 0
                weight = 1.0
                row_indices.append(row_index)
                col_indices.append(next_state_idx)
                data_values.append(weight)
            elif next_stock >= state_space[-1]:
                next_state_idx = N - 1
                weight = 1.0
                row_indices.append(row_index)
                col_indices.append(next_state_idx)
                data_values.append(weight)
            else:
                lower_idx = np.searchsorted(state_space, next_stock, side='right') - 1
                upper_idx = lower_idx + 1
                lower_state = state_space[lower_idx]
                upper_state = state_space[upper_idx]
                weight_upper = (next_stock - lower_state) / (upper_state - lower_state)
                weight_lower = 1.0 - weight_upper

                row_indices.extend([row_index, row_index])
                col_indices.extend([lower_idx, upper_idx])
                data_values.extend([weight_lower, weight_upper])

transition_matrix = csr_matrix(
    (data_values, (row_indices, col_indices)), shape=(N * nA, N)
)

# Define the Bellman operator function
def bellman_operator(value_function, utility_vector, transition_matrix, delta, N, nA):
    """Computes the Bellman operator for value function iteration."""
    expected_value = transition_matrix.dot(value_function)
    bellman_values = utility_vector + delta * expected_value
    return bellman_values.reshape(N, nA)

# Define the value function iteration algorithm
def value_function_iteration(
    utility_vector, transition_matrix, delta, N, nA, tol=1e-8, max_iter=1000
):
    """Performs value function iteration to find optimal policy."""
    v = np.zeros(N)
    for iteration in range(max_iter):
        B_sa = bellman_operator(v, utility_vector, transition_matrix, delta, N, nA)
        v_new = np.max(B_sa, axis=1)
        policy = np.argmax(B_sa, axis=1)
        if np.max(np.abs(v_new - v)) < tol:
            print(f"Converged in {iteration + 1} iterations")
            return v_new, policy
        v = v_new
    print("Did not converge within max iterations")
    return v, policy

# Define a function to construct the optimal transition matrix
def construct_optimal_transition_matrix(N, nA, transition_matrix, policy):
    """Constructs the optimal transition matrix based on the policy."""
    row_indices = []
    col_indices = []
    data_values = []

    for state_idx in range(N):
        optimal_action_idx = policy[state_idx]
        row_index = state_idx * nA + optimal_action_idx
        row = transition_matrix.getrow(row_index)
        cols = row.indices
        probs = row.data
        for col, prob in zip(cols, probs):
            if prob > 0:
                row_indices.append(state_idx)
                col_indices.append(col)
                data_values.append(prob)

    return csr_matrix((data_values, (row_indices, col_indices)), shape=(N, N))

# Perform value function iteration for utility function 1
print("Solving for utility function 1:")
v_u1, C_u1 = value_function_iteration(
    utility_vector1, transition_matrix, delta, N, nA
)
Topt_u1 = construct_optimal_transition_matrix(N, nA, transition_matrix, C_u1)

# Perform value function iteration for utility function 2
print("Solving for utility function 2:")
v_u2, C_u2 = value_function_iteration(
    utility_vector2, transition_matrix, delta, N, nA
)
Topt_u2 = construct_optimal_transition_matrix(N, nA, transition_matrix, C_u2)

# Define the simulation function
def simulate_extraction(
    T, starting_stock, marginal_utility_func, policy, action_space, state_space
):
    """Simulates the extraction and price over T periods."""
    remaining_stock = starting_stock
    extraction_path = []
    stock_path = []
    price_path = []

    for t in range(T):
        stock_path.append(remaining_stock)
        current_state = np.searchsorted(state_space, remaining_stock, side="right") - 1
        if current_state < 0 or current_state >= N:
            print("Error: State index out of bounds")
            break
        action_index = policy[current_state]
        action = action_space[action_index]
        extraction_path.append(action)
        if action == 0:
            print(f"Action is zero at time {t}, remaining stock {remaining_stock}")
        price = marginal_utility_func(action)
        price_path.append(price)
        remaining_stock -= action
        if remaining_stock < 0:
            print(f"Stock depleted at period {t + 1}")
            break

    return extraction_path, stock_path, price_path

# Define the plotting function using matplotlib
def plot_simulation_results(extraction_path, stock_path, price_path, title_suffix):
    """Plots the extraction, stock, and price over time."""
    periods = np.arange(1, len(extraction_path) + 1)

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    axs[0].plot(periods, extraction_path, label='Extraction', color='blue')
    axs[0].set_title(f'Extraction Over Time {title_suffix}')
    axs[0].set_xlabel('Period')
    axs[0].set_ylabel('Extraction')
    axs[0].legend()

    axs[1].plot(periods, stock_path[:len(periods)], label='Stock', color='green')
    axs[1].set_title(f'Stock Over Time {title_suffix}')
    axs[1].set_xlabel('Period')
    axs[1].set_ylabel('Stock')
    axs[1].legend()

    axs[2].plot(periods, price_path, label='Price', color='red')
    axs[2].set_title(f'Price Over Time {title_suffix}')
    axs[2].set_xlabel('Period')
    axs[2].set_ylabel('Price')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

# Simulation parameters
T = 80

# Simulate and plot results for utility function 1
extraction_path1, stock_path1, price_path1 = simulate_extraction(
    T, stock, marginal_utility1, C_u1, action_space, state_space
)
plot_simulation_results(extraction_path1, stock_path1, price_path1, '(Utility Function 1)')

# Simulate and plot results for utility function 2
extraction_path2, stock_path2, price_path2 = simulate_extraction(
    T, stock, marginal_utility2, C_u2, action_space, state_space
)
plot_simulation_results(extraction_path2, stock_path2, price_path2, '(Utility Function 2)')
