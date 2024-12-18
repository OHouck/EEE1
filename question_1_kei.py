import numpy as np
from scipy.sparse import csr_matrix
import altair as alt
import pandas as pd
import pdfkit

# part (a) (b) (c)
def utility1(y: int) -> float:
    return 2 * y ** (1/2)
def marginal_utility1(y: int) -> float:
    return y ** (-0.5)

def utility2(y: int) -> float:
    return 5 * y - 0.05 * y ** 2
def marginal_utility2(y: int) -> float:
    return 5 - 0.1 * y

# Set up starting values
stock = 1000

discount_rate = 0.5
delta = 1 / (1 + discount_rate)

N = 501 # number of states
nA = 501 # number of actions
step_size = stock / (N - 1)
# part a) and b)
# state space and action space are 0 to 1000 by 2
state_space = np.linspace(0, stock, N)
action_space = np.linspace(0, stock, nA)

state_space_matrix = np.tile(state_space.reshape(N, 1), (1, nA)) # N x nA
action_space_matrix = np.tile(action_space.reshape(1, nA), (N, 1)) # N x nA
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
for i in range(N): # state index
    for j in range(nA): # action index
        if feasible_actions[i, j]:
            next_state_index = next_state_indices[i, j] # state after action
            row_indices.append(i)
            state_action_index = j * N + next_state_index
            # state_action_index = j + N * next_state_index
            col_indices.append(state_action_index)  # Column index for state-action pair
            data.append(1) # deterministic transition

transition_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(N, nA * N)) 

# part (g)
def bellman(v, U_flat, transition_matrix, delta, N, nA):
    '''Evaluated RHS of bellman before max
    - v: current value function (N, )
    - U_flat: flattened utility matrix (N * nA, ) 
    - transition_matrix: (N, N * nA) sparse matrix
    - delta: dicsount factor
    - N: number of states
    - nA: number of actions
    ''' # XX come back and check this
    v_next = transition_matrix.T.dot(v)  # N * nA, 1
    B_flat = U_flat + delta * v_next # (N * nA, 1)
    B_sa = B_flat.reshape(N, nA)  # (N, nA) sa for state action
    print(f"v = {v[-1]}")
    print(f"v_next = {v_next[-1]}")
    # print max value in v_next
    print(f"v_next_max = {np.max(v_next)}")
    print(f"B_flat = {B_flat[-1]}")
    print(f"B_sa = {B_sa[-1, -1]}")
    print(f"B_flat = {B_flat[0]}")
    print(f"B_sa = {B_sa[0, 0]}")
    return B_sa

# Value function iteration function
max_iterations = 1000
np.random.seed(454)

value_function = np.random.randn(N)
utility_matrix = utility_matrix1
iteration = 0
tolerance = 1e-8
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
    - v: Value function (N x nA array)
    - policy: Optimal action indices for each state (N x 1 array)
    """
    v = value_function
    for iteration in range(max_iterations):
        print(f'Iteration {iteration + 1}')
        print(f"v = {v[400]}")
        B_sa = bellman(v, U_flat, transition_matrix, delta, N, nA)
        v_new = np.max(B_sa, axis=1) # find value of best action for each state
        print(f"V_new: : {v_new[400]}")
        policy = np.argmax(B_sa, axis=1) # find index of best action for each state
        print(f"Policy: {policy[500]}")
        diff = np.max(np.abs(v_new - v))
        print(f"diff = {diff}")
        v = v_new
        if diff < tolerance:
            print(f'Converged in {iteration + 1} iterations')
            break
    else:
        print('Did not converge within max iterations')
    return v, policy

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

# Solve for utility function 1
print("Solving for utility function 1:")
v_u1, C_u1 = value_function_iteration(utility_matrix1_flat, transition_matrix, delta, N, nA, tolerance, max_iterations)

# Part (h) Find optimal transition matrix
Topt_u1 = get_optimal_transition_matrix(N, C_u1, next_state_indices)

# Solve for utility function 2
print("Solving for utility function 2:")
v_u2, C_u2 = value_function_iteration(utility_matrix2_flat, transition_matrix, delta, N, nA, tolerance, max_iterations)

# Part (h) Find optimal transition matrix
Topt_u2 = get_optimal_transition_matrix(N, C_u2, next_state_indices)

# Part (i) Simulate the model for t= 80 periods
T = 80
remaining_stock = stock
extraction_path = []
stock_path = []
price_path = []

def simulation(T, starting_stock, price_function, C, action_space, step_size):
    remaining_stock = starting_stock
    extraction_path = []
    stock_path = []
    price_path = []
    for t in range(T):
        print(f"Period {t + 1}: Remaining stock = {remaining_stock}")
        stock_path.append(remaining_stock)
        current_state = int(remaining_stock / step_size)
        if current_state >= N:
            print("Error: State index out of bounds")
            break
        action_index = C[current_state]
        action = action_space[action_index]
        remaining_stock -= action
        if remaining_stock < 0:
            break
        extraction_path.append(action)
        price = price_function(action)
        price_path.append(price)
    return extraction_path, stock_path, price_path

def make_extraction_plot(extraction_path, stock_path, price_path):
    df = pd.DataFrame({'Period': np.arange(1, T + 1),
        'Extraction': extraction_path,
        "Price": price_path,
        "Stock": stock_path})

    # Create the first chart for Extraction
    extraction_chart = alt.Chart(df).mark_line(color='blue').encode(
        x='Period',
        y=alt.Y('Extraction', axis=alt.Axis(title='Extraction'))
    )
    price_chart = alt.Chart(df).mark_line(color='red').encode(
        x='Period',
        y=alt.Y('Price', axis=alt.Axis(title='Price'), scale=alt.Scale(zero=False))
    ).transform_calculate(
        y='datum.Price'
    )
    # Layer the two charts together
    combined_chart = alt.layer(
        extraction_chart,
        price_chart
    ).resolve_scale(
        y='independent'
    )
    return combined_chart

extraction_path, stock_path, price_path = simulation(T, stock, marginal_utility1, C_u1, action_space, step_size)
chart_1 = make_extraction_plot(extraction_path, stock_path, price_path)

extraction_path, stock_path, price_path = simulation(T, stock, marginal_utility2, C_u2, action_space, step_size)
chart_2 = make_extraction_plot(extraction_path, stock_path, price_path)
combined_chart = chart_1 | chart_2

chart_1.save('chart_1.html')
chart_2.save('chart_2.html')
