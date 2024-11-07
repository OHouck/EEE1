import numpy as np
from scipy.sparse import csr_matrix
import altair as alt
import pandas as pd

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

# action space is the same as before
action_space = np.linspace(0, stock, nA)

# create statespace even over the squre root of the action space
state_space = np.linspace(0, stock ** 0.5, N) ** 2

# constant on the columns, state space on the rows
state_space_matrix = np.tile(state_space.reshape(N, 1), (1, nA)) # N x nA

# constant on the rows, action space on the columns
action_space_matrix = np.tile(action_space.reshape(1, nA), (N, 1)) # N x nA
feasible_actions = action_space_matrix <= state_space_matrix

utility_matrix1 = np.where(feasible_actions, 
                            utility1(action_space_matrix), -np.inf)    
utility_matrix2 = np.where(feasible_actions,
                                utility2(action_space_matrix), -np.inf)

utility_matrix1_flat = utility_matrix1.flatten() # N * nA 1D array to use in Bellman with trnasition matrix
utility_matrix2_flat = utility_matrix2.flatten()


update_state_matrix = state_space_matrix - action_space_matrix

# part (b) create new state transition matrix using interpolation
# create  N x N * nA matrix transition matrix
row_indices = []
col_indices = []
data = []
feasible_action_count = []
for i in range(N): # state index
    current_state = state_space[i]
    for j in range(nA): # action index
        if feasible_actions[i , j]:
            action = action_space[j]
            current_state = state_space_matrix[i, j]
            next_state = update_state_matrix[i, j]

            # if next_state in state_space proceed as normal
            if next_state in state_space:
                next_state_index = np.where(state_space == next_state)[0][0] # state after action
                state_action_index = j * N + next_state_index
                col_indices.append(state_action_index)  # Column index for state-action pair
                row_indices.append(i)
                data.append(1) # deterministic transition
            else:
                next_state_index_low = np.where(state_space < next_state)[0][-1] 
                next_state_index_high = np.where(state_space > next_state)[0][0] 
                state_action_index_low = j * N + next_state_index_low
                state_action_index_high = j * N + next_state_index_high

                weight_low = (next_state - state_space[next_state_index_low]) / (state_space[next_state_index_high] - state_space[next_state_index_low]) 
                weight_high = (state_space[next_state_index_high] - next_state) / (state_space[next_state_index_high] - state_space[next_state_index_low]) 
                col_indices.append(state_action_index_low)
                data.append(weight_low)
                row_indices.append(i)

                col_indices.append(state_action_index_high)
                data.append(weight_high)
                row_indices.append(i)
            
transition_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(N, nA * N)) 

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

def get_optimal_transition_matrix(N, nA, transition_matrix, C):
    """
    Constructs the optimal transition matrix Topt using the N x N * nA transition matrix.

    Parameters:
    - N: Number of states
    - nA: Number of actions
    - transition_matrix: N x N * nA transition matrix (sparse matrix)
    - C: Optimal action indices for each state (array of size N)

    Returns:
    - Topt: Optimal transition matrix (N x N sparse matrix)
    """
    row_indices = []
    col_indices = []
    data = []

    for i in range(N):
        optimal_action = C[i]
        for j in range(N):
            transition_prob = transition_matrix[i, optimal_action * N + j]
            if transition_prob > 0:
                row_indices.append(i)
                col_indices.append(j)
                data.append(transition_prob)

    Topt = csr_matrix((data, (row_indices, col_indices)), shape=(N, N))
    return Topt

# Solve for utility function 1
print("Solving for utility function 1:")
v_u1, C_u1 = value_function_iteration(utility_matrix1_flat, transition_matrix, delta, N, nA, tolerance, max_iterations)

# Part (h) Find optimal transition matrix
Topt_u1 = get_optimal_transition_matrix(N, nA, transition_matrix, C_u1) 

print(Topt_u1.shape)

# Solve for utility function 2
print("Solving for utility function 2:")
v_u2, C_u2 = value_function_iteration(utility_matrix2_flat, transition_matrix, delta, N, nA, tolerance, max_iterations)

# Part (h) Find optimal transition matrix
Topt_u2 = get_optimal_transition_matrix(N, nA, transition_matrix, C_u2)

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


chart_1 | chart_2
