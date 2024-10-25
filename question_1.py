import numpy as np
from scipy.optimize import csr_matrix


def utility1(y: int) -> float:
    return 2 * np.sqrt(y)

def utility2(y: int) -> float:
    return 2 * y ** (1/2)

def main():

    # Set up starting values
    stock = 1000

    discount_rate = 0.05
    delta = 1 / (1 + discount_rate)

    N = 501 # number of states
    nA = 501 # number of actions
    step_size = stock / (N - 1)

    # state space and action space are 0 to 1000 by 2
    state_space = np.linspace(0, stock, N)
    action_space = np.linspace(0, stock, nA)

    # both are N x nA
    state_space_matrix = np.tile(state_space.reshape(N, 1), (1, nA)) 
    action_space_matrix = np.tile(action_space.reshape(1, nA), (N, 1))

    feasible_actions = action_space_matrix <= state_space_matrix

    utility_matrix1 = np.where(feasible_actions, 
                               utility1(action_space_matrix), -np.inf)    
    utility_matrix2 = np.where(feasible_actions,
                                 utility2(action_space_matrix), -np.inf)

    # get next state_indices
    update_state_matrix = state_space_matrix - action_space_matrix
    next_state_indices = np.zeros((N, nA), dtype=int)

    next_state_indices[feasible_actions] = (update_state_matrix[feasible_actions] / step_size).astype(int)

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
    transition_matrix = csr_matrix((data, (row_indices, col_indices)), 
                                   shape=(N, nA * N))


    def value_function_iteration(utility_matrix, delta, value_function, transition_matrix,
                                 tolerance, max_iterations):

        iteration = 0
        while iteration < max_iterations:
            iteration += 1  
            previous_V = value_function.copy()

            # Compute V_next for each state-action pair
            V_prev_repeated = np.tile(previous_V, nA)  # Shape: (nA * N,)
            Q_flat = utility_matrix.flatten() + delta * V_prev_repeated  # Shape: (nA * N,)

            # Multiply transition matrix with V_prev_repeated
            V_next_flat = transition_matrix.dot(V_prev_repeated)  # Shape: (N,)
            Q_flat = utility_matrix.flatten() + delta * V_next_flat

            # Reshape Q_flat to (N, nA)
            Q = Q_flat.reshape(N, nA)

            # Maximization step
            value_function_new = np.max(Q, axis=1)
            optimal_action = np.argmax(Q, axis=1)

            diff = np.max(np.abs(value_function_new - previous_V))

            value_function = value_function_new.copy()

            if diff < tolerance:
                print(f'Converged in {iteration} iterations')
                break
        else:
            print('Did not converge within max iterations')
        return value_function, optimal_action


if __name__ == "__main__":
    main()