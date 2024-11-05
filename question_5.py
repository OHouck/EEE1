import numpy as np
from scipy.stats import norm
import pandas as pd

## (a)
# Generate a vector with values of p from 0 to 80 with a step size of 1
p_values = np.arange(0, 81, 1)
print(f"Length of p_values vector: {len(p_values)}")

# Define a function that applies the formula to each value in the vector
def compute_profit(p_vector):
    return p_vector * 100000 - 3000000

# Compute the result
result = compute_profit(p_values)

# Display the result
print(result)


## (b)

# Parameters
states = np.arange(81)  # State values from 0 to 80
num_states = len(states)
mean_u = 0              # Mean of the transition distribution
sd_u = 4                # Standard deviation of the transition distribution

# Transition matrix initialization
T = np.zeros((num_states, num_states))

# Calculate cutoffs for each transition
cutoffs = np.arange(-0.5, num_states, 1)

# Populate the transition matrix
for i in range(num_states):  # Iterate over each row (current state p_t)
    for j in range(num_states):  # Iterate over each column (next state p_t+1)
        if j == 0:
            # For the first state, consider lower bound as -infinity
            lower_bound = -np.inf
        else:
            lower_bound = cutoffs[j]
        
        if j == num_states - 1:
            # For the last state, consider upper bound as infinity
            upper_bound = np.inf
        else:
            upper_bound = cutoffs[j + 1]

        # Calculate the probability of transitioning from state i to state j
        probability = norm.cdf(upper_bound, loc=states[i], scale=sd_u) - norm.cdf(lower_bound, loc=states[i], scale=sd_u)
        T[i, j] = probability

# Verify rows sum to 1 (as each row represents a probability distribution)
row_sums = T.sum(axis=1)

print("Transition Matrix (T):")
print(T)
print("\nRow sums (should be 1):")
print(row_sums)

## (c)

# Parameters
num_states = 81  # States from 0 to 80
std_dev = 4  # Standard deviation of the increment u
discount_factor = 1 / 1.05  # Discount factor
states = np.arange(num_states)
utility = states * 100000 - 3000000  # Utility term (p_t * 100000 - 3000000) for each state p_t

# Fixed-point iteration parameters
tolerance = 1e-6
max_iterations = 1000

# Initialize value function (starting with zero for all states)
V = np.zeros(num_states)
decision = np.zeros(num_states, dtype=int)  # To store the decision for each state (0 or 1)

# Fixed-point iteration to solve for V(p_t)
for iteration in range(max_iterations):
    # Create a copy of the current value function to check convergence later
    V_old = V.copy()
    
    # Calculate the option value for each state
    option_value = discount_factor * np.dot(T, V_old)
    
    # Determine V and decision based on max between utility and option value
    V = np.maximum(utility, option_value)
    decision = (V == utility).astype(int)  # 1 if choosing utility, 0 if choosing option value
    
    # Check for convergence
    if np.max(np.abs(V - V_old)) < tolerance:
        break

# Convert the resulting V and decision to a DataFrame for display
df_results = pd.DataFrame({'Value Function V(p_t)': V, 'Decision (0=option, 1=utility)': decision})

# Find the row number where the decision is 1
trigger_price_row = df_results[df_results['Decision (0=option, 1=utility)'] == 1].index[0]
print(f"The trigger price is: {trigger_price_row}")

## (d)
import matplotlib.pyplot as plt

# Plot the Value Function
plt.figure(figsize=(10, 6))
plt.plot(df_results.index, df_results['Value Function V(p_t)'], marker='o', linestyle='-')
plt.xlabel('Price')
plt.ylabel('Value Function V(p_t)')
plt.title('Value Function vs Price')
plt.grid(True)
plt.show()
