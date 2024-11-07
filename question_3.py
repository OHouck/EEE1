from math import exp
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

## (a)
def y1(l, t):
    lt = l * exp(0.05 * t)
    return 1 / (lt ** 2)

def y2(l, t):
    lt = l * exp(0.05 * t)
    return 50 - 10 * lt

## (b)
def cum1(l, k):
    result, _ = integrate.quad(lambda t: y1(l, t), 0, k)
    return result

def cum2(l, k):
    result, _ = integrate.quad(lambda t: y2(l, t), 0, k)
    return result

## (c)
def cum_l1(l):
    for k in range(1001):  # Loop from k = 0 to k = 1000
        cumulative_value = cum1(l, k)
        
        # Check if cumulative_value has reached or exceeded 1000
        if cumulative_value >= 1000:
            return cumulative_value, k
    
    # If we reach k = 1000 without hitting the threshold, return the last value
    return cumulative_value, 1000

def cum_l2(l):
    for k in range(1001):  # Loop from k = 0 to k = 1000
        cumulative_value = cum2(l, k)
        
        # Check if cumulative_value has reached or exceeded 1000
        if cumulative_value >= 1000:
            return cumulative_value, k
    
    # If we reach k = 1000 without hitting the threshold, return the last value
    return cumulative_value, 1000

# get a rough idea of the value of l
l_values = np.linspace(0.5, 0.01, 100)

# find the value of l that gives the closest value to 1000
closest_l1 = min(l_values, key=lambda l: abs(cum_l1(l)[0] - 1000))
closest_l2 = min(l_values, key=lambda l: abs(cum_l2(l)[0] - 1000))

print(f"The value of l that gives the closest value to 1000 for cum_l1 is: {closest_l1}")
print(f"The value of l that gives the closest value to 1000 for cum_l2 is: {closest_l2}")

# plot the time path of y1(closest_l1,t) and y2(closest_l2,t) for t in [0,100]
t_values = np.linspace(0, 100, 500)
y1_values = []
y2_values = []

for t in t_values:
    y1_val = y1(closest_l1, t)
    y2_val = y2(closest_l2, t)
    
    if y1_val < 0:
        break
    y1_values.append(y1_val)
    
    if y2_val < 0:
        break
    y2_values.append(y2_val)

plt.figure(figsize=(10, 5))
plt.plot(t_values[:len(y1_values)], y1_values, label=f'y1(l={closest_l1:.2f}, t)')
plt.plot(t_values[:len(y2_values)], y2_values, label=f'y2(l={closest_l2:.2f}, t)')
plt.xlabel('Time (t)')
plt.ylabel('y(t)')
plt.title('Time Path of y1 and y2')
plt.legend()
plt.grid(True)
plt.show()