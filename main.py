
import numpy as np
import math
from scipy.optimize import minimize, root

# Parameters
n, m = 10, 10
np.random.seed(0)  # For reproducibility

# Load solutions and their corresponding coefficients from a file
def load_solutions_and_coefficients(filename):
    with open(filename, 'r') as f:
        data = f.read().split('\n\n')  # Each solution group is separated by a blank line
    solutions = []
    for block in data:
        if block.strip():  # Ignore empty blocks
            lines = block.split('\n')
            w = np.loadtxt(lines[1:2])
            B = np.loadtxt(lines[3:13])
            v = np.loadtxt(lines[14:15])
            b = np.loadtxt(lines[16:17])
            C = np.loadtxt(lines[18:28])
            c = np.loadtxt(lines[29:30])
            A = np.loadtxt(lines[31:41])
            a = np.loadtxt(lines[42:43])
            solutions.append((w, B, v, b, C, c, A, a))
    return solutions

# Regularization parameters
mu = 1
sigma = 0.95
alpha = 0.5

def L(v, w):
    return np.linalg.norm(A @ v - a) ** 2 + w @ (np.exp(B @ v) - b) - np.linalg.norm(C @ w - c) ** 2

def L_variable(v, w):
    return np.linalg.norm(A @ v - a) ** 2 + w @ (np.exp(B @ v) - b)

def grad_v_L(v, w, v_old, r_k):
    return 2 * A.T @ (A @ v - a) + B.T @ (np.diag(np.exp(B @ v))) @ w + r_k * (v - v_old)

def grad_v_Ltg(v, w):
    return 2 * A.T @ (A @ v - a) + B.T @ (np.diag(np.exp(B @ v))) @ w

def grad_v_phi(v):
    return B.T @ np.diag(np.exp(B @ v))

def grad_w_L(v, w):
    return np.exp(B @ v) - b - 2 * C.T @ (C @ w - c)

# Main algorithm
all_sequence = []
# Load solutions and their corresponding coefficients
loaded_solutions = load_solutions_and_coefficients('parameter.txt')
for i, (wopt, B, vopt, b, C, c, A, a) in enumerate(loaded_solutions):
    w = np.ones(m)
    # data = np.linalg.inv(A.T @ A) @ A.T @ a
    # v = (data - data.min()) / (data.max() - data.min())
    v = np.ones(n) * 2.0
    sequence_error = []
    for k in range(10):
        # Prediction step
        r_k = np.linalg.norm(grad_v_phi(v)) / mu
        # print('r_k:', r_k)
        # print(v)

        def grad_v_L_root(v_):
            return grad_v_L(v_, w, v, r_k)

        res_v = root(grad_v_L_root, v, tol=1e-8)
        v_tilde = res_v.x
        # print("CCCAAAAAAOOOO", grad_v_L_root(v_tilde))
        # print("v_tilde:", np.sum(grad_v_L_root(v_tilde)))

        s_k = (alpha + (1 - alpha) ** 2) * np.linalg.norm(grad_v_phi(v_tilde)) ** 2 / (sigma * r_k)
        # print('s_k:', s_k)
        w_hat = np.linalg.inv(2 * C.T @ C + s_k * np.eye(m)) @ (
                alpha * np.diag(np.exp(B @ v_tilde)) @ B @ (v_tilde - v) +
                np.exp(B @ v) - b + 2 * C.T @ c + s_k * w
        )
        w_tilde = np.maximum(w_hat, 0)

        # Correction step
        eta_k = np.hstack((v, w))
        eta_tilde = np.hstack((v_tilde, w_tilde))
        M_k_top_right = (alpha - 1) / r_k * B.T @ np.diag(np.exp(B @ v_tilde))
        M_k = np.block([
            [np.eye(n), M_k_top_right],
            [np.zeros((m, n)), np.eye(m)]
        ])
        eta_k1 = eta_k - 1 * M_k @ (eta_k - eta_tilde)

        v = eta_k1[:n]
        w = eta_k1[n:]
        if k % 10000 == 0:
            print('Iteration:', k, 'Sample', i)
        sequence = 0.5 * (np.linalg.norm(v - vopt) + np.linalg.norm(w - wopt))
        sequence_error.append(sequence)
    # print('finish at the random round:', i, 'finish at tau', k, f'sequence error: {sequence}')
    if not np.isnan(sum(sequence_error)):
        all_sequence.append(sequence_error)
    if i == 100:
        break
# print('index', all_sequence[1][1], all_sequence)
# error = 0.5 * (np.linalg.norm(v - v_tilde) + np.linalg.norm(w - w_tilde))
log_sequences = [[math.log10(seq) for seq in sublist] for sublist in all_sequence]
# print('log_sequences', log_sequences)
# First, calculate the sum of all sublist elements
sub_sum = np.array([sum(values) for values in zip(*log_sequences)])

# Calculate the number of sublists
number_of_sublists = len(log_sequences)

# Calculate the average, i.e., the total sum divided by the number of sublists
# average = 10 ** (sub_sum / number_of_sublists) # for ele in sub_sum / number_of_sublists
average = [10 ** (ele / number_of_sublists) for ele in sub_sum]

print("sub sum:", sub_sum)
print("Number of sublists:", number_of_sublists)
print("Average:", average)

with open('average_sequence_test.txt', 'w') as f:
    f.write(str(average))

import math

# Display results
# print(f'Optimized v: {v}')
# print(f'Optimized w: {w}')
# print(f'Number of iterations: {k}')

# print(f'Optimized abs(v-vopt): {np.linalg.norm(v - vopt)}')
# print(f'Optimized abs(w-wopt): {np.linalg.norm(w - wopt)}')

# Compare with optimizer
# res = minimize(lambda x: -L(x[:n], x[n:]), np.concatenate([v0, w0]))
# v_opt, w_opt = res.x[:n], res.x[n:]
# print(f'Optimizer v: {v_opt}')
# print(f'Optimizer w: {w_opt}')

# Verify if it is a saddle point
def verify_saddle_point(v, w, epsilon=1e-4):
    # Check if v is a local minimum
    v_perturbed = v + epsilon * np.random.randn(n)
    print(L(v_perturbed, w) - L(v, w))
    if L(v_perturbed, w) < L(v, w):
        return False

    # Check if w is a local maximum
    w_perturbed = w + epsilon * np.random.randn(m)
    print(L(v, w_perturbed) - L(v, w))
    if L(v, w_perturbed) > L(v, w):
        return False

    return True

is_saddle_point = verify_saddle_point(v, w)
print("Is saddle point:", is_saddle_point)

# print(grad_w_L(vopt, wopt), grad_v_Ltg(vopt, wopt))
print('gradient', np.sum(np.abs(grad_w_L(v, w))), np.sum(np.abs(grad_v_Ltg(v, w))))
