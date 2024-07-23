# # import numpy as np
# # from scipy.optimize import minimize
# #
# # # Problem parameters
# # n, m = 30, 30
# # np.random.seed(0)
# # A = np.random.rand(n, n)
# # B = np.random.rand(m, n)
# # C = np.random.rand(m, m)
# # a = np.random.rand(n)
# # b = np.random.rand(m)
# # c = np.random.rand(m)
# #
# # # Algorithm parameters
# # mu = 5
# # sigma = 0.95
# # alpha = 0.5
# # tau = 1e-10
# #
# # # Initial values
# # v0 = np.linalg.inv(A.T @ A) @ A.T @ a
# # w0 = np.ones(m)
# #
# #
# # # Helper function to compute the gradient of Phi
# # def grad_Phi(v):
# #     return 2 * A.T @ (A @ v - a)
# #
# #
# # # Helper function to compute L(v, w)
# # def L(v, w):
# #     term1 = np.linalg.norm(A @ v - a) ** 2
# #     term2 = w @ (np.exp(B @ v) - b)
# #     term3 = -np.linalg.norm(C @ w - c) ** 2
# #     return term1 + term2 + term3
# #
# #
# # # Helper function to compute the gradient of L w.r.t. v
# # def grad_L_v(v, w):
# #     term1 = 2 * A.T @ (A @ v - a)
# #     term2 = B.T @ (w * np.exp(B @ v))
# #     return term1 + term2
# #
# #
# # # Helper function to compute the gradient of L w.r.t. w
# # def grad_L_w(v, w):
# #     term1 = np.exp(B @ v) - b
# #     term2 = -2 * C.T @ (C @ w - c)
# #     return term1 + term2
# #
# #
# # # Algorithm implementation
# # v = v0
# # w = w0
# # k = 0
# # err = float('inf')
# #
# # while err >= tau:
# #     # Prediction step
# #     r_k = np.linalg.norm(grad_Phi(v)) / mu
# #     res_v = minimize(lambda v_: L(v_, w) + (r_k / 2) * np.linalg.norm(v_ - v) ** 2, v, method='L-BFGS-B')
# #     v_tilde = res_v.x
# #
# #     s_k = (alpha + (1 - alpha) ** 2) * np.linalg.norm(grad_Phi(v_tilde)) ** 2 / (sigma * r_k)
# #     res_w = minimize(lambda w_: -(
# #                 L(v_tilde, w_) + alpha * w_ @ grad_Phi(v_tilde) @ (v_tilde - v) - (s_k / 2) * np.linalg.norm(
# #             w_ - w) ** 2), w, method='L-BFGS-B', bounds=[(0, None)] * m)
# #     w_tilde = res_w.x
# #
# #     # Correction step
# #     eta_k = np.concatenate([v, w])
# #     eta_tilde = np.concatenate([v_tilde, w_tilde])
# #     M_k = np.identity(n + m)
# #     eta_k_plus_1 = eta_k - M_k @ (eta_k - eta_tilde)
# #     v, w = eta_k_plus_1[:n], eta_k_plus_1[n:]
# #
# #     # Compute error
# #     err = 0.5 * (np.linalg.norm(v - v0) + np.linalg.norm(w - w0))
# #     k += 1
# #
# # print(f'Algorithm solution: v={v}, w={w}, iterations={k}')
# #
# #
# # # Compare with optimizer
# # def obj(x):
# #     v, w = x[:n], x[n:]
# #     return L(v, w)
# #
# #
# # x0 = np.concatenate([v0, w0])
# # res = minimize(obj, x0, method='L-BFGS-B', bounds=[(None, None)] * n + [(0, None)] * m)
# # v_opt, w_opt = res.x[:n], res.x[n:]
# #
# # print(f'Optimizer solution: v={v_opt}, w={w_opt}')
#
#
# import numpy as np
# from scipy.linalg import inv
# from scipy.optimize import minimize
#
# # Set the random seed for reproducibility
# np.random.seed(42)
#
# # Dimensions
# n, m = 30, 30
#
# # Randomly generate matrices and vectors
# A = np.random.rand(n, n)
# B = np.random.rand(m, n)
# C = np.random.rand(m, m)
# a = np.random.rand(n)
# b = np.random.rand(m)
# c = np.random.rand(m)
#
# # Regularization parameters
# mu = 5
# sigma = 0.95
# alpha = 0.5
# tau = 1e-20
#
# # Initial values
# v0 = np.linalg.inv(A.T @ A) @ (A.T @ a)
# w0 = np.ones(m)
#
# # Objective function
# def L(v, w):
#     return np.linalg.norm(A @ v - a)**2 + np.dot(w, np.exp(B @ v) - b) - np.linalg.norm(C @ w - c)**2
#
# # Gradient of the primal function
# def grad_phi(v):
#     return 2 * A.T @ (A @ v - a)
#
# # Gradient of the objective function
# def grad_L_v(v, w):
#     return 2 * A.T @ (A @ v - a) + B.T @ (w * np.exp(B @ v))
#
# def grad_L_w(v):
#     return np.exp(B @ v) - b - 2 * C.T @ (C @ w0 - c)
#
# # Prediction step
# def prediction_step(v_k, w_k):
#     r_k = np.linalg.norm(grad_phi(v_k)) / mu
#     L_v = lambda v: L(v, w_k) + r_k / 2 * np.linalg.norm(v - v_k)**2
#     res = minimize(L_v, v_k)
#     v_tilde = res.x
#     s_k = [alpha + (1 - alpha)**2] / (sigma * r_k) * np.linalg.norm(grad_phi(v_tilde))**2
#     w_tilde = inv(2 * C.T @ C + s_k * np.eye(m)) @ (
#         alpha * np.diag(np.exp(B @ v_tilde)) @ B @ (v_tilde - v_k) +
#         np.exp(B @ v_tilde) - b + 2 * C.T @ c + s_k * w_k
#     )
#     return v_tilde, w_tilde
#
# # Correction step
# def correction_step(eta_k, eta_tilde, M_k):
#     return eta_k - M_k @ (eta_k - eta_tilde)
#
# # Initialization
# v_k, w_k = v0, w0
# k = 0
# err = np.inf
#
# # Iteration
# while err >= tau:
#     v_k_old, w_k_old = v_k, w_k
#     v_tilde, w_tilde = prediction_step(v_k, w_k)
#     M_k = np.eye(n + m)  # Simplified for demonstration; modify as needed
#     eta_k = np.concatenate([v_k, w_k])
#     eta_tilde = np.concatenate([v_tilde, w_tilde])
#     eta_k = correction_step(eta_k, eta_tilde, M_k)
#     v_k, w_k = eta_k[:n], eta_k[n:]
#     err = 0.5 * (np.linalg.norm(v_k - v_k_old) + np.linalg.norm(w_k - w_k_old))
#     k += 1
#     print(f'Iteration {k}, Error: {err}')
#
# # Final result
# v_star, w_star = v_k, w_k
# print(f'Optimal v: {v_star}')
# print(f'Optimal w: {w_star}')
#
# # Compare with optimizer
# res = minimize(lambda x: -L(x[:n], x[n:]), np.concatenate([v0, w0]))
# v_opt, w_opt = res.x[:n], res.x[n:]
# print(f'Optimizer v: {v_opt}')
# print(f'Optimizer w: {w_opt}')
import numpy as np
import math
from scipy.optimize import minimize, root

# Parameters
n, m = 10, 10
np.random.seed(0)  # For reproducibility

# 从文件中加载解及其对应的系数
def load_solutions_and_coefficients(filename):
    with open(filename, 'r') as f:
        data = f.read().split('\n\n')  # 每组解之间有一个空行
    solutions = []
    for block in data:
        if block.strip():  # 忽略空块
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


# # Generate random matrices and vectors
# A = np.random.rand(n, n)
# a = np.random.rand(n)
# b = 0.2 * (np.random.rand(m) + 1)
# C = np.random.rand(m, m)
# u = np.random.rand(m)
# c = C @ np.abs(np.linalg.inv(C.T @ C) @ (C.T @ u * 12) + 12 * u)
# B = 0.2 * np.random.rand(m, n)

# print(a)

# Initialize variables
# v0 = np.linalg.inv(A.T @ A) @ A.T @ a
# w0 = np.ones(m)
# v0 = np.ones(n)
# Regularization parameters
mu = 1
sigma = 0.95
alpha = 0



def L(v, w):
    return np.linalg.norm(A @ v - a) ** 2 + w @ (np.exp(B @ v) - b) - np.linalg.norm(C @ w - c) ** 2



def L_variable(v, w):
    return np.linalg.norm(A @ v - a) ** 2 + w @ (np.exp(B @ v) - b)

def grad_v_L(v, w, v_old, r_k):
    return 2 * A.T @ (A @ v - a) + B.T @ (np.diag(np.exp(B @ v))) @ w + r_k * (v-v_old)


def grad_v_Ltg(v, w):
    return 2 * A.T @ (A @ v - a) + B.T @ (np.diag(np.exp(B @ v))) @ w

def grad_v_phi(v):
    return B.T @ np.diag(np.exp(B @ v))


def grad_w_L(v, w):
    return np.exp(B @ v) - b - 2 * C.T @ (C @ w - c)


# Main algorithm
all_sequence = []
# 加载解及其对应的系数
loaded_solutions = load_solutions_and_coefficients('parameter.txt')
for i, (wopt, B, vopt, b, C, c, A, a) in enumerate(loaded_solutions):
    w = np.ones(m)
    # data = np.linalg.inv(A.T @ A) @ A.T @ a
    # v = (data - data.min()) / (data.max() - data.min())
    v = np.ones(n) * 2.0
    sequence_error = []
    ####################
    Lrs = np.linalg.norm(grad_v_phi(v))
    # Calculate L
    print("Lrs:", Lrs)
    ########################

    for k in range(10000):
        # Prediction step
        r_k = Lrs
        # print('r_k:', r_k)
        # print(v)


        def grad_v_L_root(v_):
            return grad_v_L(v_, w, v, r_k)

        res_v = root(grad_v_L_root, v, tol=1e-8)
        v_tilde = res_v.x
        # print("CCCAAAAAAOOOO", grad_v_L_root(v_tilde))
        # print("v_tilde:", np.sum(grad_v_L_root(v_tilde)))

        s_k = Lrs
        # print('s_k:', s_k)
        w_hat = np.linalg.inv(2 * C.T @ C + s_k * np.eye(m)) @ (
                alpha * np.diag(np.exp(B @ v_tilde)) @ B @ (v_tilde - v) +
                np.exp(B @ v) - b + 2 * C.T @ c + s_k * w
        )
        w_tilde = np.maximum(w_hat, 0)

        # # Correction step
        # eta_k = np.hstack((v, w))
        # eta_tilde = np.hstack((v_tilde, w_tilde))
        # M_k_top_right = (alpha - 1) / r_k * B.T @ np.diag(np.exp(B @ v_tilde))
        # M_k = np.block([
        #     [np.eye(n), M_k_top_right],
        #     [np.zeros((m, n)), np.eye(m)]
        # ])
        # eta_k1 = eta_k - 1 * M_k @ (eta_k - eta_tilde)

        v = v_tilde
        w = w_tilde
        if k % 10000 == 0:
            print('Iteration:', k, 'Sample',i)
        sequence = 0.5 * (np.linalg.norm(v - vopt) + np.linalg.norm(w - wopt))
        sequence_error.append(sequence)
    # print('finish at the random round:', i, 'finish at tau', k, f'sequence error: {sequence}')
    if not np.isnan(sum(sequence_error)):
        all_sequence.append(sequence_error)
    if i == 100:
        break
# print('index',all_sequence[1][1],all_sequence)
        # error = 0.5 * (np.linalg.norm(v - v_tilde) + np.linalg.norm(w - w_tilde))
log_sequences = [[math.log10(seq) for seq in sublist] for sublist in all_sequence ]
# print('log_sequences',log_sequences)
# 先计算所有子列表的元素和
sub_sum = np.array([sum(values) for values in zip(*log_sequences)])

# 计算子列表的个数
number_of_sublists = len(log_sequences)

# 计算平均值，即总和除以子列表的个数
# average = 10 ** (sub_sum / number_of_sublists) # for ele in sub_sum / number_of_sublists
average = [10 ** (ele/ number_of_sublists) for ele in sub_sum ]

print("sub sum:", sub_sum)
print("Number of sublists:", number_of_sublists)
print("Average:", average)



with open('average_sequence_AH.txt', 'w') as f:
    f.write(str(average))


import math



#
# # Display results
# print(f'Optimized v: {v}')
# print(f'Optimized w: {w}')
# print(f'Number of iterations: {k}')
#
# print(f'Optimized abs(v-vopt): {np.linalg.norm(v - vopt)}')
# print(f'Optimized abs(w-wopt): {np.linalg.norm(w - wopt)}')



# # Compare with optimizer
# res = minimize(lambda x: -L(x[:n], x[n:]), np.concatenate([v0, w0]))
# v_opt, w_opt = res.x[:n], res.x[n:]
# print(f'Optimizer v: {v_opt}')
# print(f'Optimizer w: {w_opt}')


# 验证是否为鞍点
def verify_saddle_point(v, w, epsilon=1e-4):
    # 检查v是否是局部最小值
    v_perturbed = v + epsilon * np.random.randn(n)
    print(L(v_perturbed, w) - L(v, w))
    if L(v_perturbed, w) < L(v, w):
        return False

    # 检查w是否是局部最大值
    w_perturbed = w + epsilon * np.random.randn(m)
    print(L(v, w_perturbed) - L(v, w))
    if L(v, w_perturbed) > L(v, w):
        return False

    return True


is_saddle_point = verify_saddle_point(v, w)
print("Is saddle point:", is_saddle_point)


# print(grad_w_L(vopt, wopt), grad_v_Ltg(vopt, wopt))

print('gradient',np.sum(np.abs(grad_w_L(v, w))), np.sum(np.abs(grad_v_Ltg(v, w))))

