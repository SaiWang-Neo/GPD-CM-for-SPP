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

# Regularization parameters
mu = 1
sigma = 0.95




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
    v = np.ones(n) * 1.0
    sequence_error = []
    ####################
    Lrs = np.linalg.norm(grad_v_phi(v))
    # Calculate L
    print("Lrs:", Lrs)
    ########################

    for k in range(10000):
        wold = w
        vold = v
        s_k = Lrs
        
        w_hat = np.linalg.inv(2 * C.T @ C + s_k * np.eye(m)) @ (
                np.exp(B @ vold) - b + 2 * C.T @ c + s_k * w
        )
        w = np.maximum(w_hat, 0)
        w_tilde = 2 * w - wold

        gradold = grad_v_phi(vold)
        r_k = Lrs
        A_T = A.T
        A_T_A = 2 * A_T @ A
        r_I_n = r_k * np.eye(n)
        B_T = B.T
        v = np.linalg.inv(A_T_A + r_I_n) @ (-B_T @ np.diag(np.exp(B @ vold)) @ w_tilde + 2 * A_T @ a + r_k * vold)


        if k % 10000 == 0:
            print('Iteration:', k)
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



with open('average_sequence_GY.txt', 'w') as f:
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

