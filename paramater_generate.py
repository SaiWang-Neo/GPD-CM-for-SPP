import numpy as np


np.random.seed(0)  # For reproducibility

# 计算 a 的闭式解
def compute_a(A, v, w, B):
    e_Bv = np.exp(B @ v)
    a = 0.5 * np.linalg.inv(A.T) @ (2 * A.T @ A @ v + B.T @ (np.diag(e_Bv)) @ w )
    return a


# 定义函数，计算 w 的闭式解
def compute_w(B, v, b, C, c):
    e_Bv = np.exp(B @ v)
    CtC_inv = np.linalg.inv(C.T @ C)
    w = 0.5 * CtC_inv @ (e_Bv - b + 2 * C.T @ c)
    return w

# 生成满足要求的正数解
def generate_positive_solutions(num_solutions, n):
    solutions = []
    w = np.abs(np.random.randn(n)) + 0.2  # 确保生成的 w 元素为正数且不为零
    for _ in range(num_solutions):
        # w = np.abs(np.random.randn(n)) +0.2 # 确保生成的 w 元素为正数且不为零
        solutions.append(w)
    return solutions

# 依据解生成对应的系数
def generate_coefficients_from_solution(w,v):
    n = len(w)
    m = 10  # 设置 v 的尺寸
    C = np.random.randn(n, n)
    # v = np.random.randn(m)
    B = np.random.randn(n, m)
    e_Bv = np.exp(B @ v)
    b = np.random.randn(m)
    c = 0.5 * np.linalg.inv(C.T) @ (-e_Bv + b + 2 * C.T @ C @ w )
    A = np.random.randn(n, n)
    a = compute_a(A, v, w, B)
    return B, v, b, C, c, A, a

# 保存解及其对应的系数到文件
def save_solutions_and_coefficients(solutions, filename):
    with open(filename, 'w') as f:
        m = 10
        v = np.random.randn(m)
        for w in solutions:
            B, v, b, C, c, A, a = generate_coefficients_from_solution(w,v)
            np.savetxt(f, w.reshape(1, -1), header='w', comments='')
            np.savetxt(f, B, header='B', comments='')
            np.savetxt(f, v.reshape(1, -1), header='v', comments='')
            np.savetxt(f, b.reshape(1, -1), header='b', comments='')
            np.savetxt(f, C, header='C', comments='')
            np.savetxt(f, c.reshape(1, -1), header='c', comments='')
            np.savetxt(f, A, header='A', comments='')
            np.savetxt(f, a.reshape(1, -1), header='a', comments='')
            f.write('\n')  # 每组解之间空一行

# 生成100组解
num_solutions = 100
n = 10  # 设置解的尺寸
solutions = generate_positive_solutions(num_solutions, n)

# 保存到文件
save_solutions_and_coefficients(solutions, 'parameter.txt')

if solutions:
    print(f"成功生成并保存了 {len(solutions)} 组解及其对应的系数到 'solutions.txt' 文件。")
else:
    print("未能生成任何解。")







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
# print('sdfsdfs',solutions)
# 验证解及其对应的系数
def verify_solutions_and_coefficients(solutions):
    for i, (w, B, v, b, C, c, A, a) in enumerate(solutions):
        computed_w = compute_w(B, v, b, C, c)
        if not np.allclose(w, computed_w, rtol=1e-2, atol=1e-5):
            print(f"解 {i} 验证失败：计算得到的 w 和原始 w 不匹配。")
            print("原始 w：", w)
            print("计算得到的 w：", computed_w)
            return False
        e_Bv = np.exp(B @ v)
        lhs = 2 * A.T @ (A @ v - a)
        rhs = - B.T @ (np.diag(e_Bv)) @ w
        if not np.allclose(lhs, rhs):
            print(f"系数 {i} 验证失败：计算得到的 A @ v - a 和 -0.5 * w * e_Bv 不匹配。")
            print("lhs：", lhs)
            print("rhs：", rhs)
            return False
    print("所有解及其对应的系数验证成功。")
    return True






