import numpy as np

np.random.seed(0)  # For reproducibility

# Compute the closed-form solution for a
def compute_a(A, v, w, B):
    e_Bv = np.exp(B @ v)
    a = 0.5 * np.linalg.inv(A.T) @ (2 * A.T @ A @ v + B.T @ (np.diag(e_Bv)) @ w )
    return a

# Define a function to compute the closed-form solution for w
def compute_w(B, v, b, C, c):
    e_Bv = np.exp(B @ v)
    CtC_inv = np.linalg.inv(C.T @ C)
    w = 0.5 * CtC_inv @ (e_Bv - b + 2 * C.T @ c)
    return w

# Generate positive solutions that meet the requirements
def generate_positive_solutions(num_solutions, n):
    solutions = []
    w = np.abs(np.random.randn(n)) + 0.2  # Ensure the generated w elements are positive and non-zero
    for _ in range(num_solutions):
        # w = np.abs(np.random.randn(n)) +0.2 # Ensure the generated w elements are positive and non-zero
        solutions.append(w)
    return solutions

# Generate corresponding coefficients based on the solution
def generate_coefficients_from_solution(w, v):
    n = len(w)
    m = 10  # Set the size of v
    C = np.random.randn(n, n)
    # v = np.random.randn(m)
    B = np.random.randn(n, m)
    e_Bv = np.exp(B @ v)
    b = np.random.randn(m)
    c = 0.5 * np.linalg.inv(C.T) @ (-e_Bv + b + 2 * C.T @ C @ w )
    A = np.random.randn(n, n)
    a = compute_a(A, v, w, B)
    return B, v, b, C, c, A, a

# Save solutions and their corresponding coefficients to a file
def save_solutions_and_coefficients(solutions, filename):
    with open(filename, 'w') as f:
        m = 10
        v = np.random.randn(m)
        for w in solutions:
            B, v, b, C, c, A, a = generate_coefficients_from_solution(w, v)
            np.savetxt(f, w.reshape(1, -1), header='w', comments='')
            np.savetxt(f, B, header='B', comments='')
            np.savetxt(f, v.reshape(1, -1), header='v', comments='')
            np.savetxt(f, b.reshape(1, -1), header='b', comments='')
            np.savetxt(f, C, header='C', comments='')
            np.savetxt(f, c.reshape(1, -1), header='c', comments='')
            np.savetxt(f, A, header='A', comments='')
            np.savetxt(f, a.reshape(1, -1), header='a', comments='')
            f.write('\n')  # Leave a blank line between each set of solutions

# Generate 100 sets of solutions
num_solutions = 100
n = 10  # Set the size of the solutions
solutions = generate_positive_solutions(num_solutions, n)

# Save to file
save_solutions_and_coefficients(solutions, 'parameter.txt')

if solutions:
    print(f"Successfully generated and saved {len(solutions)} sets of solutions and their corresponding coefficients to 'parameter.txt'.")
else:
    print("Failed to generate any solutions.")

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

# Verify the solutions and their corresponding coefficients
def verify_solutions_and_coefficients(solutions):
    for i, (w, B, v, b, C, c, A, a) in enumerate(solutions):
        computed_w = compute_w(B, v, b, C, c)
        if not np.allclose(w, computed_w, rtol=1e-2, atol=1e-5):
            print(f"Solution {i} verification failed: Computed w does not match original w.")
            print("Original w:", w)
            print("Computed w:", computed_w)
            return False
        e_Bv = np.exp(B @ v)
        lhs = 2 * A.T @ (A @ v - a)
        rhs = - B.T @ (np.diag(e_Bv)) @ w
        if not np.allclose(lhs, rhs):
            print(f"Coefficients {i} verification failed: Computed A @ v - a does not match -0.5 * w * e_Bv.")
            print("lhs:", lhs)
            print("rhs:", rhs)
            return False
    print("All solutions and their corresponding coefficients verified successfully.")
    return True
