import numpy as np
import matplotlib.pyplot as plt

def factorizareLU(A):
    n = A.shape[0]
    L = np.eye(n, dtype=float)
    U = np.zeros_like(A, dtype=float)
    A_copy = A.astype(float).copy()

    for k in range(n):
        U[k, k] = A_copy[k, k]
        for i in range(k + 1, n):
            L[i, k] = A_copy[i, k] / A_copy[k, k]
            U[k, i] = A_copy[k, i]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A_copy[i, j] -= (A_copy[i, k] * A_copy[k, j]) / A_copy[k, k]
    return L, U


def Subs_Asc(L, b):
    n = L.shape[0]
    x = np.zeros((n, 1))
    for i in range(n):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
    return x


def Subs_Desc(U, b):
    n = U.shape[0]
    x = np.zeros((n, 1))
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def polinomLagrange(X, k, n, x):
    L_val = 1.0
    for i in range(n + 1):
        if i != k:
            L_val *= (x - X[i]) / (X[k] - X[i])
    return L_val


def interpolareLagrange(X, Y, x):
    y = 0.0
    for i in range(len(X)):
        y += Y[i] * polinomLagrange(X, i, len(X) - 1, x)
    return y


def k_func(x):
    return x + 1.0


def u_exact(x):
    return x**4 - 2*x**3 + x


def f_func(x):
    return -16*x**3 + 6*x**2 + 12*x - 1


def build_matrix_and_rhs(N, L, kf, ff, A0, B0):
    h = L / (N + 1)
    xi = np.linspace(0, L, N + 2)
    fi = ff(xi[1:-1])
    kmid = kf(0.5 * (xi[:-1] + xi[1:]))

    main_diag = (kmid[:-1] + kmid[1:]) / h**2
    off_diag = -kmid[1:-1] / h**2

    A = np.diag(main_diag) + np.diag(off_diag, k=-1) + np.diag(off_diag, k=1)

    F = fi.reshape(-1, 1)
    F[0]  -= off_diag[0]   * A0   
    F[-1] -= off_diag[-1]  * B0   
    return xi, A, F


def solve_heat_1d(N, L=1.0, A0=0.0, B0=0.0):
    x, A, F = build_matrix_and_rhs(N, L, k_func, f_func, A0, B0)
    Lm, Um = factorizareLU(A)
    y = Subs_Asc(Lm, F)
    u_inner = Subs_Desc(Um, y).flatten()

    u = np.zeros(N + 2)
    u[0], u[-1] = A0, B0
    u[1:-1] = u_inner
    return x, u, u_exact(x)


if __name__ == "__main__":
    Ns = [10, 20, 40, 80, 160]
    errors, hs = [], []

    for N in Ns:
        x, u_num, u_ex = solve_heat_1d(N)
        errors.append(np.max(np.abs(u_num - u_ex)))
        hs.append(1.0 / (N + 1))

    plt.figure(figsize=(6, 4))
    plt.plot(x, u_ex, label="u exact")
    plt.plot(x, u_num, marker="", linestyle="--", label=f"u numeric (N={Ns[-1]})")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Comparație soluții – noul test")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.loglog(hs, errors, marker="o")
    plt.xlabel("h")
    plt.ylabel("eroare max")
    plt.title("Descăderea erorii (așteptat O(h²))")
    plt.grid(True, which="both")
    plt.show()

    p = np.polyfit(np.log(hs), np.log(errors), 1)[0]
    print(f"Rată estimată de convergență: {abs(p):.2f}")