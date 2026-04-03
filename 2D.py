import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import InterpolatedUnivariateSpline
 
pi = np.pi
u_exact = lambda x, y: np.sin(pi * x) * np.sin(pi * y)
f       = lambda x, y: 2 * pi ** 2 * np.sin(pi * x) * np.sin(pi * y)
g       = lambda x, y: u_exact(x, y)  
 

def discretizare_domeniu(n, x_vals, y_vals):
    X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")
    return X.flatten(), Y.flatten()
 
 
def discretizare_ecuatii(n, h):
    N = (n + 1) ** 2
    A = np.zeros((N, N))
    for i in range(n + 1):
        for j in range(n + 1):
            idx = i * (n + 1) + j
            if i == 0 or i == n or j == 0 or j == n:  
                A[idx, idx] = 1.0
            else:                                    
                A[idx, idx] = -4.0
                A[idx, idx + 1]           = 1.0      
                A[idx, idx - (n + 1)]     = 1.0      
                A[idx, idx + (n + 1)]     = 1.0      
    return A / (h ** 2)
 
def plot_heatmap(U_grid, x_vals, y_vals, title):
    plt.figure(figsize=(6, 5))
    extent = [x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]]
    plt.imshow(U_grid.T, origin="lower", extent=extent, cmap="hot", aspect="equal")
    plt.colorbar(label="u_h(x, y)")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
 
 
def plot_error_spline(err_grid, x_vals, y_vals, title):
    mid_index = len(y_vals) // 2
    x_mid = x_vals
    err_line = err_grid[:, mid_index]
 
    # Construct spline
    spline = InterpolatedUnivariateSpline(x_mid, err_line, k=3)
    x_fine = np.linspace(x_vals[0], x_vals[-1], 400)
    err_spline = spline(x_fine)
 
    plt.figure(figsize=(6, 4))
    plt.scatter(x_mid, err_line, color="blue", label="noduri")
    plt.plot(x_fine, err_spline, color="red", label="spline")
    plt.title(title)
    plt.xlabel("x, pe y = 0.5")
    plt.ylabel("|eroare|")
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()

n = 50                   
h = 1.0 / n
x_vals = np.linspace(0.0, 1.0, n + 1)
y_vals = np.linspace(0.0, 1.0, n + 1)
 

X_flat, Y_flat = discretizare_domeniu(n, x_vals, y_vals)
A = discretizare_ecuatii(n, h)
 

N = (n + 1) ** 2
B = np.zeros(N)
for i in range(n + 1):
    for j in range(n + 1):
        idx = i * (n + 1) + j
        x, y = x_vals[i], y_vals[j]
        if i == 0 or i == n or j == 0 or j == n:        
            B[idx] = g(x, y)
        else:
            B[idx] = f(x, y)
 

lu, piv = lu_factor(A)
U = lu_solve((lu, piv), B)
 

U_grid = U.reshape((n + 1, n + 1))
 

U_exact_grid = u_exact(*np.meshgrid(x_vals, y_vals, indexing="ij"))
err_grid = np.abs(U_grid - U_exact_grid)
 

plot_heatmap(U_grid, x_vals, y_vals, "Temperatură numerică (heat‑map)")
plot_error_spline(err_grid, x_vals, y_vals, "Eroare absolută pe y = 0.5 (spline)")
plt.show()