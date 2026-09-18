import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})

def solve_ode(x0, xn, n, u_fn, f_fn):
    x = np.linspace(x0, xn, n+1)
    h = x[1] - x[0]

    row, col, vals = [], [], []
    b = np.zeros(n+1)
    row.append(0)
    col.append(0)
    vals.append(1.0)
    b[0] = u_fn(x[0])


    for i in range(1, n):
        row.append(i)
        col.append(i-1)
        vals.append(1.0/h**2 - np.sin(x[i])/(2*h))
        row.append(i)
        col.append(i)
        vals.append(1. - 2.0/h**2)
        row.append(i)
        col.append(i+1)
        vals.append(1.0/h**2 + np.sin(x[i])/(2*h))
        b[i] = f_fn(x[i])

    row.append(n)
    col.append(n)
    vals.append(1.)
    b[-1] = u_fn(x[n])

    A = sp.coo_matrix((vals, (row, col)), shape=(n + 1, n + 1)).tocsc()
    u_numerical = sp.linalg.spsolve(A, b)
    return u_numerical

x0, xn = 0.0, 5.0
u_fn = lambda x: np.sin(x)
f_fn = lambda x: np.sin(x) * np.cos(x)

nvals = np.array([10, 20, 50, 70, 100])
errInf = np.zeros(len(nvals))
errL2 = np.zeros(len(nvals))

for k, N in enumerate(nvals):
    u_num = solve_ode(x0, xn, N, u_fn, f_fn)
    x = np.linspace(x0, xn, N+1)
    u_exact = u_fn(x)
    errInf[k] = np.linalg.norm(u_exact - u_num, np.inf) / np.linalg.norm(u_exact, np.inf)
    errL2[k] = np.linalg.norm(u_exact - u_num) / np.linalg.norm(u_exact)

plt.figure()
plt.plot(x, u_exact, 'b', linewidth=4, label='Exact')
plt.plot(x, u_num, 'r--', linewidth=4, color='orange', label='Finite Difference')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title("Solution of $u'' + sin(x)u' + u = f(x)$")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.loglog(nvals, errInf, 'o-', linewidth=4, label=r'$L_\infty$ Error Slope $\approx$ %0.2f' % np.polyfit(np.log(nvals), np.log(errInf), 1)[0])
plt.loglog(nvals, errL2, 's-', linewidth=4, label=r'$L_2$ Error Slope $\approx$ %0.2f' % np.polyfit(np.log(nvals), np.log(errL2), 1)[0])
plt.loglog(nvals, errInf[0] * (nvals / nvals[0]) ** (-2.0), 'k--', linewidth=4, label='Slope $O(N^{-2}$)')
plt.grid(True, which='both')
plt.xlabel('N')
plt.ylabel('Relative Error')
plt.legend(loc='lower left')
plt.title('Convergence of Finite Difference ODE Solution')
plt.show()