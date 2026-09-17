import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def solve_ode(x0, xn, n, f_fn, u_exact_fn):
    x = np.linspace(x0, xn, n + 1)
    h = x[1] - x[0]

    u_exact = u_exact_fn(x)
    f_vals = f_fn(x)

    main = np.zeros(n + 1)
    lower = np.zeros(n + 1)
    upper = np.zeros(n + 1)
    b = np.zeros(n + 1)

    i = np.arange(1, n)
    lower[i] = 1.0 / h**2 - np.sin(x[i]) / (2 * h)
    main[i]  = 1.0 - 2.0 / h**2
    upper[i] = 1.0 / h**2 + np.sin(x[i]) / (2 * h)
    b[i] = f_vals[i]

    main[0] = 1.0
    b[0] = u_exact[0]
    main[-1] = 1.0
    b[-1] = u_exact[-1]

    A = sp.diags(
        [lower[1:], main, upper[:-1]],
        offsets=[-1, 0, 1],
        format='csr'
    )

    u_numerical = spla.spsolve(A, b)

    return x, u_numerical, u_exact


def rel_errors(exact, numerical):
    err_inf = np.linalg.norm(exact - numerical, np.inf) / np.linalg.norm(exact, np.inf)
    err_l2 = np.linalg.norm(exact - numerical) / np.linalg.norm(exact)
    return err_inf, err_l2


x0, xn = 0.0, 5.0
u_exact_fn = lambda x: np.sin(x)
f_fn = lambda x: np.sin(x) * np.cos(x)

nvals = np.array([10, 20, 50, 70, 100])
errInf = np.zeros(len(nvals))
errL2 = np.zeros(len(nvals))

for k, N in enumerate(nvals):
    x, u_num, u_exact = solve_ode(x0, xn, N, f_fn, u_exact_fn)
    errInf[k], errL2[k] = rel_errors(u_exact, u_num)

plt.figure()
plt.plot(x, u_exact, 'b', linewidth=3, label='Exact')
plt.plot(x, u_num, 'r--', linewidth=3, label='Finite Difference', color='orange')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title("Solution of $u'' + sin(x)u' + u = f(x)$")
plt.legend()
plt.grid(True)

plt.figure()
plt.loglog(nvals, errInf, 'o-', linewidth=2, label=r'$L_\infty$ Error')
plt.loglog(nvals, errL2, 's-', linewidth=2, label=r'$L_2$ Error')
plt.loglog(nvals, errInf[0] * (nvals / nvals[0]) ** (-2.0), 'k--', linewidth=1.5, label='Slope -2')
plt.grid(True, which='both')
plt.xlabel('N')
plt.ylabel('Relative Error')
plt.legend(loc='lower left')
plt.title('Convergence of ODE')

plt.show()