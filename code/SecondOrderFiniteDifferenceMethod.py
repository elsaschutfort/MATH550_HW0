import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

plt.rcParams.update({'font.size': 18})

def second_order_finite_difference(x0, xn, n, f, fp_exact_fn):
    x = np.linspace(x0, xn, n + 1)
    h = x[1] - x[0]

    f_vals = f(x)
    fp_exact = fp_exact_fn(x)

    fp_numerical = np.zeros(n + 1)
    fp_numerical[0] = (f_vals[1] - f_vals[0]) / h
    fp_numerical[-1] = (f_vals[-1] - f_vals[-2]) / h
    fp_numerical[1:-1] = (f_vals[2:] - f_vals[:-2]) / (2 * h)

    return x, fp_numerical, fp_exact


def second_order_finite_difference_dxx(x0, xn, n, f, f_exact_fn):
    x = np.linspace(x0, xn, n)
    h = x[1] - x[0]

    row, col, val = [], [], []

    row.append(0)
    col.append(0)
    val.append(-2.)
    row.append(0)
    col.append(1)
    val.append(1.)
    row.append(0)
    col.append(n-2)
    val.append(1.)
    
    for i in range(1, n-1):
        row.append(i)
        col.append(i-1)
        val.append(1.)
        
        row.append(i)
        col.append(i)
        val.append(-2.)
        
        row.append(i)
        col.append(i+1)
        val.append(1.)

    row.append(n-1)
    col.append(n-1)
    val.append(-2.)
    row.append(n-1)
    col.append(n-2)
    val.append(1.)
    row.append(n-1)
    col.append(1)
    val.append(1.)

    D2 = sp.coo_matrix((val, (row, col)), shape=(n, n)).tocsc()

    f_vals = f(x) / h**2
    f_numerical = D2.dot(f_vals)
    f_exact = f_exact_fn(x)

    return x, f_numerical, f_exact


def rel_errors(exact, numerical):
    err_inf = np.linalg.norm(exact - numerical, np.inf) / np.linalg.norm(exact, np.inf)
    err_l2 = np.linalg.norm(exact - numerical) / np.linalg.norm(exact)
    return err_inf, err_l2


nvals = np.array([10, 20, 50, 70, 100])
x0, xn = 0.0, 2 * np.pi

f = lambda x: np.exp(np.sin(x))
fp_exact_fn = lambda x: np.exp(np.sin(x)) * np.cos(x)

errInf = np.zeros(len(nvals))
errL2 = np.zeros(len(nvals))

for k, N in enumerate(nvals):
    x, fp_numerical, fp_exact = second_order_finite_difference(x0, xn, N, f, fp_exact_fn)
    errInf[k], errL2[k] = rel_errors(fp_exact, fp_numerical)

plt.figure()
plt.plot(x, fp_exact, 'b', linewidth=4, label='Exact')
plt.plot(x, fp_numerical, 'r--', linewidth=4, label='Finite Difference', color='orange')
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('Derivative of $f(x) = e^{sin(x)}$')
plt.legend()
plt.grid(True)

plt.figure()
plt.loglog(nvals, errInf, 'o-', linewidth=4, label=r'$L_\infty$ Error Slope $\approx$ %0.2f' % np.polyfit(np.log(nvals), np.log(errInf), 1)[0])
plt.loglog(nvals, errL2, 's-', linewidth=4, label=r'$L_2$ Error Slope $\approx$ %0.2f' % np.polyfit(np.log(nvals), np.log(errL2), 1)[0])
plt.loglog(nvals, errInf[0] * (nvals / nvals[0]) ** (-1.0), 'k--', linewidth=4, label='Slope $O(N^{-1}$)')
plt.loglog(nvals, errL2[0] * (nvals / nvals[0]) ** (-1.5), 'r--', linewidth=4, label='Slope $O(N^{-1.5})$')
plt.grid(True, which='both')
plt.xlabel('N')
plt.ylabel('Relative Error')
plt.legend(loc='lower left')
plt.title('Convergence of Finite Difference Derivative')



nvals2 = np.array([10, 20, 50, 70, 100])
x0, xn = 0.0, 2 * np.pi

f = lambda x: np.exp(np.sin(x))
fpp_exact_fn = lambda x: np.exp(np.sin(x)) * (np.cos(x) ** 2 - np.sin(x))

errInf2 = np.zeros(len(nvals2))
errL22 = np.zeros(len(nvals2))

for k, N in enumerate(nvals2):
    x2, fpp_numerical, fpp_exact = second_order_finite_difference_dxx(x0, xn, N, f, fpp_exact_fn)
    errInf2[k], errL22[k] = rel_errors(fpp_exact, fpp_numerical)


plt.figure()
plt.plot(x2, fpp_exact, 'b', linewidth=4, label='Exact')
plt.plot(x2, fpp_numerical, 'r--', linewidth=4, label='Finite Difference', color='orange')
plt.xlabel('x')
plt.ylabel("f''(x)")
plt.title('Second Derivative of $f(x) = e^{sin(x)}$')
plt.legend()
plt.grid(True)

plt.figure()
plt.loglog(nvals2, errInf2, 'o-', linewidth=4, label=r'$L_\infty$ Error Slope $\approx$ %0.2f' % np.polyfit(np.log(nvals2), np.log(errInf2), 1)[0])
plt.loglog(nvals2, errL22, 's-', linewidth=4, label=r'$L_2$ Error Slope $\approx$ %0.2f' % np.polyfit(np.log(nvals2), np.log(errL22), 1)[0])
plt.loglog(nvals2, errInf2[0] * (nvals2 / nvals2[0]) ** (-2.0), 'k--', linewidth=4, label='Slope $O(N^{-2}$)')
plt.grid(True, which='both')
plt.xlabel('N')
plt.ylabel('Relative Error')
plt.legend(loc='lower left')
plt.title('Convergence of Finite Difference Derivative')


plt.show()