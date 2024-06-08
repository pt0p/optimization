import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)

class LASSOOptOracle(BaseSmoothOracle):
    def __init__(self, A, b, regcoef, t):
        self.A = A.copy()
        self.b = b.copy()
        self.regcoef = regcoef
        self.t = t


    def func(self, x):
        n = x.shape[0] // 2
        x, u = x[:n], x[n:]
        f = self.t * (1/2 * scipy.linalg.norm(self.A.dot(x) - self.b) ** 2 + \
             self.regcoef * np.ones(n).dot(u)) - \
             np.sum(np.log(u + x) + np.log(u - x))
        return f


    def grad(self, x):
        n = x.shape[0] // 2
        x, u = x[:n], x[n:]
        a = self.A.T.dot(self.A.dot(x))
        b = self.A.T.dot(self.b)

        x_part = self.t * (a - b) - \
                (1/(u+x) - 1/(u-x))
        u_part = self.t * self.regcoef - (1/(u+x) + 1/(u-x))
        return np.hstack([x_part, u_part])


    def hess(self, x):
        n = x.shape[0] // 2
        x, u = x[:n], x[n:]
        # I = scipy.sparse.diags(np.ones(n))
        common = scipy.sparse.diags(1/(u+x)**2 + 1/(u-x)**2)
        xx = self.t * self.A.T.dot(self.A) + common
        xu = scipy.sparse.diags(1/(u+x)**2 - 1/(u-x)**2)
        return scipy.sparse.vstack([scipy.sparse.hstack([xx, xu]),
                          scipy.sparse.hstack([xu, common])])

    def hess_components(self, x):
        n = x.shape[0] // 2
        x, u = x[:n], x[n:]
        c = scipy.sparse.diags(1/(u+x)**2 + 1/(u-x)**2)
        a = self.t * self.A.T.dot(self.A) + c
        b = scipy.sparse.diags(1/(u+x)**2 - 1/(u-x)**2)
        return a, b, c
        

def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for 
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    # TODO: implement.
    mu = np.min([1, regcoef / scipy.linalg.norm(ATAx_b(x), ord=np.inf)]) * \
    Ax_b(x)
    gap = 1/2 * scipy.linalg.norm(Ax_b(x)) ** 2 + regcoef * \
    scipy.linalg.norm(x, ord=1) + 1/2 * scipy.linalg.norm(mu) ** 2 + \
    np.dot(b, mu)
    return gap


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the gradient
    e = np.diag(np.ones_like(x))
    grad = np.zeros_like(x)
    fx = func(x)
    for i in range(len(x)):
        grad[i] = (func(x + eps * e[i, :]) - fx) / eps
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the Hessian
    e = np.diag(np.ones_like(x))
    grad = grad_finite_diff
    hess = np.zeros((x.shape[0], ) * 2)
    for i in range(len(x)):
        hess[i, :] = (grad(func, x + eps * e[i, :], eps=eps) - grad(func, x, eps=eps) )/ eps
    return hess


