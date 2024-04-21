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


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A

    def minimize_directional(self, x, d):
        """
        Minimizes the function with respect to a specific direction:
            Finds alpha = argmin f(x + alpha d)
        """
        # TODO: Implement for bonus part
        a = (self.A.dot(x)).dot(d) - self.b.dot(d)
        b = (self.A.dot(d)).dot(d)
        if scipy.linalg.norm(b) == 0:
            return 0
        return max(0, - a / b)


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATy : function of y
            Computes matrix-vector product A^Ty, where y is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        # TODO: Implement
        Ax = self.matvec_Ax(x)
        m = Ax.shape[0]
        n = x.shape[0]
        I_m = np.ones(m)
        error = I_m.dot(np.logaddexp(0, - self.b * Ax)) / m
        reg = self.regcoef * x.dot(x.T) / 2
        return error + reg

    def grad(self, x):
        # TODO: Implement
        Ax = self.matvec_Ax(x)
        k = scipy.special.expit(-self.b * Ax)
        m = Ax.shape[0]
        grad_error = - self.matvec_ATx(self.b * k) / m
        grad_reg = self.regcoef * x
        return grad_error + grad_reg

    def hess(self, x):
        # TODO: Implement
        Ax = self.matvec_Ax(x)
        m = Ax.shape[0]
        n = x.shape[0]
        p = scipy.special.expit(self.b * Ax)
        k = scipy.special.expit(-self.b * Ax)
        hess_error = self.matmat_ATsA(p * k) / m
        hess_reg = self.regcoef * np.diag(np.ones(n))
        return hess_error + hess_reg


    def hess_vec(self, x, v):
        Ax = self.matvec_Ax(x)
        Av = self.matvec_Ax(v)
        m = Ax.shape[0]
        n = x.shape[0]
        p = scipy.special.expit(self.b * Ax)
        k = scipy.special.expit(-self.b * Ax)
        ATpkAv = self.matvec_ATx(p * k * Av)
        return ATpkAv / m + self.regcoef * v


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        return self.func(x + alpha * d)

    def grad_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        return self.grad(x + alpha * d).dot(d)


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: (A.T).dot(x)

    def matmat_ATsA(s):
        # TODO: Implement
        return (A.T).dot(scipy.sparse.diags(s).dot(A))

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def hess_vec_finite_diff(func, x, v, eps=1e-5):
    """
    Returns approximation of the matrix product 'Hessian times vector'
    using finite differences.
    """
    # TODO: Implement numerical estimation of the Hessian times vector
    e = np.diag(np.ones_like(x))
    hess = np.zeros(x.shape[0])
    for i in range(len(x)):
        a = func(x + eps * v + eps * e[i, :])
        b = func(x + eps * v)
        c = func(x + eps * e[i, :])
        d = func(x)
        hess[i] = (a - b - c + d) / eps ** 2
    return hess