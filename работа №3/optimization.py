from collections import defaultdict
import numpy as np
from numpy.linalg import norm, solve
from time import time
from datetime import datetime
from oracles import LASSOOptOracle
import scipy

class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        # TODO: Implement line search procedures for Armijo, Wolfe and Constant steps.
        phi = lambda alpha: oracle.func_directional(x_k, d_k, alpha)
        derphi = lambda alpha: oracle.grad_directional(x_k, d_k, alpha)
        if self._method == 'Constant':
            return self.c
        elif self._method == 'Wolfe':
            opt = scipy.optimize.linesearch.scalar_search_wolfe2(phi, derphi,
                                                                  c1=self.c1,
                                                                  c2=self.c2)
            if opt[0]:
                return opt[0]
        if self._method == 'Wolfe':
            alpha = 1
        elif previous_alpha:
            alpha = 2 * previous_alpha
        else:
            alpha = self.alpha_0
        while phi(alpha) >  phi(0) + self.c1 * alpha * derphi(0):
            alpha /= 2
        return alpha


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def max_alpha(x, u, d_x, d_u):
    idx1 = (-d_x - d_u > 0)
    alpha1 = (-x[idx1] - u[idx1]) / (d_x[idx1] + d_u[idx1])
    idx2 = (d_x - d_u > 0)
    alpha2 = - (x[idx2] - u[idx2]) / (d_x[idx2] - d_u[idx2])
    alpha = 1
    if np.sum(idx1) + np.sum(idx2) != 0:
        alpha = 0.99 * np.min(np.hstack([alpha1, alpha2]))
    alpha = min(alpha, 1)
    return alpha

def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    # TODO: Implement Newton's method.
    # Use line_search_tool.line_search() for adaptive step size.

    i = 0
    n = len(x_k) // 2
    grad = oracle.grad(x_k)
    grad_norm = scipy.linalg.norm(grad)
    criterion = np.sqrt(tolerance) * grad_norm
    previous_alpha = None
    message = 'success'
    if trace:
        start_time = datetime.now()
        history['time'].append((start_time - start_time).total_seconds())
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(grad_norm)
        if len(x_k) <= 2:
            history['x'].append(x_k)
    if display:
        print(f'Итерация: {i}, время: {(datetime.now() - start_time).total_seconds()}')
        print(f'Норма градиента: {grad_norm}')  
    while i < max_iter and grad_norm > criterion:
        grad = oracle.grad(x_k)
        if (not np.isfinite(grad).all()):
            message = 'computational_error'
            break
        grad_norm = scipy.linalg.norm(grad)
        hess = oracle.hess(x_k).toarray()
        L, low = scipy.linalg.cho_factor(hess)
        d_k = scipy.linalg.cho_solve((L, low), -grad)

        x, u = x_k[:n], x_k[n:]
        d_x, d_u = d_k[:n], d_k[n:]
        alpha_conditional = max_alpha(x, u, d_x, d_u)
        
        if alpha_conditional < 0:
            print('с условием что-то случилось')
            message = 'computational_error'
            break
        # d_k = np.hstack([d_x, d_u])
        alpha = line_search_tool.line_search(oracle, x_k, d_k,
                                            previous_alpha=alpha_conditional/2)
        x_k = x_k + alpha * d_k

        i += 1
        if trace:
            current_time = datetime.now()
            history['time'].append((current_time - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_norm)
            if len(x_k) <= 2:
                history['x'].append(x_k)
        if  display:
            print(f'Итерация: {i}, время: {(datetime.now() - start_time).total_seconds()}')
            print(f'Норма градиента: {grad_norm}')
    if i == max_iter and grad_norm > criterion:
        message = 'iterations_exceeded'
    return (x_k, message, history)


def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5, 
                         tolerance_inner=1e-8, max_iter=100, 
                         max_iter_inner=20, t_0=1, gamma=10, 
                         c1=1e-4, lasso_duality_gap=None,
                         trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is increased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for the outer loop stopping criterion:
        Stop the outer loop (which iterates over `k`) when
            `duality_gap(x_k) <= tolerance`
    tolerance_inner : float
        Epsilon value for the inner loop stopping criterion.
        Stop the inner loop (which iterates over `l`) when
            `|| \nabla phi_t(x_k^l) ||_2^2 <= tolerance_inner * \| \nabla \phi_t(x_k) \|_2^2 `
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    lasso_duality_gap : callable object or None.
        If calable the signature is lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)
        Returns duality gap value for esimating the progress of method.
    trace : bool
        If True, the progress information is appended into history dictionary 
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every **outer** iteration of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    # TODO: implement.

    history = defaultdict(list) if trace else None
    message = 'success'
    line_search_tool = get_line_search_tool({'method' : 'Wolfe', 'c1' : c1})
    x_k = np.copy(x_0)
    u_k = np.copy(u_0)
    t_k = t_0
    Ax_b = lambda x: A.dot(x) - b
    ATAx_b = lambda x: A.T.dot(A.dot(x)-b) 
    i = 0 
    start = datetime.now()
    duality_gap = lasso_duality_gap(x_k, Ax_b, ATAx_b, b, reg_coef)
    oracle = LASSOOptOracle(A, b, reg_coef, t_k)
    n = len(x_k)
    if trace:
        history['time'].append((datetime.now()-start).total_seconds())
        history['func'].append(oracle.func(np.hstack([x_k, u_k])))
        history['duality_gap'].append(duality_gap)
        if n <= 2:
            history['x'].append(x_k)
    while i < max_iter and duality_gap > tolerance:
        new_x, message_inner, _ = newton(oracle, np.hstack([x_k, u_k]), 
                                   tolerance=tolerance_inner,
                                   max_iter=max_iter_inner,
                                   line_search_options={'method' : 'Armijo', 'c1' : c1},
                                   trace=False, display=False)
        if message_inner == 'computational_error' or not np.isfinite(new_x).all():
            message = 'computational_error'
            break
        x_k, u_k = new_x[:n], new_x[n:]
        t_k *= gamma
        duality_gap = lasso_duality_gap(x_k, Ax_b, ATAx_b, b, reg_coef)
        oracle = LASSOOptOracle(A, b, reg_coef, t_k)
        if trace:
            history['time'].append((datetime.now()-start).total_seconds())
            history['func'].append(oracle.func(np.hstack([x_k, u_k])))
            history['duality_gap'].append(duality_gap)
            if n <= 2:
                history['x'].append(x_k)
        i+=1
    if i == max_iter and duality_gap > tolerance:
        message = 'iterations_exceeded'
    return (x_k, message, history)

























































