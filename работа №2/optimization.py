import numpy as np
import scipy
from datetime import datetime
from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
from utils import get_line_search_tool


def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    # TODO: Implement Conjugate Gradients method.
    criterion = tolerance * scipy.linalg.norm(b)
    message = 'success'
    max_iter = max_iter if max_iter else float('inf')
    i = 0
    start_time = datetime.now()
    
    g_k = matvec(x_k) - b
    norm_g_k = scipy.linalg.norm(g_k)
    d_k = - g_k

    if trace:
        history['time'].append((start_time - start_time).total_seconds())
        history['residual_norm'].append(norm_g_k)
        if len(x_k) <= 2:
            history['x'].append(x_k)
    
    A_d_k = matvec(d_k)
    while i < max_iter and norm_g_k > criterion:
        i += 1
        a = norm_g_k ** 2 / np.dot(A_d_k , d_k)
        x_k = x_k + a * d_k
        old_g_k = g_k.copy()
        norm_old_g_k = norm_g_k 
        g_k = g_k + a * A_d_k
        norm_g_k = scipy.linalg.norm(g_k)
        d_k = - g_k + norm_g_k ** 2 / norm_old_g_k ** 2 * d_k
        A_d_k = matvec(d_k)
        if trace:
            current_time = datetime.now()
            history['time'].append((current_time - start_time).total_seconds())
            history['residual_norm'].append(norm_g_k)
            if len(x_k) <= 2:
                history['x'].append(x_k)
    if i == max_iter and norm_g_k > criterion:
        message = 'iterations_exceeded'
    return (x_k, message, history)


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    # TODO: Implement L-BFGS method.
    # Use line_search_tool.line_search() for adaptive step size.
    i = 0
    grad = oracle.grad(x_k)
    grad_norm = scipy.linalg.norm(grad)
    criterion = tolerance * grad_norm ** 2
    message = 'success'
    start_time = datetime.now()
    if trace:
        history['time'].append((start_time - start_time).total_seconds())
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(grad_norm)
        if len(x_k) <= 2:
            history['x'].append(x_k)
    if display:
        print(f'Итерация: {i}, время: {(datetime.now() - start_time).total_seconds()}')
        print(f'Норма градиента: {grad_norm}')
    H = []
    while i < max_iter and grad_norm ** 2 > criterion:
        if len(H) == 0:
            d_k = - grad
        else:
            d_k = LBFGS_direction(H, -grad)
        alpha = line_search_tool.line_search(oracle, x_k, d_k,
                                            previous_alpha=0.5)
        if alpha == 0:
            break
        delta_x = alpha * d_k
        new_x_k = x_k + delta_x
        new_grad = oracle.grad(new_x_k)
        if (not np.isfinite(new_grad).all()):
            message = 'computational_error'
            break
        if len(H) <= memory_size:
            H.append((delta_x, new_grad - grad))
        else:
            H = H[1:]
            H.append(((delta_x, new_grad - grad)))
        grad = new_grad.copy()
        grad_norm = scipy.linalg.norm(grad)
        x_k = new_x_k.copy()
        i += 1
        if trace:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_norm)
            if len(x_k) <= 2:
                history['x'].append(x_k)
        if display and i % 5 == 0:
            print(f'Итерация: {i}, время: {(datetime.now() - start_time).total_seconds()}')
            print(f'Норма градиента: {grad_norm}')
    if i == max_iter and grad_norm ** 2 > criterion:
        message = 'iterations_exceeded'
    return (x_k, message, history)


def BFGS_Multiply(v, H, gamma):
    if len(H) == 0:
        return gamma * v
    s, y = H[-1]
    new_H = H[:-1]
    new_v = v - np.dot(s, v) * y / np.dot(y, s)
    z = BFGS_Multiply(new_v, new_H, gamma)
    return z + ((np.dot(s, v) - np.dot(y, z)) / np.dot(y, s)) * s


def LBFGS_direction(H, v):
    s, y = H[-1]
    gamma = np.dot(y, s) / np.dot(y, y)
    return BFGS_Multiply(v, H, gamma)





def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500, 
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    line_search_tool = get_line_search_tool(line_search_options)
    

    # TODO: Implement hessian-free Newton's method.
    # Use line_search_tool.line_search() for adaptive step size.
    i = 0
    grad = oracle.grad(x_k)
    grad_norm = scipy.linalg.norm(grad)
    criterion = tolerance * grad_norm ** 2
    previous_alpha = None
    message = 'success'
    start_time = datetime.now()
    if trace:
        start_time = datetime.now()
        history['time'].append((start_time - start_time).total_seconds())
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(grad_norm)
        if len(x_k) <= 2:
            history['x'].append(x_k)
    if display:
        print(f'Итерация: {i}, время:{(datetime.now() - start_time).total_seconds()}')
        print(f'Норма градиента: {grad_norm}')

    while i < max_iter and grad_norm ** 2 > criterion:
        grad = oracle.grad(x_k)
        if (not np.isfinite(grad).all()):
            message = 'computational_error'
        grad_norm = scipy.linalg.norm(grad)
        matvec = lambda x: oracle.hess_vec(x_k, x)
        eta = min(0.5, np.sqrt(grad_norm))
        d_k = - grad
        d_k = conjugate_gradients(matvec, -grad, d_k, tolerance=eta)[0]
        while np.dot(grad, d_k) > 0:
            eta /= 10
            d_k = conjugate_gradients(matvec, -grad, d_k, tolerance=eta)[0]
        alpha = line_search_tool.line_search(oracle, x_k, d_k,
                                            previous_alpha=0.5)
        x_k = x_k + alpha * d_k
        i += 1
        if trace:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_norm)
            if len(x_k) <= 2:
                history['x'].append(x_k)
        if display and i % 5 == 0:
            print(f'Итерация: {i}, время:{(datetime.now() - start_time).total_seconds()}')
            print(f'Норма градиента: {grad_norm}')
    if i == max_iter and grad_norm ** 2 > criterion:
        message = 'iterations_exceeded'
    return (x_k, message, history)


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
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
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    # TODO: Implement gradient descent
    # Use line_search_tool.line_search() for adaptive step size.
    i = 0
    previous_alpha = None
    message = 'success'
    grad = oracle.grad(x_k)
    grad_norm = scipy.linalg.norm(grad)
    criterion = np.sqrt(tolerance) * grad_norm
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
        d_k = - grad

        alpha = line_search_tool.line_search(oracle, x_k, d_k,
                                             previous_alpha=previous_alpha)
   
        x_k = x_k + alpha * d_k
    
        i += 1
        if  display and i % 50 == 0:
            print(f'Итерация: {i}, время: {(datetime.now() - start_time).total_seconds()}')
            print(f'Норма градиента: {grad_norm}')
        if trace:
            current_time = datetime.now()
            history['time'].append((current_time - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_norm)
            if len(x_k) <= 2:
                history['x'].append(x_k)
    if i == max_iter and grad_norm > criterion:
        message = 'iterations_exceeded'
    return (x_k, message, history)
