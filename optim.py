import numpy as np
import scipy.linalg

def gauss_newton(x_init, model, cost_thresh=1e-14, delta_thresh=1e-14, max_num_it=10):
    """Implements nonlinear least squares using the Gauss-Newton algorithm

    :param x_init: The initial state
    :param model: Model with a function linearise() the returns A, b and the cost for the current state estimate.
    :param cost_thresh: Threshold for cost function
    :param delta_thresh: Threshold for update vector
    :param max_num_it: Maximum number of iterations
    :return:
      - x: State estimates at each iteration, the final state in x[-1]
      - cost: The cost at each iteration
      - A: The full measurement Jacobian at the final state
      - b: The full measurement error at the final state
    """
    x = [None] * (max_num_it + 1)
    cost = np.zeros(max_num_it + 1)

    x[0] = x_init
    for it in range(max_num_it):
        A, b, cost[it] = model.linearise(x[it])
        tau = np.linalg.lstsq(A, b, rcond=None)[0]
        x[it + 1] = x[it] + tau

        if cost[it] < cost_thresh or np.linalg.norm(tau) < delta_thresh:
            x = x[:it + 2]
            cost = cost[:it + 2]
            break

    A, b, cost[-1] = model.linearise(x[-1])

    return x, cost, A, b


def levenberg_marquardt(x_init, model, cost_thresh=1e-14, delta_thresh=1e-14, max_num_it=10):
    """Implements nonlinear least squares using the Levenberg-Marquardt algorithm

    :param x_init: The initial state
    :param model: Model with a function linearise() the returns A, b and the cost for the current state estimate.
    :param cost_thresh: Threshold for cost function
    :param delta_thresh: Threshold for update vector
    :param max_num_it: Maximum number of iterations
    :return:
      - x: State estimates at each iteration, the final state in x[-1]
      - cost: The cost at each iteration
      - A: The full measurement Jacobian at the final state
      - b: The full measurement error at the final state
    """
    x = [None] * (max_num_it + 1)
    cost = np.zeros(max_num_it + 1)

    x[0] = x_init
    A, b, cost[0] = model.linearise(x[0])

    curr_lambda = 1e-4
    for it in range(max_num_it):
        inf_mat = A.T @ A

        tau = scipy.linalg.solve(inf_mat + np.diag(curr_lambda * np.diag(inf_mat)), A.T @ b, 'pos')
        x_new = x[it] + tau

        A, b, cost_new = model.linearise(x_new)

        if cost_new < cost[it]:
            x[it + 1] = x_new
            cost[it + 1] = cost_new
            curr_lambda = 0.1 * curr_lambda
        else:
            x[it + 1] = x[it]
            cost[it + 1] = cost[it]

        if cost[it] < cost_thresh or np.linalg.norm(tau) < delta_thresh:
            x = x[:it + 2]
            cost = cost[:it + 2]
            break

    return x, cost, A, b
