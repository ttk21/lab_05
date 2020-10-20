import numpy as np
import scipy.linalg


class CompositeStateVariable:
    """A composition of a set of state variables.
    All variables must implement the __add__ and __sub__ operators appropriately,
    and return the dimension of the tangent space with the __len__ operator.
    """

    @property
    def variables(self):
        """The list of variables"""
        return self._variables

    @property
    def dim(self):
        """The dimension of the tangent space for the composited state manifold"""
        return self._dim

    def __init__(self, variables):
        """Constructs a composited state variable

        :param variables: A list of state variables.
        """
        self._variables = variables

        self._dim = 0
        for var in self.variables:
            self._dim = self._dim + len(var)

    def __add__(self, update_vector):
        """Add operator performs the "oplus" operation over all states.

        :param update_vector: A vector with the same dimension as self.dim().
        :return: The perturbed composite state
        """
        updated_variables = [None] * len(self.variables)
        start_elem = 0
        for i, current_state in enumerate(self.variables):
            end_elem = start_elem + len(current_state)
            updated_variables[i] = current_state + update_vector[start_elem:end_elem, :]
            start_elem = end_elem

        return CompositeStateVariable(updated_variables)

    def __getitem__(self, item):
        """Gets a specific state variable from the composite.

        :param item: An index in [0, len(self)).
        :return: The state
        """
        return self._variables[item]

    def __len__(self):
        """Returns the number of individual states in the composite

        :return: The number of states
        """
        return len(self._variables)

    def __sub__(self, other):
        """Subtract operator performs the "ominus" operation over all states.

        :param other: The other composite state
        :return: The difference vector in the composite tangent space
        """
        tau = np.zeros((self.dim, 1))
        start_elem = 0
        for i, (this_state, other_state) in enumerate(zip(self.variables, other.variables)):
            end_elem = start_elem + len(this_state)
            tau[start_elem:end_elem, :] = this_state - other_state
            start_elem = end_elem

        return tau


class BundleAdjustmentState(CompositeStateVariable):
    """A composition of pose and point states for BA"""

    @property
    def num_poses(self):
        """The number of pose states"""
        return self._num_poses

    @property
    def num_points(self):
        """The number of point states"""
        return self._num_points

    def __init__(self, poses, points):
        """Constructs the composite state for BA problems

        :param poses: A list of pose states, one for each camera
        :param points: A list of point states
        """
        super().__init__(poses + points)
        self._num_poses = len(poses)
        self._num_points = len(points)

    def get_pose(self, pose_index):
        """Gets a specific pose state from the composite

        :param pose_index: An index in [0, self.num_poses)
        :return: The pose
        """
        return self._variables[pose_index]

    def get_point(self, point_index):
        """Gets a specific point state from the composite

        :param point_index: An index in [0, self.num_points)
        :return: The point
        """
        return self._variables[self._num_poses + point_index]

    def __add__(self, other):
        comp = super().__add__(other)
        return BundleAdjustmentState(comp.variables[:self.num_poses], comp.variables[self.num_poses:])


def gauss_newton(x_init, model, cost_thresh=1e-9, delta_thresh=1e-9, max_num_it=10):
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


def levenberg_marquardt(x_init, model, cost_thresh=1e-9, delta_thresh=1e-9, max_num_it=10):
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
