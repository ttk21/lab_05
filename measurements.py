import numpy as np
import scipy.linalg
from pylie import SE3

from camera import PerspectiveCamera


class PrecalibratedCameraMeasurementsFixedWorld:
    """Measurements of fixed world points given in the normalised image plane"""

    def __init__(self, camera: PerspectiveCamera, u: np.ndarray, covs_u: list, x_w: np.ndarray):
        """Constructs the 2D-3D measurement

        :param camera: A PerspectiveCamera representing the camera that performed the measurement.
        :param u: A 2xn matrix of n pixel observations.
        :param covs_u: A list of covariance matrices representing the uncertainty in each pixel observation.
        :param x_w: A 3xn matrix of the n corresponding world points.
        """

        self.camera = camera
        self.x_w = x_w

        # Transform to the normalised image plane.
        self.xn = camera.pixel_to_normalised(u)

        # Propagate uncertainty, and precompute square root of information matrices.
        self.num = u.shape[1]
        self.covs = [np.identity(2)] * self.num
        self.sqrt_inv_covs = [np.identity(2)] * self.num
        for c in range(self.num):
            self.covs[c] = self.camera.pixel_cov_to_normalised_com(covs_u[c])
            self.sqrt_inv_covs[c] = scipy.linalg.sqrtm(scipy.linalg.inv(self.covs[c]))

    @classmethod
    def generate(cls, camera: PerspectiveCamera, true_pose_w_c: SE3, true_points_w: np.ndarray):
        """Generate a 2D-3D measurement

        :param camera: A PerspectiveCamera representing the camera that performed the measurement.
        :param true_pose_w_c: The true pose of the camera in the world frame.
        :param true_points_w: The true world points.
        :return: The generated measurements
        """
        num_points = true_points_w.shape[1]

        # Generate observations in pixels.
        u = camera.project_to_pixel(true_pose_w_c.inverse() * true_points_w)
        covs_u = [np.diag(np.array([2, 2]) ** 2)] * num_points  # Same for all observations.

        # Add noise according to uncertainty.
        for c in range(num_points):
            u[:2, [c]] = u[:2, [c]] + np.random.multivariate_normal(np.zeros(2), covs_u[c]).reshape(-1, 1)

        # Construct measurement.
        return cls(camera, u, covs_u, true_points_w)


class PrecalibratedCameraMeasurementsFixedCamera:
    """Measurements of world points given in the normalised image plane of a fixed camera"""

    def __init__(self, camera: PerspectiveCamera, pose_w_c: SE3, u: np.ndarray, covs_u: list):
        """Constructs the 2D-3D measurement

        :param camera: A PerspectiveCamera representing the camera that performed the measurement.
        :param pose_w_c: The pose of the camera in the world frame.
        :param u: A 2xn matrix of n pixel observations corresponding to each and every 3D world point state.
        :param covs_u: A list of covariance matrices representing the uncertainty in each pixel observation.
        """

        self.camera = camera
        self.pose_w_c = pose_w_c
        self.pose_c_w = pose_w_c.inverse()

        # Transform to the normalised image plane.
        self.xn = camera.pixel_to_normalised(u)

        # Propagate uncertainty, and precompute square root of information matrices.
        self.num = u.shape[1]
        self.covs = [np.identity(2)] * self.num
        self.sqrt_inv_covs = [np.identity(2)] * self.num
        for c in range(self.num):
            self.covs[c] = self.camera.pixel_cov_to_normalised_com(covs_u[c])
            self.sqrt_inv_covs[c] = scipy.linalg.sqrtm(scipy.linalg.inv(self.covs[c]))

    @classmethod
    def generate(cls, camera: PerspectiveCamera, true_pose_w_c: SE3, true_points_w: np.ndarray):
        """Generate a 2D-3D measurement

        :param camera: A PerspectiveCamera representing the camera that performed the measurement.
        :param true_pose_w_c: The true pose of the camera in the world frame.
        :param true_points_w: The true world points.
        :return: The generated measurements
        """
        num_points = true_points_w.shape[1]

        # Generate observations in pixels.
        u = camera.project_to_pixel(true_pose_w_c.inverse() * true_points_w)
        covs_u = [np.diag(np.array([2, 2]) ** 2)] * num_points  # Same for all observations.

        # Add noise according to uncertainty.
        for c in range(num_points):
            u[:2, [c]] = u[:2, [c]] + np.random.multivariate_normal(np.zeros(2), covs_u[c]).reshape(-1, 1)

        # Construct measurement.
        return cls(camera, true_pose_w_c, u, covs_u)


class PrecalibratedCameraMeasurements:
    """Measurements of world points given in the normalised image plane"""

    def __init__(self, camera: PerspectiveCamera, u: np.ndarray, covs_u: list):
        """Constructs the 2D-3D measurement

        :param camera: A PerspectiveCamera representing the camera that performed the measurement.
        :param u: A 2xn matrix of n pixel observations.
        :param covs_u: A list of covariance matrices representing the uncertainty in each pixel observation.
        """

        self.camera = camera

        # Transform to the normalised image plane.
        self.xn = camera.pixel_to_normalised(u)

        # Propagate uncertainty, and precompute square root of information matrices.
        self.num = u.shape[1]
        self.covs = [np.identity(2)] * self.num
        self.sqrt_inv_covs = [np.identity(2)] * self.num
        for c in range(self.num):
            self.covs[c] = self.camera.pixel_cov_to_normalised_com(covs_u[c])
            self.sqrt_inv_covs[c] = scipy.linalg.sqrtm(scipy.linalg.inv(self.covs[c]))

    @classmethod
    def generate(cls, camera: PerspectiveCamera, true_pose_w_c: SE3, true_points_w: np.ndarray):
        """Generate a 2D-3D measurement

        :param camera: A PerspectiveCamera representing the camera that performed the measurement.
        :param true_pose_w_c: The true pose of the camera in the world frame.
        :param true_points_w: The true world points.
        :return: The generated measurements
        """
        num_points = true_points_w.shape[1]

        # Generate observations in pixels.
        u = camera.project_to_pixel(true_pose_w_c.inverse() * true_points_w)
        covs_u = [np.diag(np.array([2, 2]) ** 2)] * num_points  # Same for all observations.

        # Add noise according to uncertainty.
        for c in range(num_points):
            u[:2, [c]] = u[:2, [c]] + np.random.multivariate_normal(np.zeros(2), covs_u[c]).reshape(-1, 1)

        # Construct measurement.
        return cls(camera, u, covs_u)
