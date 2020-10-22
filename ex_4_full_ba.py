import numpy as np
import visgeom as vg

from camera import PerspectiveCamera
from measurements import PrecalibratedCameraMeasurements
from optim import BundleAdjustmentState, levenberg_marquardt
from visualise_ba import visualise_full

"""Example 4 - Full Bundle Adjustment"""


class PrecalibratedFullBAObjective:
    """Implements linearisation of the full BA objective function"""

    def __init__(self, measurements, pose1_prior, point1_prior):
        """Constructs the objective

        :param measurements: A list of PrecalibratedCameraMeasurements objects, one for each camera.
        :param pose1_prior: The prior state of the first pose
        :param point1_prior: The prior state of the first point
        """
        self.measurements = measurements
        self.pose1_prior = pose1_prior
        self.point1_prior = point1_prior

    @staticmethod
    def extract_measurement_jacobian_wrt_pose(point_index, pose_state_c_w, point_state_w, measurement):
        """Computes the measurement Jacobian wrt the pose for a specific point and camera measurement.

        :param point_index: Index of current point.
        :param pose_state_c_w: Current pose state given as the pose of the world in the camera frame.
        :param point_state_w: Current point state in the world frame
        :param measurement: The measurement
        :return: The measurement Jacobian
        """
        P = measurement.sqrt_inv_covs[point_index] @ \
            measurement.camera.jac_project_world_to_normalised_wrt_pose_w_c(pose_state_c_w, point_state_w)

        return P

    @staticmethod
    def extract_measurement_jacobian_wrt_point(point_index, pose_state_c_w, point_state_w, measurement):
        """Computes the measurement Jacobian wrt the point for a specific point and camera measurement.

        :param point_index: Index of current point.
        :param pose_state_c_w: Current pose state given as the pose of the world in the camera frame.
        :param point_state_w: Current state of a specific world point.
        :param measurement: The measurement
        :return: The measurement Jacobian
        """
        S = measurement.sqrt_inv_covs[point_index] @ \
            measurement.camera.jac_project_world_to_normalised_wrt_x_w(pose_state_c_w, point_state_w)

        return S

    @staticmethod
    def extract_measurement_error(point_index, pose_state_c_w, point_state_w, measurement):
        """Computes the measurement error for a specific point and camera measurement.

        :param point_index: Index of current point.
        :param pose_state_c_w: Current pose state given as the pose of the world in the camera frame.
        :param point_state_w: Current state of a specific world point.
        :param measurement: The measurement
        :return: The measurement error
        """
        b = measurement.sqrt_inv_covs[point_index] @ \
            measurement.camera.reprojection_error_normalised(pose_state_c_w * point_state_w,
                                                             measurement.xn[:, [point_index]])

        return b

    def linearise(self, states):
        """Linearises the objective over all states and measurements

        :param states: The current pose and point states.
        :return:
          A - The full measurement Jacobian
          b - The full measurement error
          cost - The current cost
        """
        num_cameras = states.num_poses
        num_points = states.num_points

        A = np.zeros((2 * num_cameras * num_points + 9, 6 * num_cameras + 3 * num_points))
        b = np.zeros((2 * num_cameras * num_points + 9, 1))

        for i in range(num_cameras):
            pose_state_c_w = states.get_pose(i).inverse()

            for j in range(num_points):
                point_state_w = states.get_point(j)

                # Insert submatrix for Jacobian wrt poses.
                rows = slice(i * 2 * num_points + j * 2, i * 2 * num_points + (j + 1) * 2)
                cols = slice(i * 6, (i + 1) * 6)
                A[rows, cols] = self.extract_measurement_jacobian_wrt_pose(j, pose_state_c_w, point_state_w,
                                                                           self.measurements[i])

                # Insert submatrix for Jacobian wrt points.
                cols = slice(6 * num_cameras + j * 3, 6 * num_cameras + (j + 1) * 3)
                A[rows, cols] = self.extract_measurement_jacobian_wrt_point(j, pose_state_c_w, point_state_w,
                                                                            self.measurements[i])

                b[rows, :] = self.extract_measurement_error(j, pose_state_c_w, point_state_w, self.measurements[i])

        # Add priors.
        A[2 * num_cameras * num_points:2 * num_cameras * num_points + 6, :6] = np.identity(6)
        b[2 * num_cameras * num_points:2 * num_cameras * num_points + 6] = -(states.get_pose(0) - self.pose1_prior)

        A[2 * num_cameras * num_points + 6:, 6 * num_cameras:6 * num_cameras + 3] = np.identity(3)
        b[2 * num_cameras * num_points + 6:] = -(states.get_point(0) - self.point1_prior)

        return A, b, b.T.dot(b)


def main():
    # World box.
    true_points_w = vg.utils.generate_box()

    # Define common camera parameters.
    w = 640
    h = 480
    focal_lengths = 0.75 * h * np.ones((2, 1))
    principal_point = 0.5 * np.array([[w, h]]).T
    camera = PerspectiveCamera(focal_lengths, principal_point)

    # Define a set of cameras.
    true_poses_w_c = [
        PerspectiveCamera.looks_at_pose(np.array([[3, -4, 0]]).T, np.zeros((3, 1)), np.array([[0, 0, 1]]).T),
        PerspectiveCamera.looks_at_pose(np.array([[3, 4, 0]]).T, np.zeros((3, 1)), np.array([[0, 0, 1]]).T)]

    # Generate a set of camera measurements.
    measurements = [PrecalibratedCameraMeasurements.generate(camera, pose, true_points_w) for pose in true_poses_w_c]

    # Construct model from measurements.
    model = PrecalibratedFullBAObjective(measurements, true_poses_w_c[0], true_points_w[:, [0]])

    # Perturb camera poses and world points and use as initial state.
    init_poses_wc = [pose + 0.3 * np.random.randn(6, 1) for pose in true_poses_w_c]
    init_points_w = [true_points_w[:, [i]] + 0.3 * np.random.randn(3, 1) for i in range(true_points_w.shape[1])]
    init_state = BundleAdjustmentState(init_poses_wc, init_points_w)

    # Estimate pose in the world frame from point correspondences.
    x, cost, A, b = levenberg_marquardt(init_state, model)
    cov_x_final = np.linalg.inv(A.T @ A)

    # Print covariance.
    with np.printoptions(precision=3, suppress=True):
        print('Covariance:')
        print(cov_x_final)

    # Visualise
    visualise_full(true_poses_w_c, true_points_w, measurements, x, cost)


if __name__ == "__main__":
    main()
