import numpy as np
import visgeom as vg

from camera import PerspectiveCamera
from measurements import PrecalibratedCameraMeasurementsFixedCamera
from optim import CompositeStateVariable, levenberg_marquardt

from visualise_ba import visualise_soba

"""Example 3 - Structure-only Bundle Adjustment"""


class PrecalibratedStructureOnlyBAObjective:
    """Implements linearisation of the structure-only BA objective function"""

    def __init__(self, measurements):
        """Constructs the objective

        :param measurements: A list of PrecalibratedCameraMeasurementsFixedCamera objects, one for each camera.
        """
        self.measurements = measurements

    @staticmethod
    def extract_measurement_jacobian(point_index, point_state_w, measurement):
        """Computes the measurement Jacobian for a specific point and camera measurement.

        :param point_index: Index of current point.
        :param point_state_w: Current state of a specific world point.
        :param measurement: The measurement
        :return: The measurement Jacobian
        """
        A = measurement.sqrt_inv_covs[point_index] @ \
            measurement.camera.jac_project_world_to_normalised_wrt_x_w(measurement.pose_c_w, point_state_w)

        return A

    @staticmethod
    def extract_measurement_error(point_index, point_state_w, measurement):
        """Computes the measurement error for a specific point and camera measurement.

        :param point_index: Index of current point.
        :param point_state_w: Current state of a specific world point.
        :param measurement: The measurement
        :return: The measurement error
        """
        b = measurement.sqrt_inv_covs[point_index] @ \
            measurement.camera.reprojection_error_normalised(measurement.pose_c_w * point_state_w,
                                                             measurement.xn[:, [point_index]])

        return b

    def linearise(self, point_states_w):
        """Linearises the objective over all states and measurements

        :param point_states_w: The current state of the points in the world frame.
        :return:
          A - The full measurement Jacobian
          b - The full measurement error
          cost - The current cost
        """
        num_cameras = len(self.measurements)
        num_points = len(point_states_w)

        A = np.zeros((2 * num_cameras * num_points, 3 * num_points))
        b = np.zeros((2 * num_cameras * num_points, 1))

        for i in range(num_cameras):
            for j in range(num_points):
                rows = slice(i * 2 * num_points + j * 2, i * 2 * num_points + (j + 1) * 2)
                cols = slice(j * 3, (j + 1) * 3)
                A[rows, cols] = self.extract_measurement_jacobian(j, point_states_w[j], self.measurements[i])
                b[rows, :] = self.extract_measurement_error(j, point_states_w[j], self.measurements[i])

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
    measurements = \
        [PrecalibratedCameraMeasurementsFixedCamera.generate(camera, pose, true_points_w) for pose in true_poses_w_c]

    # Construct model from measurements.
    model = PrecalibratedStructureOnlyBAObjective(measurements)

    # Perturb world points and use as initial state.
    init_points_w = [true_points_w[:, [i]] + 0.3 * np.random.randn(3, 1) for i in range(true_points_w.shape[1])]
    init_state = CompositeStateVariable(init_points_w)

    # Estimate pose in the world frame from point correspondences.
    x, cost, A, b = levenberg_marquardt(init_state, model)
    cov_x_final = np.linalg.inv(A.T @ A)

    # Print covariance.
    with np.printoptions(precision=3, suppress=True):
        print('Covariance:')
        print(cov_x_final)

    # Visualise
    visualise_soba(true_poses_w_c, true_points_w, measurements, x, cost)


if __name__ == "__main__":
    main()
