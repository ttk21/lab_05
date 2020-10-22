import numpy as np
import visgeom as vg

from camera import PerspectiveCamera
from measurements import PrecalibratedCameraMeasurementsFixedWorld
from optim import CompositeStateVariable, levenberg_marquardt

from visualise_ba import visualise_multicam_moba

"""Example 2 - Multicamera motion-only Bundle Adjustment"""


class PrecalibratedMulticameraMotionOnlyBAObjective:
    """Implements linearisation of the multicamera motion-only BA objective function"""

    def __init__(self, measurements):
        """Constructs the objective

        :param measurements: A list of PrecalibratedCameraMeasurementsFixedWorld objects, one for each camera.
        """
        self.measurements = measurements

    @staticmethod
    def extract_measurement_jacobian(point_index, pose_state_c_w, measurement):
        """Computes the measurement Jacobian for a specific point and camera measurement.

        :param point_index: Index of current point.
        :param pose_state_c_w: Current pose state given as the pose of the world in the camera frame.
        :param measurement: The measurement
        :return: The measurement Jacobian
        """
        A = measurement.sqrt_inv_covs[point_index] @ \
            measurement.camera.jac_project_world_to_normalised_wrt_pose_w_c(pose_state_c_w,
                                                                            measurement.x_w[:, [point_index]])

        return A

    @staticmethod
    def extract_measurement_error(point_index, pose_state_c_w, measurement):
        """Computes the measurement error for a specific point and camera measurement.

        :param point_index: Index of current point.
        :param pose_state_c_w: Current pose state given as the pose of the world in the camera frame.
        :param measurement: The measurement
        :return: The measurement error
        """
        b = measurement.sqrt_inv_covs[point_index] @ \
            measurement.camera.reprojection_error_normalised(pose_state_c_w * measurement.x_w[:, [point_index]],
                                                             measurement.xn[:, [point_index]])

        return b

    def linearise(self, pose_states_w_c):
        """Linearises the objective over all states and measurements

        :param pose_states_w_c: The current camera pose states in the world frame.
        :return:
          A - The full measurement Jacobian
          b - The full measurement error
          cost - The current cost
        """
        num_cameras = len(self.measurements)
        num_points = self.measurements[0].num

        A = np.zeros((2 * num_cameras * num_points, 6 * num_cameras))
        b = np.zeros((2 * num_cameras * num_points, 1))

        for i in range(num_cameras):
            pose_state_c_w = pose_states_w_c[i].inverse()

            for j in range(num_points):
                rows = slice(i * 2 * num_points + j * 2, i * 2 * num_points + (j + 1) * 2)
                cols = slice(i * 6, (i + 1) * 6)
                A[rows, cols] = self.extract_measurement_jacobian(j, pose_state_c_w, self.measurements[i])
                b[rows, :] = self.extract_measurement_error(j, pose_state_c_w, self.measurements[i])

        return A, b, b.T.dot(b)


def main():
    # World box.
    points_w = vg.utils.generate_box()

    # Define common camera parameters.
    w = 640
    h = 480
    focal_lengths = 0.75 * h * np.ones((2, 1))
    principal_point = 0.5 * np.array([[w, h]]).T
    camera = PerspectiveCamera(focal_lengths, principal_point)

    # Generate a set of camera measurements.
    true_poses_w_c = [
        PerspectiveCamera.looks_at_pose(np.array([[3, -4, 0]]).T, np.zeros((3, 1)), np.array([[0, 0, 1]]).T),
        PerspectiveCamera.looks_at_pose(np.array([[3, 4, 0]]).T, np.zeros((3, 1)), np.array([[0, 0, 1]]).T)]
    measurements = \
        [PrecalibratedCameraMeasurementsFixedWorld.generate(camera, pose, points_w) for pose in true_poses_w_c]

    # Construct model from measurements.
    model = PrecalibratedMulticameraMotionOnlyBAObjective(measurements)

    # Perturb camera pose and use as initial state.
    init_poses_wc = [pose + 0.3 * np.random.randn(6, 1) for pose in true_poses_w_c]
    init_state = CompositeStateVariable(init_poses_wc)

    # Estimate pose in the world frame from point correspondences.
    x, cost, A, b = levenberg_marquardt(init_state, model)
    cov_x_final = np.linalg.inv(A.T @ A)

    # Print covariance.
    with np.printoptions(precision=3, suppress=True):
        print('Covariance:')
        print(cov_x_final)

    # Visualise
    visualise_multicam_moba(true_poses_w_c, points_w, measurements, x, cost)


if __name__ == "__main__":
    main()
