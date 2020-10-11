import numpy as np
import visgeom as vg
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg
from pylie import SE3
from optim import levenberg_marquardt
from camera import PerspectiveCamera

"""Example 1 - Motion-only Bundle Adjustment"""

class PrecalibratedCameraMeasurements:
    def __init__(self, camera, u, covs_u, x_w):
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



class PrecalibratedMotionOnlyBAObjective:
    """Implements linearisation of motion-only BA objective function"""

    def __init__(self, measurement):
        self.measurement = measurement

    def extract_measurement_jacobian(self, T_cw, measurement):
        A = np.zeros((2 * measurement.num, 6))

        # Enter the submatrices from each measurement:
        for i in range(measurement.num):
            A[2 * i:2 * (i + 1), :] = \
                measurement.sqrt_inv_covs[i] @ \
                measurement.camera.jac_project_world_to_normalised_wrt_pose_w_c(T_cw, measurement.x_w[:, [i]])

        return A

    def extract_measurement_error(self, T_cw, measurement):
        b = np.zeros((2 * measurement.num, 1))

        # Enter the submatrices from each measurement:
        for i in range(measurement.num):
            b[2 * i:2 * (i + 1)] = \
                measurement.sqrt_inv_covs[i] @ \
                measurement.camera.reprojection_error_normalised(T_cw * measurement.x_w[:, [i]], measurement.xn[:, [i]])

        return b

    def linearise(self, T_wc: SE3):
        T_cw = T_wc.inverse()

        A = self.extract_measurement_jacobian(T_cw, self.measurement)
        b = self.extract_measurement_error(T_cw, self.measurement)

        return A, b, b.T.dot(b)

def visualise(true_pose_w_c, true_box_w, measurement, x, cost):
    # Visualize (press a key to jump to the next iteration).
    # Use Qt 5 backend in visualisation.
    matplotlib.use('qt5agg')

    # Create figure and axis.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plot box and true state
    vg.plot_pose(ax, true_pose_w_c.to_tuple(), scale=1, alpha=0.4)
    vg.utils.plot_as_box(ax, true_box_w)

    # Normalised in 3d.
    xn_3d = np.vstack((measurement.xn, np.ones((1, measurement.num))))


    # Plot initial state (to run axis equal first time).
    ax.set_title('Cost: ' + str(cost[0]))
    artists = vg.plot_pose(ax, x[0].to_tuple(), scale=1)
    artists.extend(vg.plot_camera_image_plane(ax, measurement.camera.K(), x[0].to_tuple()))
    artists.extend(vg.utils.plot_as_box(ax, x[0] * xn_3d, alpha=0.4))
    artists.extend(vg.utils.plot_as_box(ax, x[0] * measurement.camera.project_to_normalised_3d(x[0].inverse() * measurement.x_w)))
    vg.plot.axis_equal(ax)
    plt.draw()

    while True:
        if plt.waitforbuttonpress():
            break

    # Plot iterations
    for i in range(1, len(x)):
        for artist in artists:
            artist.remove()

        ax.set_title('Cost: ' + str(cost[i]))
        artists = vg.plot_pose(ax, x[i].to_tuple(), scale=1)

        artists.extend(vg.plot_camera_image_plane(ax, measurement.camera.K(), x[i].to_tuple()))
        artists.extend(vg.utils.plot_as_box(ax, x[i] * xn_3d, alpha=0.4))
        artists.extend(vg.utils.plot_as_box(ax, x[i] * measurement.camera.project_to_normalised_3d(x[i].inverse() * measurement.x_w)))
        plt.draw()
        while True:
            if plt.waitforbuttonpress():
                break


def generate_measurements(camera, true_pose_w_c, true_points_w):
    num_points = true_points_w.shape[1]

    # Generate observations in pixels.
    u = camera.project_to_pixel(true_pose_w_c.inverse() * true_points_w)
    covs_u = [np.diag(np.array([2, 2]) ** 2)] * num_points  # Same for all observations.

    # Add noise according to uncertainty.
    for c in range(num_points):
        u[:2, [c]] = u[:2, [c]] + np.random.multivariate_normal(np.zeros(2), covs_u[c]).reshape(-1, 1)

    # Construct measurement.
    return PrecalibratedCameraMeasurements(camera, u, covs_u, true_points_w)

def main():
    # World box.
    points_w = vg.utils.generate_box()

    # Define common camera.
    w = 640
    h = 480
    focal_lengths = 0.75 * h * np.ones((2, 1))
    principal_point = 0.5 * np.array([[w, h]]).T
    camera = PerspectiveCamera(focal_lengths, principal_point)

    # Generate a set of camera measurements.
    true_pose_w_c = PerspectiveCamera.looks_at_pose(np.array([[3, -4, 0]]).T, np.zeros((3, 1)), np.array([[0, 0, 1]]).T)
    measurement = generate_measurements(camera, true_pose_w_c, points_w)

    # Construct model from measurements.
    model = PrecalibratedMotionOnlyBAObjective(measurement)

    # Perturb camera pose and use as initial state.
    init_pose_wc = true_pose_w_c + 0.3 * np.random.randn(6, 1)

    # Estimate pose in the world frame from point correspondences.
    x, cost, A, b = levenberg_marquardt(init_pose_wc, model)
    cov_x_final = np.linalg.inv(A.T @ A)

    # Print covariance.
    with np.printoptions(precision=3, suppress=True):
        print('Covariance:')
        print(cov_x_final)

    # Visualise
    visualise(true_pose_w_c, points_w, measurement, x, cost)


if __name__ == "__main__":
    main()
