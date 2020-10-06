import numpy as np
import visgeom as vg
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg
from pylie import SO3, SE3
from optim import levenberg_marquardt
from camera import PerspectiveCamera

"""Example 1 - Motion-only Bundle Adjustment"""

class SimpleMotionOnlyBAObjective:
    """Implements linearisation of motion-only BA objective function"""

    def __init__(self, K, x_w, x_n, covs):
        if x_w.shape[0] != 3 or x_n.shape[0] != 2 or x_w.shape[1] != x_n.shape[1]:
            raise TypeError('Matrices with corresponding points must have same size')

        self.K = K
        self.x_w = x_w
        self.x_n = x_n
        self.num_points = x_w.shape[1]

        if len(covs) != self.num_points:
            raise TypeError('Must have a covariance matrix for each point observation')

        # Compute square root of information matrices.
        self.sqrt_inv_covs = [None] * self.num_points
        for i in range(self.num_points):
            self.sqrt_inv_covs[i] = scipy.linalg.sqrtm(scipy.linalg.inv(covs[i]))

    def linearise(self, T_wc: SE3):
        cam = PerspectiveCamera(self.K, T_wc)

        A = np.zeros((2 * self.num_points, 6))
        b = np.zeros((2 * self.num_points, 1))

        # Enter the submatrices from each measurement:
        for i in range(self.num_points):
            A[2 * i:2 * (i + 1), :] = self.sqrt_inv_covs[i] @ \
                                      cam.jac_project_normalised_wrt_pose_w_c(self.x_w[:, [i]])
            b[2 * i:2 * (i + 1)] = self.sqrt_inv_covs[i] @ \
                                   cam.reprojection_error_normalised(self.x_w[:, [i]], self.x_n[:, [i]])

        return A, b, b.T.dot(b)

def visualise(true_cam, points_w, observed_x_n, x, cost):
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
    vg.plot_pose(ax, true_cam.pose_w_c.to_tuple(), scale=1, alpha=0.4)
    vg.utils.plot_as_box(ax, points_w)

    # Plot initial state (to run axis equal first time).
    ax.set_title('Cost: ' + str(cost[0]))
    artists = vg.plot_pose(ax, x[0].to_tuple(), scale=1)
    curr_cam = PerspectiveCamera(true_cam.K, x[0])
    artists.extend(vg.plot_camera_image_plane(ax, curr_cam.K, curr_cam.pose_w_c.to_tuple()))
    artists.extend(vg.utils.plot_as_box(ax, curr_cam.camera_to_world(observed_x_n), alpha=0.4))
    artists.extend(vg.utils.plot_as_box(ax, curr_cam.camera_to_world(curr_cam.project_to_normalised_3d(points_w))))
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

        curr_cam = PerspectiveCamera(true_cam.K, x[i])
        artists.extend(vg.plot_camera_image_plane(ax, curr_cam.K, curr_cam.pose_w_c.to_tuple()))
        artists.extend(vg.utils.plot_as_box(ax, curr_cam.camera_to_world(observed_x_n), alpha=0.4))
        artists.extend(vg.utils.plot_as_box(ax, curr_cam.camera_to_world(curr_cam.project_to_normalised_3d(points_w))))
        plt.draw()
        while True:
            if plt.waitforbuttonpress():
                break


def main():
    # World box.
    points_w = vg.utils.generate_box()

    # True camera.
    w = 640
    h = 480
    f = 0.75 * h
    K = np.array([[f, 0, 0.5*w], [0, f, 0.5*h], [0, 0, 1]])
    true_cam = PerspectiveCamera.looks_at(K, np.array([[3, -4, 0]]).T, np.zeros((3, 1)), np.array([[0, 0, 1]]).T)

    # Box observations.
    observed_x_n = true_cam.project_to_normalised_3d(points_w)
    num_points = observed_x_n.shape[1]
    point_covariances = [np.diag(np.array([1e-3, 1e-3]) ** 2)] * num_points
    for c in range(num_points):
        observed_x_n[:2, [c]] = observed_x_n[:2, [c]] + \
                                np.random.multivariate_normal(np.zeros(2), point_covariances[c]).reshape(-1, 1)

    # Perturb observer pose and use as initial state.
    init_pose_wc = true_cam.pose_w_c + 0.3 * np.random.randn(6, 1)

    # Estimate pose in the world frame from point correspondences.
    model = SimpleMotionOnlyBAObjective(true_cam.K, points_w, observed_x_n[:2, :], point_covariances)
    x, cost, A, b = levenberg_marquardt(init_pose_wc, model)
    cov_x_final = np.linalg.inv(A.T @ A)

    # Print covariance.
    with np.printoptions(precision=3, suppress=True):
        print('Covariance:')
        print(cov_x_final)

    # Visualise
    visualise(true_cam, points_w, observed_x_n, x, cost)


if __name__ == "__main__":
    main()
