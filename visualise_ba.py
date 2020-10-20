import matplotlib
import numpy as np
import visgeom as vg
from matplotlib import pyplot as plt


def visualise_moba(true_pose_w_c, true_box_w, measurement, x, cost):
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
    artists.extend(
        vg.utils.plot_as_box(ax, x[0] * measurement.camera.project_to_normalised_3d(x[0].inverse() * measurement.x_w)))
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
        artists.extend(vg.utils.plot_as_box(ax, x[i] * measurement.camera.project_to_normalised_3d(
            x[i].inverse() * measurement.x_w)))
        plt.draw()
        while True:
            if plt.waitforbuttonpress():
                break

    plt.close()


def visualise_multicam_moba(true_pose_w_c, true_box_w, measurement, x, cost):
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
    for true_pose in true_pose_w_c:
        vg.plot_pose(ax, true_pose.to_tuple(), scale=1, alpha=0.4)
    vg.utils.plot_as_box(ax, true_box_w)

    # Plot initial state (to run axis equal first time).
    ax.set_title('Cost: ' + str(cost[0]))
    artists = []
    for pose, meas in zip(x[0], measurement):
        # Normalised in 3d.
        xn_3d = np.vstack((meas.xn, np.ones((1, meas.num))))

        artists.extend(vg.plot_pose(ax, pose.to_tuple(), scale=1))
        artists.extend(vg.plot_camera_image_plane(ax, meas.camera.K(), pose.to_tuple()))
        artists.extend(vg.utils.plot_as_box(ax, pose * xn_3d, alpha=0.4))
        artists.extend(
            vg.utils.plot_as_box(ax, pose * meas.camera.project_to_normalised_3d(pose.inverse() * meas.x_w)))
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
        artists = []
        for pose, meas in zip(x[i], measurement):
            # Normalised in 3d.
            xn_3d = np.vstack((meas.xn, np.ones((1, meas.num))))

            artists.extend(vg.plot_pose(ax, pose.to_tuple(), scale=1))
            artists.extend(vg.plot_camera_image_plane(ax, meas.camera.K(), pose.to_tuple()))
            artists.extend(vg.utils.plot_as_box(ax, pose * xn_3d, alpha=0.4))
            artists.extend(vg.utils.plot_as_box(ax, pose * meas.camera.project_to_normalised_3d(
                pose.inverse() * meas.x_w)))

        plt.draw()
        while True:
            if plt.waitforbuttonpress():
                break

    plt.close()


def visualise_soba(true_pose_w_c, true_box_w, measurement, x, cost):
    # Visualize (press a key to jump to the next iteration).
    # Use Qt 5 backend in visualisation.
    matplotlib.use('qt5agg')

    # Create figure and axis.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plot true box
    vg.utils.plot_as_box(ax, true_box_w, alpha=0.4)

    # Plot camera poses
    for true_pose in true_pose_w_c:
        vg.plot_pose(ax, true_pose.to_tuple(), scale=1)

    # Plot initial state (to run axis equal first time).
    ax.set_title('Cost: ' + str(cost[0]))
    artists = []

    # Extract points as matrix.
    x_w = np.zeros((3, len(x[0])))
    for j, state in enumerate(x[0]):
        x_w[:, [j]] = state
    artists.extend(vg.utils.plot_as_box(ax, x_w))

    for meas in measurement:
        # Normalised in 3d.
        xn_3d = np.vstack((meas.xn, np.ones((1, meas.num))))

        artists.extend(vg.plot_camera_image_plane(ax, meas.camera.K(), meas.pose_w_c.to_tuple()))
        artists.extend(vg.utils.plot_as_box(ax, meas.pose_w_c * xn_3d, alpha=0.4))
        artists.extend(
            vg.utils.plot_as_box(ax, meas.pose_w_c * meas.camera.project_to_normalised_3d(meas.pose_c_w * x_w)))

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
        artists = []

        # Extract points as matrix.
        x_w = np.zeros((3, len(x[i])))
        for j, state in enumerate(x[i]):
            x_w[:, [j]] = state
        artists.extend(vg.utils.plot_as_box(ax, x_w))

        for meas in measurement:
            # Normalised in 3d.
            xn_3d = np.vstack((meas.xn, np.ones((1, meas.num))))

            artists.extend(vg.plot_camera_image_plane(ax, meas.camera.K(), meas.pose_w_c.to_tuple()))
            artists.extend(vg.utils.plot_as_box(ax, meas.pose_w_c * xn_3d, alpha=0.4))
            artists.extend(
                vg.utils.plot_as_box(ax, meas.pose_w_c * meas.camera.project_to_normalised_3d(meas.pose_c_w * x_w)))

        plt.draw()
        while True:
            if plt.waitforbuttonpress():
                break

    plt.close()


def visualise_full(true_pose_w_c, true_box_w, measurement, x, cost):
    # Visualize (press a key to jump to the next iteration).
    # Use Qt 5 backend in visualisation.
    matplotlib.use('qt5agg')

    # Create figure and axis.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plot true state
    for true_pose in true_pose_w_c:
        vg.plot_pose(ax, true_pose.to_tuple(), scale=1, alpha=0.4)
    vg.utils.plot_as_box(ax, true_box_w, alpha=0.4)

    num_cameras = x[0].num_poses
    num_points = x[0].num_points

    # Plot initial state (to run axis equal first time).
    ax.set_title('Cost: ' + str(cost[0]))
    artists = []

    # Extract points as matrix.
    x_w = np.zeros((3, num_points))
    for j in range(num_points):
        x_w[:, [j]] = x[0].get_point(j)
    artists.extend(vg.utils.plot_as_box(ax, x_w))

    for i in range(num_cameras):
        pose = x[0].get_pose(i)

        # Normalised in 3d.
        xn_3d = np.vstack((measurement[i].xn, np.ones((1, measurement[i].num))))

        artists.extend(vg.plot_pose(ax, pose.to_tuple(), scale=1))
        artists.extend(vg.plot_camera_image_plane(ax, measurement[i].camera.K(), pose.to_tuple()))
        artists.extend(vg.utils.plot_as_box(ax, pose * xn_3d, alpha=0.4))
        artists.extend(
            vg.utils.plot_as_box(ax, pose * measurement[i].camera.project_to_normalised_3d(pose.inverse() * x_w)))

    vg.plot.axis_equal(ax)
    plt.draw()

    while True:
        if plt.waitforbuttonpress():
            break

    # Plot iterations
    for it in range(1, len(x)):
        for artist in artists:
            artist.remove()

        ax.set_title('Cost: ' + str(cost[it]))
        artists = []

        # Extract points as matrix.
        x_w = np.zeros((3, num_points))
        for j in range(num_points):
            x_w[:, [j]] = x[it].get_point(j)
        artists.extend(vg.utils.plot_as_box(ax, x_w))

        for i in range(num_cameras):
            pose = x[it].get_pose(i)

            # Normalised in 3d.
            xn_3d = np.vstack((measurement[i].xn, np.ones((1, measurement[i].num))))

            artists.extend(vg.plot_pose(ax, pose.to_tuple(), scale=1))
            artists.extend(vg.plot_camera_image_plane(ax, measurement[i].camera.K(), pose.to_tuple()))
            artists.extend(vg.utils.plot_as_box(ax, pose * xn_3d, alpha=0.4))
            artists.extend(
                vg.utils.plot_as_box(ax, pose * measurement[i].camera.project_to_normalised_3d(pose.inverse() * x_w)))

        plt.draw()
        while True:
            if plt.waitforbuttonpress():
                break

    plt.close()
