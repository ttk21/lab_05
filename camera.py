import numpy as np
from pylie import SO3, SE3


class PerspectiveCamera:
    """Camera model for the perspective camera"""

    def __init__(self, K: np.ndarray, pose_w_c: SE3):
        self.K = K
        self.pose_w_c = pose_w_c
        self.pose_c_w = pose_w_c.inverse()

    @classmethod
    def looks_at(cls, K: np.ndarray, camera_pos_w: np.ndarray, target_pos_w: np.ndarray, up_vector_w: np.ndarray):
        cam_to_target_w = target_pos_w - camera_pos_w
        cam_z_w = cam_to_target_w.flatten() / np.linalg.norm(cam_to_target_w)

        cam_to_right_w = np.cross(-up_vector_w.flatten(), cam_z_w)
        cam_x_w = cam_to_right_w / np.linalg.norm(cam_to_target_w)

        cam_y_w = np.cross(cam_z_w, cam_x_w)

        pose_w_c = SE3((SO3(np.vstack((cam_x_w, cam_y_w, cam_z_w)).T), camera_pos_w))

        return cls(K, pose_w_c)

    def camera_to_world(self, x_c: np.ndarray):
        return self.pose_w_c * x_c

    def world_to_camera(self, x_w: np.ndarray):
        return self.pose_c_w * x_w

    def project_to_normalised_3d(self, x_w: np.ndarray):
        x_c = self.world_to_camera(x_w)
        return x_c / x_c[-1]

    def project_to_normalised(self, x_w: np.ndarray):
        x_n = self.project_to_normalised_3d(x_w)
        return x_n[:2]

    def project_to_pixel(self, x_w: np.ndarray):
        u = self.K * self.project_to_normalised_3d(x_w)
        return u[:2]

    def reprojection_error_normalised(self, x_w: np.ndarray, measured_x_n: np.ndarray):
        return measured_x_n[:2] - self.project_to_normalised(x_w)

    def reprojection_error_pixel(self, x_w: np.ndarray, measured_u: np.ndarray):
        return measured_u - self.project_to_pixel(x_w)

    def jac_project_normalised_wrt_x_c(self, x_c: np.ndarray):
        x_c = x_c.flatten()
        d = 1 / x_c[-1]
        x_n = d * x_c

        return np.array([[d, 0, -d * x_n[0]],
                         [0, d, -d * x_n[1]]])

    def jac_project_normalised_wrt_pose_w_c(self, x_w: np.ndarray):
        x_c = self.world_to_camera(x_w).flatten()

        d = 1 / x_c[-1]
        x_n = d * x_c

        # Corresponds to self.jac_project_normalised_wrt_x_c(self.world_to_camera(x_w)) @ \
        #                self.pose_c_w.jac_action_Xx_wrt_X(x_w) @ pose.w_c.jac_inverse_X_wrt_X()
        return np.array([[-d, 0, d * x_n[0], x_n[0] * x_n[1],  -1 - x_n[0] ** 2,  x_n[1]],
                         [0, -d, d * x_n[1], 1 + x_n[1] ** 2, -x_n[0] * x_n[1], -x_n[0]]])

    def jac_project_normalised_wrt_x_w(self, x_w: np.ndarray):
        return self.jac_project_normalised_wrt_x_c(self.world_to_camera(x_w)) @ self.pose_c_w.jac_action_Xx_wrt_x()
