import numpy as np
from pylie import SO3, SE3


class PerspectiveCamera:
    """Camera model for the perspective camera"""

    def __init__(self, f: np.ndarray, c: np.ndarray):
        self.f = f
        self.c = c

    @staticmethod
    def looks_at_pose(camera_pos_w: np.ndarray, target_pos_w: np.ndarray, up_vector_w: np.ndarray):
        cam_to_target_w = target_pos_w - camera_pos_w
        cam_z_w = cam_to_target_w.flatten() / np.linalg.norm(cam_to_target_w)

        cam_to_right_w = np.cross(-up_vector_w.flatten(), cam_z_w)
        cam_x_w = cam_to_right_w / np.linalg.norm(cam_to_target_w)

        cam_y_w = np.cross(cam_z_w, cam_x_w)

        return SE3((SO3(np.vstack((cam_x_w, cam_y_w, cam_z_w)).T), camera_pos_w))

    @staticmethod
    def project_to_normalised_3d(x_c: np.ndarray):
        return x_c / x_c[-1]

    @classmethod
    def project_to_normalised(cls, x_c: np.ndarray):
        xn = cls.project_to_normalised_3d(x_c)
        return xn[:2]

    def K(self):
        K = np.identity(3)
        np.fill_diagonal(K[:2, :2], self.f)
        K[:2, [2]] = self.c
        return K

    def project_to_pixel(self, x_c: np.ndarray):
        return (self.project_to_normalised(x_c) * self.f) + self.c

    def pixel_to_normalised(self, u: np.ndarray):
        return (u - self.c) / self.f

    def pixel_cov_to_normalised_com(self, pixel_cov : np.ndarray):
        S_f = np.diag(1/self.f.flatten())
        return S_f @ pixel_cov @ S_f.T

    @classmethod
    def reprojection_error_normalised(cls, x_c: np.ndarray, measured_x_n: np.ndarray):
        return measured_x_n[:2] - cls.project_to_normalised(x_c)

    def reprojection_error_pixel(self, x_c: np.ndarray, measured_u: np.ndarray):
        return measured_u - self.project_to_pixel(x_c)

    @staticmethod
    def jac_project_normalised_wrt_x_c(x_c: np.ndarray):
        x_c = x_c.flatten()
        d = 1 / x_c[-1]
        xn = d * x_c

        return np.array([[d, 0, -d * xn[0]],
                         [0, d, -d * xn[1]]])

    @staticmethod
    def jac_project_world_to_normalised_wrt_pose_w_c(pose_c_w: SE3, x_w: np.ndarray):
        x_c = (pose_c_w * x_w).flatten()

        d = 1 / x_c[-1]
        xn = d * x_c

        # Corresponds to PerspectiveCamera.jac_project_normalised_wrt_x_c(pose_c_w * x_w) @ \
        #                pose_c_w.jac_action_Xx_wrt_X(x_w) @ pose_c_w.inverse().jac_inverse_X_wrt_X()
        return np.array([[-d, 0, d * xn[0], xn[0] * xn[1],  -1 - xn[0] ** 2,  xn[1]],
                         [0, -d, d * xn[1], 1 + xn[1] ** 2, -xn[0] * xn[1], -xn[0]]])

    @classmethod
    def jac_project_world_to_normalised_wrt_x_w(cls, pose_c_w: SE3, x_w: np.ndarray):
        return cls.jac_project_normalised_wrt_x_c(pose_c_w * x_w) @ pose_c_w.jac_action_Xx_wrt_x()
