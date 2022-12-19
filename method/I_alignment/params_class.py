import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
import quaternion as npq
import vg


class ParamsSampler:
    def __init__(self, env, head, handle, seed):
        self.seed = seed
        np.random.seed(seed)
        self.scale_list = [0.5, 0.75, 1.]
        self.head = head
        self.handle = handle
        self.env = env
        self.ind_keypoints = self.get_keypoints_indices_velocity_peaks(head, q=0.95)
        self.sigma = 0.1
        self.min_ofset = 0
        self.max_ofset = 0.5
        self.base_rotation = -np.math.pi / 2

    def sample_params(self):
        params = {}
        params['trajectory_scale'] = np.random.choice(self.scale_list)
        head = self.head * params['trajectory_scale']
        handle = self.handle * params['trajectory_scale']
        # (i) sample scene object placement + check if not violating min distance
        prev_elements = []
        scene_element_sampled_ids = list(np.random.choice(list(self.ind_keypoints), replace=False,
                                                     size=len(self.env.scene.scene_object_params)))
        for i, scene_element in enumerate(self.env.scene.scene_object_params):
            params[scene_element] = self.gmm_sample(head[scene_element_sampled_ids[i]], self.sigma ** 2)
            params[scene_element][2] = 0
            if len(prev_elements) > 0:
                for element in prev_elements:
                    if np.sqrt(sum(
                            (params[scene_element] - element) ** 2)) < self.env.scene.min_dist_between_scene_objects:
                        return self.sample_params()
            prev_elements.append(params[scene_element])
        # (ii) sample tool tip trajectory
        offset_pose = np.array([0., 0., np.random.normal(self.min_ofset, self.max_ofset)])

        tool_rotation = self.sample_rotations_from_keypoints_index(self.ind_keypoints, scene_element_sampled_ids, len(head))
        # tool_rotation_x = sample_rotations_from_keypoints_index(ind_keypoints, scene_element_sampled_ids, len(head))
        tool_rotation_x = 0
        # base_tool_rotation_x = -0.5

        params['tip_poses'] = self.get_poses_from_points(head + offset_pose, handle + offset_pose, tool_rotation,
                                                         base_rotation=self.base_rotation,
                                                         # rot_x=tool_rotation_x,
                                                         # base_x_rotation=base_tool_rotation_x,
                                                         # base_xUpd_rotation=-0.7
                                                       )
        # params['tip_poses'] = check_for_z_turn_consistency(params['tip_poses'])
        # params['tip_poses'] = get_poses_from_points(head + offset_pose, handle + offset_pose, 1.57)
        if self.env.tool_name == 'spade':
            params['sand_buffer_yaw'] = np.random.uniform(0, 2 * np.math.pi, size=1)[0] - np.math.pi

        return params

    def get_keypoints_indices_velocity_peaks(self, head, q=0.9, include_start=True, include_end=True):
        """ Find peaks with values larger than q-th quartile of the velocity.
            Repeat for negative signal to have peaks for both the smallest as well largest velocities. """

        head_vel = head[1:] - head[:-1]
        d_head_vel = np.linalg.norm(head_vel, axis=1)

        ind1, _ = find_peaks(d_head_vel, np.quantile(d_head_vel, q))
        ind2, _ = find_peaks(-d_head_vel, np.quantile(-d_head_vel, q))
        ind = np.concatenate([ind1, ind2])
        if include_start and not np.isin(0, ind):
            ind = np.append(ind, 0)
        if include_end:  # note that end is never in ind due to the velocity computation
            ind = np.append(ind, head.shape[0] - 1)
        return np.sort(ind)

    @staticmethod
    def gmm_sample(mean, sigma_squared, zero_z_axis=False):
        out = np.random.multivariate_normal(mean, sigma_squared * np.eye(len(mean)))
        if zero_z_axis:
            out[2] = 0.
        return out

    @staticmethod
    def sample_rotations_from_keypoints_index(ind, ind_scene_obj, n):
        rot = np.zeros(n)
        angles = np.random.uniform(0, np.math.pi, size=len(ind) + 2) - np.math.pi / 2
        scene_obj_angles = np.random.uniform(0, 2 * np.math.pi, size=len(ind_scene_obj)) - np.math.pi
        j = 0
        rot[:ind[0]] = angles[-2]
        for i in range(1, len(ind)):
            if j < len(scene_obj_angles) and ind[i - 1] == ind_scene_obj[j]:
                rot[ind[i - 1]:ind[i]] = scene_obj_angles[j]
                j += 1
            else:
                rot[ind[i - 1]:ind[i]] = angles[i - 1]

        if j < len(scene_obj_angles) and ind[-1] == ind_scene_obj[j]:
            rot[ind[-1]:] = scene_obj_angles[j]
        else:
            rot[ind[-1]:] = angles[-1]
        return rot

    def get_poses_from_points(self, head, handle, rot=0., rot_x=0., base_rotation=0., base_x_rotation=0.):
        poses = []
        prev_x = np.array([1, 0, 0])
        symmetry_transformation = npq.from_float_array(
            self.env.scene.symmetry_quat[np.random.choice(len(self.env.scene.symmetry_quat))])
        for i in range(head.shape[0]):
            pos, quat, prev_x = self.get_pose_from_points(head[i], handle[i],
                                                          base_rotation + rot if np.isscalar(rot) else base_rotation +
                                                                                                       rot[i],
                                                          base_x_rotation + rot_x if np.isscalar(
                                                              rot_x) else base_x_rotation + rot_x[i], prev_x)
            poses.append((pos, symmetry_transformation * quat))
        return poses

    def get_pose_from_points(self, head, handle, rot=0, rot_x=0, prev_x=0):
        quat, prev_x = self.get_quat_from_points(head, handle, rot, add_rot_x=rot_x, prev_x=prev_x)
        # quat, prev_x = get_quat_from_points_v2(head, handle, rot, add_rot_x=rot_x, prev_x=prev_x)
        return head, quat, prev_x

    def get_quat_from_points(self, head, handle, rot=0, add_rot_x=0, prev_x=0):
        z_axis = vg.normalize(head - handle)
        x_axis = self.get_perpendicular_vector_3d(z_axis, prev_x)
        y_axis = vg.normalize(vg.cross(z_axis, x_axis))
        dcm = np.vstack([x_axis, y_axis, z_axis]).T
        # rot = Rotation.from_dcm(dcm) * Rotation.from_euler('z', rot)
        rot = Rotation.from_matrix(dcm) * Rotation.from_euler('z', rot) * Rotation.from_euler('x', add_rot_x)
        # rot = Rotation.from_matrix(dcm) * Rotation.from_euler('z', rot) * Rotation.from_euler('x', add_rot_x) * Rotation.from_euler('y', add_rot_y)
        return npq.from_rotation_matrix(rot.as_matrix()), x_axis

    def get_perpendicular_vector_3d(self, v, prev_x, threshold=0.1):
        t1 = np.array([v[2], 0., -v[0]])
        t2 = np.array([-v[2], 0., v[0]])
        if np.linalg.norm((prev_x - t1)) < np.linalg.norm((prev_x - t2)):
            t = t1
        else:
            t = t2
        if np.all(np.isclose(t, np.zeros_like(t))):
            t = prev_x
            # t = np.array([v[2], v[2], -v[0] - v[1]])
        return vg.normalize(t)