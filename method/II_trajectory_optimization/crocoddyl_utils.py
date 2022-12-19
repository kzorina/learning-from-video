import numpy as np
import pinocchio as pin
import quaternion as npq
import vizutils
import crocoddyl
import os
import pathlib


def quat_from_euler(seq='xyz', angles=None):
    """ Compute quaternion from intrinsic (e.g. 'XYZ') or extrinsic (fixed axis, e.g. 'xyz') euler angles. """
    angles = np.atleast_1d(angles)
    q = npq.one
    for s, a in zip(seq, angles):
        axis = np.array([
            1 if s.capitalize() == 'X' else 0,
            1 if s.capitalize() == 'Y' else 0,
            1 if s.capitalize() == 'Z' else 0,
        ])
        if s.isupper():
            q = q * npq.from_rotation_vector(axis * a)
        else:
            q = npq.from_rotation_vector(axis * a) * q
    return q


def produce_circle_points(z_coord=0.4, radius=0.5, n_points=10):
    return [np.array([radius * np.cos(x), radius * np.sin(x), z_coord]) for x in np.linspace(0, 2 * np.pi, n_points)]


def interpolate_color(startcolor=(1, 0, 0), goalcolor=(0, 1, 0), steps=10):
    """
    Take two RGB color sets and mix them over a specified number of steps.  Return the list
    """

    return [(startcolor[0] + (goalcolor[0] - startcolor[0]) * i / steps,
             startcolor[1] + (goalcolor[1] - startcolor[1]) * i / steps,
             startcolor[2] + (goalcolor[2] - startcolor[2]) * i / steps, 1) for i in range(steps)]


def create_spheres_for_targets(viz, targets, size=0.1, colors=None):
    if colors is None:
        colors = interpolate_color(steps=len(targets))
    for i, target in enumerate(targets):
        vizutils.addViewerSphere(viz, f'world/ball{i}', .01, colors[i])
        vizutils.applyViewerConfiguration(viz, f'world/ball{i}', list(target) + [0., 0., 0., 1.])

class ActionModelRobot2D(crocoddyl.ActionModelAbstract):
    def __init__(self, target_pose=((0.5, 0.5, 0.5), np.eye(3)), dt=0.01, base_opt=False,
                 u_weight=0.01, jpos_weight=0.001, barier_weight=1., pose_weight=1., pose_rot_scale=1.,
                 nq=7, last_link="spade_tip", base_opt_n=2,
                 q_lower=np.array([-100, -100, -2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671]),
                 q_upper=np.array([100, 100, 2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671]),
                 robot_model=None, robot_data=None, q_ref=np.zeros(7)):
        state_vector = crocoddyl.StateVector(nq)
        crocoddyl.ActionModelAbstract.__init__(
            self, state_vector, nq, 6 + 3 * nq,  # state dim, action dim, and residual dim
        )
        self.M_target = pin.SE3(np.array(target_pose[1]), np.array(target_pose[0]))
        self.base_opt = base_opt
        self.dt = dt
        self.u_weight = u_weight
        self.jpos_weight = jpos_weight
        self.pose_weight = pose_weight
        self.pose_rot_scale = pose_rot_scale
        self.barier_weight = barier_weight
        self.barrier_scale = 1.
        self.nq = nq
        self.last_link = last_link
        self.base_opt_n = base_opt_n
        self.robot_model = robot_model
        self.robot_data = robot_data
        self.q_ref = q_ref
        self.q_lower = q_lower
        self.q_upper = q_upper


    def q_barrier(self, x):
        rlb_min = np.minimum((x - self.q_lower), np.zeros(len(x)))
        rlb_max = np.maximum((x - self.q_upper), np.zeros(len(x)))
        return 0.5 * (self.barrier_scale * rlb_min) ** 2 + 0.5 * (self.barrier_scale * rlb_max) ** 2

    def q_barrier_dx(self, x):
        rlb_min = np.minimum((x - self.q_lower), np.zeros_like(x))
        rlb_max = np.maximum((x - self.q_upper), np.zeros_like(x))
        return self.barrier_scale * rlb_min + self.barrier_scale * rlb_max

    def q_barrier_dxx(self, x):
        out = np.zeros_like(x)
        out[x < self.q_lower] = self.barrier_scale
        out[x > self.q_upper] = self.barrier_scale
        return out

    def calc(self, data, x, u=None):
        """ u is acceleration """
        if u is None:
            u = self.unone
        if self.base_opt:
            u[self.base_opt_n:] = [0] * (len(u) - self.base_opt_n)
        else:
            u[:self.base_opt_n] = [0] * self.base_opt_n
        jpos = x[:self.nq]
        data.xnext = jpos + u * self.dt
        pin.forwardKinematics(self.robot_model, self.robot_data, jpos, u)
        pin.updateFramePlacements(self.robot_model, self.robot_data)
        M = self.robot_data.oMf[self.robot_model.getFrameId(self.last_link)]

        self.deltaM = self.M_target.inverse() * M
        if self.base_opt:
            data.r[:] = np.zeros(len(data.r))
        else:
            lg = pin.log(self.deltaM).vector
            data.r[:3] = self.pose_weight * lg[:3]
            data.r[3:6] = self.pose_weight * self.pose_rot_scale * lg[3:]
            data.r[6:6 + self.nq] = self.u_weight * u  # regularization, penalize large velocities
            data.r[6 + self.nq:6 + self.nq * 2 - self.base_opt_n] = self.jpos_weight * (
                        jpos[self.base_opt_n:] - self.q_ref[self.base_opt_n:])
            data.r[6 + self.nq * 2:6 + self.nq * 3] = self.barier_weight * self.q_barrier(jpos)
        data.cost = .5 * sum(data.r ** 2)
        return data.xnext, data.cost

    def calcDiff(self, data, x, u=None):
        """ Automatic numerical differentiation """
        if u is None:
            u = self.unone
        xnext, cost = self.calc(data, x, u)

        J = pin.computeFrameJacobian(self.robot_model, self.robot_data, x[:self.nq],
                                     self.robot_model.getFrameId(self.last_link))
        r = data.r[:6]
        Jlog = pin.Jlog6(self.deltaM)

        data.Lx[:self.nq] = J.T @ Jlog.T @ r + self.jpos_weight * data.r[
                    6 + self.nq:6 + self.nq * 2] + self.q_barrier_dx(x) + self.q_barrier_dxx(x)

        # data.Lx[:nq] = 2 * J.T @ Jlog.T @ data.r[:6]
        data.Lu[:] = self.u_weight * data.r[6:6 + self.nq]
        if self.base_opt:
            data.Lx[:self.nq] = np.zeros(self.nq)
            data.Lxx[:self.nq, :self.nq] = np.zeros((self.nq, self.nq))
        else:
            data.Lxx[:self.nq, :self.nq] = (Jlog @ J).T.dot((Jlog @ J))
            np.fill_diagonal(data.Luu, self.u_weight ** 2)

        # Dynamic derivatives
        np.fill_diagonal(data.Fx, 1)
        np.fill_diagonal(data.Fu, self.dt)
        if self.base_opt:
            for i in range(2, len(data.Fu)):
                data.Fu[i, i] = 0
                data.Luu[i, i] = 0
        else:
            data.Fu[0, 0] = 0
            data.Fu[1, 1] = 0
            data.Luu[0, 0] = 0
            data.Luu[1, 1] = 0

        return xnext, cost

# TODO: replacec this with generation from pyphysx_envs package (add joints if stuff move)
def get_robot_variables(robot_name, tool_name, optimize_base_rotation=True, optimize_z_robot_base=False):
    q_ref = [0., 0.]
    q_lower = [-2., -2]
    q_upper = [2, 2]
    if optimize_z_robot_base:
        q_ref += [0]
        q_lower += [-2]
        q_upper += [2]
    if optimize_base_rotation:
        q_ref += [0]
        q_lower += [-6]
        q_upper += [6]
    if robot_name == 'panda':
        model_path = os.path.join(pathlib.Path(__file__).parent.parent.parent, 'data/robots/panda')
        urdf_path = os.path.join(pathlib.Path(__file__).parent.parent.parent,
                                 f'data/robots/panda/panda_{tool_name}_opt_base_rot.urdf')
        q_ref += [0., 0., 0.2, -1.3, -0.1, 1.2, 0.]
        q_lower += [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671]
        q_upper += [2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671]

    elif robot_name == 'ur5':
        model_path = os.path.join(pathlib.Path(__file__).parent.parent.parent, 'data/robots/ur5')
        urdf_path = os.path.join(pathlib.Path(__file__).parent.parent.parent,
                                 f'data/robots/ur5/ur5_{tool_name}_opt_base_rot.urdf')
        q_ref += [0, -1.7, 1.8, -1.6, -1.6, 2.]
        q_lower += [-6.28, -6.28, -3.14, -6.28, -6.28, -6.28]
        q_upper += [6.28, 6.28, 3.14, 6.28, 6.28, 6.28]
    else:
        raise NotImplementedError(f"implement for {robot_name}")

    q_lower = np.array(q_lower)
    q_upper = np.array(q_upper)
    return model_path, urdf_path, q_ref, q_lower, q_upper