import numpy as np
from crocoddyl_utils import *
import crocoddyl
from matplotlib import pyplot as plt
import os
from pinocchio.visualize import MeshcatVisualizer
from pinocchio.robot_wrapper import RobotWrapper
import pickle
import meshcat

class WrapperMeshcatVisualizer(MeshcatVisualizer):
    def display(self, q, frame=None):
        """Display the robot at configuration q in the viewer by placing all the bodies."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateGeometryPlacements(self.model, self.data, self.visual_model, self.visual_data)
        for visual in self.visual_model.geometryObjects:
            # Get mesh pose.
            M = self.visual_data.oMg[self.visual_model.getGeometryId(visual.name)]
            # Manage scaling
            scale = np.asarray(visual.meshScale).flatten()
            S = np.diag(np.concatenate((scale, [1.0])))
            T = np.array(M.homogeneous).dot(S)
            # Update viewer configuration.
            if frame is None:
                self.viewer[self.getViewerNodeName(visual, pin.GeometryType.VISUAL)].set_transform(T)
            else:
                frame[self.getViewerNodeName(visual, pin.GeometryType.VISUAL)].set_transform(T)

def optimize_for_poses(poses, tool_name, robot_name='panda', seed=None, verbose=False, visualize=False,
                       last_n_points=None, optimize_base_rotation=False, repeate_q_traj=None):
    if seed is not None:
        pin.seed(seed)
    target_poses = []
    t0 = np.eye(4)
    if last_n_points is None:
        last_n_points = len(poses)
    for i, el in enumerate(poses[-last_n_points:]):
        t0[:3, :3] = np.array(npq.as_rotation_matrix(el[1]))
        t0[:3, 3] = el[0]
        t = t0.copy()
        target_poses.append([t[:3, 3], t[:3, :3]])
    model_path, urdf_path, q_ref, q_lower, q_upper = get_robot_variables(robot_name, tool_name)
    robot = RobotWrapper.BuildFromURDF(urdf_path, model_path)
    robot_model = robot.model
    robot_data = robot_model.createData()
    q0 = pin.randomConfiguration(robot_model)
    nq = len(q0)
    if visualize:
        # viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
        viz = WrapperMeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
        import inspect
        print("MESCAT PATH")
        print(inspect.getfile(viz.__class__))
        vis = meshcat.Visualizer()

        vis.delete()
        # vis.url("http://127.0.0.1:7001/static/")
        viz.initViewer(vis, open=True, )
        animation = meshcat.animation.Animation(default_framerate=30)
        viz.loadViewerModel()
        viz.display(np.array(q_ref))
        create_spheres_for_targets(viz, targets=[pose[0] for pose in target_poses])

    last_link_set = f'{tool_name}_tip'
    diff = 'num'
    # diff = 'anal'
    horizon = 10
    base_opt_steps = 10
    dt = 0.01
    if tool_name == 'spade':
        u_weight = 0.3
        jpos_weight = 0.01
        pose_weight = 0.8
        pose_rot_scale = 0.5
        # worked for UR5
        # u_weight = 0.3
        # jpos_weight = 0.01
        # pose_weight = 0.8
        # pose_rot_scale = 0.5
        # u_weight = 0.3
        # jpos_weight = 0.01
        # pose_weight = 0.8
        # pose_rot_scale = 0.5
    elif tool_name == 'hammer':
        # 07.06.21 config for 1,3,4,5
        # u_weight = 0.1
        # jpos_weight = 0.0001
        # pose_weight = 2
        # pose_rot_scale = 0.1
        # change
        u_weight = 0.1
        jpos_weight = 0.
        pose_weight = 0.8
        pose_rot_scale = 0.5
    elif tool_name == 'scythe':
        u_weight = 0.1
        jpos_weight = 0.
        pose_weight = 1.
        pose_rot_scale = 0.5
    else:
        raise ValueError(f'Unknown tool {tool_name}')

    if repeate_q_traj is None:
        action_models_list = []
        base_opt_action_models_list = []

        kwargs_action_model = dict(dt=dt, jpos_weight=jpos_weight, u_weight=u_weight, last_link=last_link_set, nq=nq,
                                   robot_model=robot_model, robot_data=robot_data, q_ref=q_ref,
                                   pose_rot_scale=pose_rot_scale,
                                   barier_weight=0.5, pose_weight=pose_weight,
                                   q_lower=q_lower, q_upper=q_upper, base_opt_n=3 if optimize_base_rotation else 2,
                                   )
        if diff == 'num':
            base_opt_action_model_t = ActionModelRobot2D(base_opt=True, **kwargs_action_model)
            base_opt_action_model = crocoddyl.ActionModelNumDiff(base_opt_action_model_t)
        else:
            base_opt_action_model = ActionModelRobot2D(base_opt=True, **kwargs_action_model)
        for i, target_pose in enumerate(target_poses):
            if diff == 'num':
                action_model_t = ActionModelRobot2D(target_pose=target_pose, **kwargs_action_model)
                action_model = crocoddyl.ActionModelNumDiff(action_model_t)
            else:
                action_model = ActionModelRobot2D(target_pose=target_pose, **kwargs_action_model)
            action_models_list.append(action_model)

        running_problems = [base_opt_action_model] * base_opt_steps
        # running_problems = []
        for i, a in enumerate(action_models_list):
            # running_problems += [a] * horizon * (10 if i == 0 else 1)
            running_problems += [a] * horizon
        terminal_problem = running_problems[-1]

        x0 = np.array(q_ref).copy()
        problem = crocoddyl.ShootingProblem(x0, running_problems, terminal_problem, )
        ddp = crocoddyl.SolverDDP(problem, )  # TODO: use Feasible DDP (start from an initial guess, precompute q with IK)
        if verbose:
            ddp.setCallbacks([crocoddyl.CallbackVerbose()])
        done = ddp.solve(init_xs=[x0] * (len(running_problems) + 1))
        if verbose:
            print(f'Converged: {done}')
        real_points = len(target_poses)
        base_opt_steps = base_opt_steps
        ids_to_take = [base_opt_steps + i * horizon for i in range(real_points)]
        ddp_q = np.array(ddp.xs)[ids_to_take].copy()
        print(len(ddp_q))
    else:
        ddp_q = repeate_q_traj
    if verbose:
        spade_robot_pose_list = []
        for i in range(len(ddp_q)):
            pin.forwardKinematics(robot_model, robot_data, ddp_q[i][:nq])
            pin.updateFramePlacements(robot_model, robot_data)
            M = robot_data.oMf[robot_model.getFrameId(last_link_set)]
            spade_robot_pose_list.append(M.translation.copy())
            if visualize:
                # with animation.at_frame(vis, i - base_opt_steps) as frame:
                with animation.at_frame(vis, i) as frame:
                    viz.display(ddp_q[i][:nq].copy(), frame)

    if visualize:
        vis.set_animation(animation, play=True, repetitions=1)
    return ddp_q


if __name__ == '__main__':
    robot_name = 'panda'
    tool_name = 'spade'
    video_id = 1
    # save_alignment_folder = '/home/kzorina/Work/learning_from_video/data/save_alignment_from_22_02'
    # save_alignment_folder = '/home/kzorina/Work/learning_from_video/data/alignment/save_from_04_03_21'

    alignment_path = "../../data/debug/alignment_params.pkl"

    alignment_params = pickle.load(open(alignment_path, 'rb'))
    print(alignment_params['tip_poses'][0])
    q_trajectory_test = pickle.load(open("../../data/debug/q_trajectory.pkl", "rb"))
    # q_trajectory_test = None
    q_trajectory = optimize_for_poses(alignment_params['tip_poses'], tool_name, robot_name=robot_name, verbose=True,
                                      visualize=True,
                                      optimize_base_rotation=True,
                                      optimize_z_robot_base=False,
                                      repeate_q_traj=q_trajectory_test,
                                      # last_n_points=15,
                                      )
    if q_trajectory_test is None:
        pickle.dump(q_trajectory, open("../../data/debug/q_trajectory.pkl", "wb"))