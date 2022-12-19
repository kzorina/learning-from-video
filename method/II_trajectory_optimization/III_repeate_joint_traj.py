import numpy as np
import pickle
from pyphysx_utils.urdf_robot_parser import quat_from_euler
from pyphysx_utils.transformations import multiply_transformations
from method.utils import create_robot_env, get_reward_to_track, get_default_robot_z


def repete_joint_traj(ddp_q, tool_name, demo_params, save_pretrain_path=None, robot_name='panda', seed=None,
                      visualize=False, verbose=False, fps=24, optimize_base_rotation=False,
                      optimize_z_robot_base=False, add_steps_end=0):
    if seed is not None:
        np.random.seed(seed)
    x_base, y_base = ddp_q[-1][:2]
    base_n = 3 if optimize_base_rotation else 2

    if optimize_z_robot_base:
        q_trajectory_sampled = np.array([x[base_n + 1:] for x in ddp_q])
        base_rot = ddp_q[-1][3] if optimize_base_rotation else 0
        robot_z_pose = ddp_q[-1][2]
    else:
        base_rot = ddp_q[-1][2] if optimize_base_rotation else 0
        q_trajectory_sampled = np.array([x[base_n:] for x in ddp_q])
        robot_z_pose = get_default_robot_z(robot_name, tool_name)

    env = create_robot_env(tool_name=tool_name, robot_name=robot_name, render=visualize, rate=fps,
                           alignment_params=demo_params, add_steps_end=add_steps_end, demo_q=q_trajectory_sampled,
                           robot_init_pose=((x_base, y_base, robot_z_pose), quat_from_euler('xyz', [0., 0., base_rot])),
                           render_to_animation=True)
    env.reset()
    obs_list = []
    q_vel_list = []
    spade_pose_list = []
    real_q = []
    if verbose:
        print(f"len repeated_traj = {len(ddp_q)}")

    reward_to_track = 0
    spade_pose_list.append(env.scene.tool.get_global_pose()[0])

    for i in range(len(q_trajectory_sampled) - 1):
        obs_list.append(env.get_obs())  # X
        real_q_value = np.asarray(list(env.q.values()))
        real_q.append(real_q_value)
        q_vel = (q_trajectory_sampled[i + 1] - real_q_value) * fps
        q_vel_list.append(q_vel)
        _, r, _, _ = env.step(q_vel)
        rewards = env.scene.get_environment_rewards()
        if env.joint.is_broken():
            rewards['is_terminal'] = True
            rewards['brake_occured'] = -1
        if verbose:
            print(rewards)
        if 'is_terminal' in rewards and rewards['is_terminal']:
            if verbose:
                print('Terminal reward obtained.')
            if visualize:
                env.renderer.publish_animation()
            return 0.
        reward_to_track += sum([rewards[reward_to_track] for reward_to_track in get_reward_to_track(tool_name)])
        # record spade positions
        spade_pose_list.append(
            multiply_transformations(env.scene.tool.get_global_pose(), env.scene.tool.to_tip_transform)[0])

    if visualize:
        env.renderer.publish_animation()
    if reward_to_track > 0 and save_pretrain_path is not None:
        pickle.dump({"x": obs_list,
                     "y": q_vel_list},
                    open(save_pretrain_path, "wb"))

    return reward_to_track / (len(q_trajectory_sampled) + 1)


if __name__ == '__main__':
    robot_name = 'panda'
    tool_name = 'spade'
    video_id = 1

    alignment_path = "../../data/debug/alignment_params.pkl"
    alignment_params = pickle.load(open(alignment_path, 'rb'))
    print(alignment_params['tip_poses'][0])
    q_traj = pickle.load(open("../../data/debug/q_trajectory.pkl", "rb"))

    save_pretrain_path = "../../data/debug/pretrain_data.pkl"

    reward_to_track = repete_joint_traj(
        q_traj, tool_name, alignment_params,
        save_pretrain_path=save_pretrain_path,
        robot_name=robot_name,
        optimize_base_rotation=True,
        optimize_z_robot_base=False,
        visualize=True,
        verbose=True)
    print(reward_to_track)
