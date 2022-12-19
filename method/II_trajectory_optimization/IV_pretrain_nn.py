import torch
from rlpyt.models.mlp import MlpModel
from pyphysx_envs.envs import ToolEnv, RobotEnv
import numpy as np
from os import path
import pickle
from matplotlib import pyplot as plt
from pyphysx_utils.urdf_robot_parser import quat_from_euler
from pyphysx_utils.transformations import multiply_transformations
from method.utils import create_robot_env, get_reward_to_track, create_if_not_exist, get_default_robot_z, LogText
# meshcat                   0.0.19                   pypi_0    pypi


def pretrain_network_and_test(robot_name='panda', tool_name='spade', seed=None,
                              ddp_q=None, demo_params=None, data_to_pretrain=None,
                              save_pretrained_model_path=None, rate=24,
                              save_alignment_path=None, snapshot_folder=None,
                              adr_append_to_x=None, cut_adr_until_n=2,
                              save_q_traj_path=None, optimize_z_robot_base=False,
                              pretrain_n_steps=6000, visualize=False,
                              optimize_base_rotation=True, add_steps_end=0,
                              use_pretrained=False, logger=None, save_sim_snapshots=False):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(0)
    if adr_append_to_x is None:
        x_train = data_to_pretrain['x']
    else:
        x_train_prep = data_to_pretrain['x']
        x_train = []
        for row in x_train_prep:
            row_to_append = row
            for param_name in adr_append_to_x:
                row_to_append = np.append(row_to_append, demo_params[param_name][:cut_adr_until_n])
            x_train.append(row_to_append)
    y_train = data_to_pretrain['y'][:len(x_train)]

    input_size = len(x_train[0])
    output_size = len(y_train[0])
    policy_hidden_sizes = [400, 300]
    policy_hidden_nonlinearity = torch.nn.Tanh
    lr = 0.0001
    model = MlpModel(input_size=input_size, hidden_sizes=policy_hidden_sizes, output_size=output_size,
                     nonlinearity=policy_hidden_nonlinearity)

    if not use_pretrained:
        x = torch.Tensor(x_train)
        y = torch.Tensor(y_train)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, )
        loss_values = []
        for epoch in range(pretrain_n_steps):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss_values.append(loss)
            loss.backward()
            if epoch % 100 == 99:
                print(epoch, loss.item())
            optimizer.step()
        # logger.log("achieved loss for {} len q, at {} steps: {}".format(len(x_train), pretrain_n_steps, loss))
        # plt.plot(loss_values)
        # plt.show()
        torch.save(model.state_dict(), save_pretrained_model_path)
    else:
        logger.log("Use previously pretrained model")
        model.load_state_dict(torch.load(save_pretrained_model_path))

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

    env = create_robot_env(tool_name=tool_name, robot_name=robot_name, render=visualize, rate=rate,
                           alignment_params=demo_params, add_steps_end=add_steps_end, demo_q=q_trajectory_sampled,
                           robot_init_pose=((x_base, y_base, robot_z_pose), quat_from_euler('xyz', [0., 0., base_rot])),
                           render_to_animation=not save_sim_snapshots)

    env.robot.set_init_q(q_trajectory_sampled[0])
    env.reset()


    rewards_to_track_name = get_reward_to_track(tool_name)
    reward_list = []

    rewards = env.scene.get_environment_rewards()
    reward_list.append(sum([rewards[name] for name in rewards_to_track_name]))
    spade_pose_list = []
    for i in range(len(q_trajectory_sampled) - 1 + add_steps_end):
        action = model(torch.tensor(env.get_obs()))
        o, r, done, info = env.step(action.detach().numpy())
        spade_pose_list.append(env.scene.tool.get_global_pose()[0])
        rewards = env.scene.get_environment_rewards()
        reward_list.append(sum([rewards[name] for name in rewards_to_track_name]))
        if visualize and save_sim_snapshots:
            img = env.renderer.vis.get_image()
            plt.imshow(img)
            plt.savefig(path.join(snapshot_folder, f"{i:06d}.png"))

        if done:
            break

    if not use_pretrained and np.sum(reward_list) > 0:
        logger.log(f"\n Save alignment to {save_alignment_path} \n")
        logger.log(f"\n Save q trajectory to {save_q_traj_path} \n")
        logger.log(f"\n Save model to {save_pretrained_model_path} \n")
        pickle.dump(ddp_q, open(save_q_traj_path, 'wb'))
        pickle.dump(demo_params, open(save_alignment_path, 'wb'))
        torch.save(model.state_dict(), save_pretrained_model_path)
    if visualize and not save_sim_snapshots:
        env.renderer.publish_animation()
    return reward_list


if __name__ == '__main__':
    robot_name = 'panda'
    tool_name = 'spade'
    prefix = ''
    video_id = 1
    id_try = 2

    folder = f"/home/kzorina/Work/git_repos/learning-from-video-debug/data/alignment/21_12_13/{tool_name}/video_{video_id}/panda"
    # snapshot_folder = f"/home/kzorina/Work/git_repos/learning-from-video/data/alignment/01_01_01/{tool_name}/video_{video_id}/panda/{id_try}_pretrain_img"
    # create_if_not_exist(snapshot_folder, remove=False)
    data_to_pretrain = pickle.load(
        open(path.join(folder, f'{id_try}_pretrain_robot_network_{tool_name}_{video_id}.pkl'), 'rb'))
    demo_params = pickle.load(open(path.join(folder, f'{id_try}_align_{tool_name}_{video_id}.pkl'), 'rb'))
    # demo_params = pickle.load(open(path.join(folder, f'{id_try}_align_{tool_name}.pkl'), 'rb'))
    ddp_q = pickle.load(open(path.join(folder, f'{id_try}_q_traj_{tool_name}_{video_id}.pkl'), 'rb'))
    # ddp_q = pickle.load(open(path.join(folder, f'{id_try}_q_traj_{tool_name}.pkl'), 'rb'))
    logger = LogText(f"/home/kzorina/Work/git_repos/learning-from-video-debug/data/logs/IV_pretrain_nn.txt")
    reward_list = pretrain_network_and_test(robot_name=robot_name,
                                            tool_name=tool_name,
                                            ddp_q=ddp_q,
                                            demo_params=demo_params,
                                            data_to_pretrain=data_to_pretrain,
                                            save_alignment_path=path.join(folder, f'align_{tool_name}_{video_id}.pkl'),
                                            save_q_traj_path=path.join(folder, f'q_traj_{tool_name}_{video_id}.pkl'),
                                            save_pretrained_model_path=path.join(folder,
                                                                                 f'{id_try}_pretrained_mu_panda_{prefix}{tool_name}_{video_id}.pkl'),
                                            snapshot_folder=None,
                                            use_pretrained=True,
                                            visualize=True,
                                            optimize_z_robot_base=False,
                                            logger=logger,
                                            save_sim_snapshots=False
                                            # adr_append_to_x=['grass_patch_location_0']
                                            )
    print(reward_list)
