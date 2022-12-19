import sys
import os
import numpy as np
import pickle
from matplotlib import pyplot as plt
from method.utils import create_if_not_exist, get_reward_to_track, LogText, get_sorted_alignments_from_folder
from I_repeate_tool_traj import repete_tool_traj_from_alignment
from II_optimize_for_jont_values import optimize_for_poses
from III_repeate_joint_traj import repete_joint_traj
from IV_pretrain_nn import pretrain_network_and_test
import shutil
import pathlib
import time
import argparse

"""
To run:
--vis_align --vis_optim --vis_joint
"""

parser = argparse.ArgumentParser()

parser.add_argument('-str_date', type=str, default='01_01_01', help='String date for alignment folder')
parser.add_argument('-tool', type=str, default='spade', help='Tool name')
parser.add_argument('-robot', type=str, default='panda', help='Robot name')
parser.add_argument('-seed', type=int, default=111, help='Seed for optimization')
parser.add_argument('-vid_id', type=int, default=1, help='ID of the video of interest')
parser.add_argument('-fps', type=int, default=24, help='Video fps')
parser.add_argument('--vis_align', dest='visualize_alignment', action='store_true')
parser.add_argument('--vis_optim', dest='visualize_optimization', action='store_true')
parser.add_argument('--vis_joint', dest='visualize_joint_traj', action='store_true')
parser.add_argument('--vis_pretrain', dest='visualize_pretrained', action='store_true')
parser.add_argument('--verbose', dest='verbose', action='store_true')

options = parser.parse_args()
visualize_alignment = options.visualize_alignment
visualize_optimization = options.visualize_optimization
visualize_joint_traj = options.visualize_joint_traj
visualize_pretrained = options.visualize_pretrained
verbose = options.verbose
measure_time = True

double_checkalignment_n = 5
double_check_seeds = [123, 42, 77331, 321, 234]

tool_name = options.tool
new_prefix = ''
video_id = options.vid_id
robot_name = options.robot
take_last_n_points = None if tool_name == 'spade' else 20
optimize_base_rotation = True
optimize_z_robot_base = False  # if tool_name == 'spade' else True
base_rotation = 0
rate = options.fps

experiment_group_name = options.str_date

logger = LogText(
    f"{pathlib.Path(__file__).parent.parent.parent}/data/logs/{tool_name}_{video_id}_{time.strftime('%Y-%m-%d')}.txt")
logger_time = LogText(
    f"{pathlib.Path(__file__).parent.parent.parent}/data/logs/optimization_{tool_name}_{time.strftime('%Y-%m-%d')}_time.txt",
    "w")

root_alignment_folder = f"{pathlib.Path(__file__).parent.parent.parent}/data/alignment/{experiment_group_name}"
save_dir_path_fin = f"{pathlib.Path(__file__).parent.parent.parent}/data/alignment/{experiment_group_name}/{tool_name}/video_{video_id}/{robot_name}"
snapshot_folder = f"{pathlib.Path(__file__).parent.parent.parent}/data/logs/{experiment_group_name}/{tool_name}/video_{video_id}/{robot_name}"

create_if_not_exist(save_dir_path_fin, remove=True)
create_if_not_exist(snapshot_folder, remove=False)

alignment_folder = os.path.join(root_alignment_folder, f'{new_prefix}{tool_name}/video_{video_id}/')
save_fin_dict_path = os.path.join(save_dir_path_fin, f"fin_dict_{new_prefix}{tool_name}_{video_id}.pkl")

# Load and sort alignments
sorted_alignment_filenames = get_sorted_alignments_from_folder(alignment_folder, verbose=verbose)
if len(sorted_alignment_filenames) == 0:
    print(f"Folder {alignment_folder} contains 0 alignments")

# if verbose:
#     print(sorted_alignment_filenames)
print(f"Processing {len(sorted_alignment_filenames)} alignments")
print(f"best alignment is :{sorted_alignment_filenames[0]}")

logger.log(
    f"===============================================================\n" +
    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}\n" +
    f"===============================================================\n" +
    f"Processing {len(sorted_alignment_filenames)} alignments from {os.path.dirname(sorted_alignment_filenames[0])}\n")

## repeate for each alignment
# Repeate traj N times, observe if reward > 0. If visualize_alignment=True, visualize
if measure_time:
    start_iter_time = time.time()
counter = 0
count_found_alignments = 0
for alignment_filename in sorted_alignment_filenames:

    counter += 1
    save_pretrain_path = os.path.join(save_dir_path_fin,
                                      f"{count_found_alignments}_pretrain_robot_network_{new_prefix}{tool_name}_{video_id}.pkl")
    if verbose:
        logger.log(
            f"\n--------------------------------------------------------\n" +
            f"Processing {os.path.basename(alignment_filename)}\n" +
            f"Results repeating tool-only trajectory: \n")
    alignment_params = pickle.load(open(alignment_filename, "rb"))
    success_tool_repeat = 0
    # if alignment gets reward at least onve in N times - continue
    for i, seed in enumerate(double_check_seeds):
        total_reward, new_alignment_params = repete_tool_traj_from_alignment(
            alignment_params.copy(),
            visualize=visualize_alignment if i == 1 else False,
            tool_name=tool_name,
            seed=seed,
            verbose=verbose,
            stop_on_positive_reward=False if tool_name == 'spade' else True,
            return_last_step_id=False if tool_name == 'spade' else True,
            rate=rate)
        logger.log(f"Total reward {total_reward} for {seed} seed\n")
        if total_reward > 0:
            if not success_tool_repeat:
                alignment_params["tip_poses"] = new_alignment_params['tip_poses'].copy()
            success_tool_repeat += 1
    if success_tool_repeat:
        assert len(alignment_params['tip_poses']) != 0
        # Run optimization, get joint traj, plot tool traj
        ddp_q = optimize_for_poses(alignment_params['tip_poses'], tool_name, robot_name=robot_name,
                                   seed=double_check_seeds[0], last_n_points=take_last_n_points,
                                   optimize_base_rotation=optimize_base_rotation, verbose=verbose,
                                   visualize=visualize_optimization)
        # Repeate joint traj in env, observe reward
        success_in_joint_repeate = 0
        logger.log(f"---- Results after following optimized robot trajectory\n")
        for i, seed in enumerate(double_check_seeds):
            reward = repete_joint_traj(ddp_q, tool_name, alignment_params, seed=seed,
                                       save_pretrain_path=save_pretrain_path,
                                       robot_name=robot_name, optimize_base_rotation=optimize_base_rotation,
                                       optimize_z_robot_base=optimize_z_robot_base,
                                       visualize=visualize_joint_traj if i == 0 else False,
                                       verbose=verbose)
            logger.log(f"---- Total reward {reward} for {seed} seed\n")
            if reward > 0:
                success_in_joint_repeate += 1
        if success_in_joint_repeate:
            save_alignment_path = os.path.join(save_dir_path_fin,
                                               f"{count_found_alignments}_align_{new_prefix}{tool_name}_{video_id}.pkl")
            save_pretrained_model_path = os.path.join(save_dir_path_fin,
                                                      f"{count_found_alignments}_pretrained_mu_{robot_name}_{new_prefix}{tool_name}_{video_id}.pkl")
            save_q_traj_path = os.path.join(save_dir_path_fin,
                                            f"{count_found_alignments}_q_traj_{new_prefix}{tool_name}_{video_id}.pkl")
            data_to_pretrain = pickle.load(open(save_pretrain_path, "rb"))
            after_pretrain_reward = []
            success_of_pretrained_policy = 0
            logger.log(f"--------- After policy pretrain: \n")
            for i, seed in enumerate(double_check_seeds):
                reward_list = pretrain_network_and_test(robot_name,
                                                        tool_name,
                                                        seed=seed,
                                                        ddp_q=ddp_q,
                                                        demo_params=alignment_params,
                                                        data_to_pretrain=data_to_pretrain,
                                                        save_pretrained_model_path=save_pretrained_model_path,
                                                        save_alignment_path=save_alignment_path,
                                                        save_q_traj_path=save_q_traj_path,
                                                        use_pretrained=success_of_pretrained_policy,
                                                        optimize_z_robot_base=optimize_z_robot_base,
                                                        logger=logger,
                                                        save_sim_snapshots=i == 0,
                                                        snapshot_folder=snapshot_folder,
                                                        visualize=visualize_pretrained,
                                                        )
                reward = np.sum(reward_list)
                after_pretrain_reward.append(reward)
                if verbose:
                    logger.log(f"--------- id_try = {count_found_alignments}, reward : {reward} for {seed} seed \n")
                if reward > 0:
                    success_of_pretrained_policy += 1
            if success_of_pretrained_policy >= 1:
                count_found_alignments += 1
                if verbose:
                    logger.log(f"-------------- After policy pretrain, success : {success_of_pretrained_policy} times\n")
    if measure_time:
        print(f"{time.time() - start_iter_time:.2f} sec for {counter} iter")
        logger_time.log(f"{time.time() - start_iter_time:.2f} sec for {counter} iter")
        start_iter_time = time.time()

if count_found_alignments == 0:
    print("DID NOT FOUND")
    logger.log(f"DID NOT FOUND good alignment\n")
else:
    print(f'found {count_found_alignments} alignments')
    logger.log(f"found {count_found_alignments} alignments\n")
