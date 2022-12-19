from utils import save_results
from method.II_trajectory_optimization.III_repeate_joint_traj import repete_joint_traj
from method.II_trajectory_optimization.IV_pretrain_nn import pretrain_network_and_test
from method.utils import LogText, create_if_not_exist
import pickle
import pathlib
import shutil
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-str_date', type=str, default='01_01_01', help='String date for alignment folder')
parser.add_argument('-robot', type=str, default='panda', help='Robot name')
options = parser.parse_args()

results_file = f'../data/results_wks_{options.str_date}.csv'
logger = LogText(f"../data/logs/IV_pretrain_nn.txt")
# align_name_base = '../data/pretrain_components/alignment/align_{}_{}.pkl'
# q_name_base = '../data/pretrain_components/init_q_trajectory/q_traj_{}_{}.pkl'
# save_pretrain_path = '../data/pretrain_component/temp_pretrain_path.pkl'
tools_list = ['spade', 'scythe', 'hammer']
# tools_list = ['hammer']
robot_name = 'panda'
# seed = 123
# seeds = [123, 42, 1888, 666, 31121995]
seeds = [123, 42, 1888]
extended_cols = True
save_best = True

result_list = []
root_alignment_folder = f"{pathlib.Path(__file__).parent.parent}/data/alignment/{options.str_date}"
for tool_name in tools_list:
    optimize_z_robot_base = False # if tool_name == 'spade' else True
    for video_id in range(1, 6):
    # for video_id in [1]:
        video_alignment_folder = pathlib.PurePath(root_alignment_folder, tool_name, f'video_{video_id}', options.robot)
        if pathlib.Path(video_alignment_folder).exists():
            count_alignments = len(
                [path for path in pathlib.Path(video_alignment_folder).iterdir()]) // 4
            best_score = 0
            best_score_k = -1
            for k in range(count_alignments):
                alignment_params = pickle.load(open(pathlib.PurePath(video_alignment_folder,
                                                                f"{k + 1}_align_{tool_name}_{video_id}.pkl"), 'rb'))
                ddp_q = pickle.load(open(pathlib.PurePath(video_alignment_folder,
                                                          f"{k + 1}_q_traj_{tool_name}_{video_id}.pkl"), 'rb'))
                data_to_pretrain = pickle.load(open(pathlib.PurePath(video_alignment_folder,
                                               f"{k + 1}_pretrain_robot_network_{tool_name}_{video_id}.pkl"), 'rb'))
                save_pretrained_model_path = str(pathlib.PurePath(video_alignment_folder,
                                                    f"{k + 1}_pretrained_mu_{robot_name}_{tool_name}_{video_id}.pkl"))
                print(
                    f"For {tool_name}_{video_id} pose seq is {len(alignment_params['tip_poses'])} full is {len(alignment_params['full_tip_poses']) if 'full_tip_poses' in alignment_params else 'None'}")
                # reward = 3
                # if tool_name == 'spade':
                #     alignment_params['tip_poses'] = alignment_params['full_tip_poses']
                #     pickle.dump(alignment_params, open(name_base.format(tool_name, video_id), 'wb'))
                avg_reward = 0
                for seed in seeds:
                    reward = pretrain_network_and_test(robot_name=robot_name,
                                                       tool_name=tool_name,
                                                       ddp_q=ddp_q,
                                                       demo_params=alignment_params,
                                                       data_to_pretrain=data_to_pretrain,
                                                       save_alignment_path=None,
                                                       save_q_traj_path=None,
                                                       save_pretrained_model_path=save_pretrained_model_path,
                                                       snapshot_folder=None,
                                                       use_pretrained=True,
                                                       # visualize= seed == seeds[0],
                                                       visualize=False,
                                                       optimize_z_robot_base=optimize_z_robot_base,
                                                       logger=logger,
                                                       save_sim_snapshots=False
                                                       # adr_append_to_x=['grass_patch_location_0']
                                                       )
                    # reward = repete_joint_traj(
                    #     ddp_q, tool_name, alignment_params,
                    #     robot_name=robot_name,
                    #     optimize_base_rotation=True,
                    #     optimize_z_robot_base=optimize_z_robot_base,
                    #     visualize=False,
                    #     verbose=True)
                    avg_reward += np.sum(reward) / len(seeds)
                # ['Tool', 'Video id', 'Description', 'Seed', 'Reward (avg 5 runs)']
                if avg_reward > 0 and avg_reward > best_score:
                    best_score = avg_reward
                    best_score_k = k + 1
                if extended_cols:
                    print('reward: ', avg_reward)
                    result_list.append([tool_name, video_id, k + 1, 'Initial policy',
                                        alignment_params['trajectory_scale'], seeds, avg_reward])
                else:
                    result_list.append([tool_name, video_id, 'Initial policy', seed, reward])
            # save best alignment to separate folder
            if save_best and best_score_k != -1:
                save_folder = f"{pathlib.Path(__file__).parent.parent}/data/alignment/pretrained/{robot_name}/{tool_name}/video_{video_id}"
                create_if_not_exist(save_folder)
                shutil.copy(pathlib.PurePath(video_alignment_folder,f"{best_score_k}_align_{tool_name}_{video_id}.pkl"),
                            pathlib.PurePath(save_folder, f"align_{tool_name}_{video_id}.pkl"))
                shutil.copy(pathlib.PurePath(video_alignment_folder, f"{best_score_k}_q_traj_{tool_name}_{video_id}.pkl"),
                            pathlib.PurePath(save_folder, f"q_traj_{tool_name}_{video_id}.pkl"))
                shutil.copy(pathlib.PurePath(video_alignment_folder, f"{best_score_k}_pretrained_mu_{robot_name}_{tool_name}_{video_id}.pkl"),
                            pathlib.PurePath(save_folder, f"pretrained_mu_{robot_name}_{tool_name}_{video_id}.pkl"))

save_results(result_list, results_file, extended_cols=True)
