from utils import save_results
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
parser.add_argument('-fps', type=int, default=24, help='Frequency of demonstration')
parser.add_argument('-tool', type=str, default='spade', help='Tool name to explore')
parser.add_argument('-seed', type=int, default=111, help='Seed for alignment')
parser.add_argument('-vid_id', type=int, default=1, help='ID of the video of interest')
options = parser.parse_args()

results_file = f'{pathlib.Path(__file__).parent.parent}/data/results_cluster_{options.str_date}.csv'
logger = LogText(f"{pathlib.Path(__file__).parent.parent}/data/logs/IV_pretrain_nn.txt")
robot_name = 'panda'
# seeds = [123, 42, 77331]
seeds = [123, 42, 77331, 321, 234]
extended_cols = True
save_best = True

result_list = []
root_alignment_folder = f"{pathlib.Path(__file__).parent.parent}/data/alignment/{options.str_date}"
tool_name = options.tool
video_id = options.vid_id
optimize_z_robot_base = False

video_alignment_folder = pathlib.PurePath(root_alignment_folder, tool_name, f'video_{video_id}', options.robot)
if pathlib.Path(video_alignment_folder).exists():
    count_alignments = len(
        [path for path in pathlib.Path(video_alignment_folder).iterdir()]) // 4
    print(f"Count alignemnts: {count_alignments}")
    best_score = 0
    best_score_k = -1
    for k in range(count_alignments):
        alignment_params = pickle.load(open(pathlib.PurePath(video_alignment_folder,
                                                             f"{k}_align_{tool_name}_{video_id}.pkl"),
                                            'rb'))
        ddp_q = pickle.load(open(pathlib.PurePath(video_alignment_folder,
                                                  f"{k}_q_traj_{tool_name}_{video_id}.pkl"), 'rb'))
        data_to_pretrain = pickle.load(open(pathlib.PurePath(video_alignment_folder,
                                                             f"{k}_pretrain_robot_network_{tool_name}_{video_id}.pkl"),
                                            'rb'))
        save_pretrained_model_path = str(pathlib.PurePath(video_alignment_folder,
                                                          f"{k}_pretrained_mu_{robot_name}_{tool_name}_{video_id}.pkl"))
        avg_reward = 0
        min_reward = 0
        for seed in seeds:
            reward = pretrain_network_and_test(robot_name=robot_name,
                                               tool_name=tool_name,
                                               seed=seed,
                                               ddp_q=ddp_q,
                                               demo_params=alignment_params,
                                               data_to_pretrain=data_to_pretrain,
                                               save_alignment_path=None,
                                               save_q_traj_path=None,
                                               save_pretrained_model_path=save_pretrained_model_path,
                                               snapshot_folder=None,
                                               use_pretrained=True,
                                               visualize=False,
                                               rate=options.fps,
                                               optimize_z_robot_base=optimize_z_robot_base,
                                               logger=logger,
                                               save_sim_snapshots=False
                                               )
            avg_reward += np.sum(reward) / len(seeds)
            if np.min(reward) < min_reward:
                min_reward = np.min(reward)
            print(f"Seed {seed}: {np.sum(reward)}")
        # ['Tool', 'Video id', 'Description', 'Seed', 'Reward (avg 5 runs)']
        if avg_reward > 0 and avg_reward > best_score:
            best_score = avg_reward
            best_score_k = k
        if extended_cols:
            result_list.append([tool_name, video_id, k, 'Initial policy',
                                alignment_params['trajectory_scale'], seeds, avg_reward, min_reward])
        else:
            result_list.append([tool_name, video_id, 'Initial policy', seed, reward])
    # save best alignment to separate folder
    if save_best and best_score_k != -1:
        save_folder = f"{pathlib.Path(__file__).parent.parent}/data/alignment/pretrained/{robot_name}/{tool_name}/video_{video_id}"
        create_if_not_exist(save_folder)
        shutil.copy(
            pathlib.PurePath(video_alignment_folder, f"{best_score_k}_align_{tool_name}_{video_id}.pkl"),
            pathlib.PurePath(save_folder, f"align_{tool_name}_{video_id}.pkl"))
        shutil.copy(
            pathlib.PurePath(video_alignment_folder, f"{best_score_k}_q_traj_{tool_name}_{video_id}.pkl"),
            pathlib.PurePath(save_folder, f"q_traj_{tool_name}_{video_id}.pkl"))
        shutil.copy(pathlib.PurePath(video_alignment_folder,
                                     f"{best_score_k}_pretrained_mu_{robot_name}_{tool_name}_{video_id}.pkl"),
                    pathlib.PurePath(save_folder, f"pretrained_mu_{robot_name}_{tool_name}_{video_id}.pkl"))

# save_results(result_list, results_file, extended_cols=True)
print(result_list)
