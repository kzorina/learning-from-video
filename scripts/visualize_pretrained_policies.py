from method.II_trajectory_optimization.IV_pretrain_nn import pretrain_network_and_test
from method.utils import LogText
import pickle
import pathlib
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-str_date', type=str, default='01_01_01', help='String date for alignment folder')
parser.add_argument('-robot', type=str, default='panda', help='Robot name')
options = parser.parse_args()

logger = LogText(f"../data/logs/temp.txt")
# tools_list = ['scythe', 'spade']
# tools_list = ['spade', 'hammer', 'scythe']
# tools_list = ['hammer']
# tools_list = ['scythe']
tools_list = ['spade']
robot_name = 'panda'
# seed = 123
seeds = [123, 42]#, 1888, 666, 31121995]
root_alignment_folder = f"{pathlib.Path(__file__).parent.parent}/data/alignment/{options.str_date}"
for tool_name in tools_list:
    optimize_z_robot_base = False  # if tool_name == 'spade' else True
    for video_id in range(1, 6):
    # for video_id in [4, 5]:
        video_alignment_folder = pathlib.PurePath(root_alignment_folder, tool_name, f'video_{video_id}', options.robot)
        save_folder = f"{pathlib.Path(__file__).parent.parent}/data/alignment/pretrained/{robot_name}/{tool_name}/video_{video_id}"
        if pathlib.Path(save_folder).exists():
            print(f"Exploring {tool_name}_{video_id}")
            alignment_path = pathlib.PurePath(save_folder, f"align_{tool_name}_{video_id}.pkl")
            q_traj_path = pathlib.PurePath(save_folder, f"q_traj_{tool_name}_{video_id}.pkl")
            pretrain_mu_path = pathlib.PurePath(save_folder, f"pretrained_mu_{robot_name}_{tool_name}_{video_id}.pkl")

            alignment_params = pickle.load(open(alignment_path, 'rb'))
            ddp_q = pickle.load(open(q_traj_path, 'rb'))
            data_to_pretrain = pickle.load(open(pathlib.PurePath(video_alignment_folder,
                                                                 f"1_pretrain_robot_network_{tool_name}_{video_id}.pkl"),
                                                'rb'))
            avg_reward = 0
            for i in range(3):
                reward = pretrain_network_and_test(robot_name=robot_name,
                                                   tool_name=tool_name,
                                                   ddp_q=ddp_q,
                                                   demo_params=alignment_params,
                                                   data_to_pretrain=data_to_pretrain,
                                                   save_alignment_path=None,
                                                   save_q_traj_path=None,
                                                   save_pretrained_model_path=str(pretrain_mu_path),
                                                   snapshot_folder=None,
                                                   use_pretrained=True,
                                                   visualize=True,
                                                   rate=60 if tool_name == 'spade' and video_id >= 4 else 24,
                                                   # visualize=False,
                                                   optimize_z_robot_base=optimize_z_robot_base,
                                                   logger=logger,
                                                   save_sim_snapshots=False
                                                   # adr_append_to_x=['grass_patch_location_0']
                                                   )
                print(reward)
