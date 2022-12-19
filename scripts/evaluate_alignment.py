from utils import save_results
from method.II_trajectory_optimization.I_repeate_tool_traj import repete_tool_traj_from_alignment
import pickle
import pathlib
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-str_date', type=str, default='01_01_01', help='String date for alignment folder')
parser.add_argument('-robot', type=str, default='panda', help='Robot name')
options = parser.parse_args()

results_file = f'../data/results_cluster_{options.str_date}.csv'
align_name_base = '../data/pretrain_components/alignment/align_{}_{}.pkl'
tools_list = ['spade']
# tools_list = ['scythe']
# seed = 123
seeds = [123, 42]
extended_cols = True

result_list = []
root_alignment_folder = f"{pathlib.Path(__file__).parent.parent}/data/alignment/{options.str_date}"
for tool_name in tools_list:
    for video_id in range(1, 6):
    # for video_id in [2, 4]:
        video_alignment_folder = pathlib.PurePath(root_alignment_folder, tool_name, f'video_{video_id}', options.robot)
        # print(video_alignment_folder)
        # print(len([path for path in pathlib.Path(video_alignment_folder).iterdir()]) // 4)
        if pathlib.Path(video_alignment_folder).exists():
            count_alignments = len(
                [path for path in pathlib.Path(video_alignment_folder).iterdir()]) // 4
            for k in range(count_alignments):
                alignment_params = pickle.load(open(pathlib.PurePath(video_alignment_folder,
                                                                     f"{k + 1}_align_{tool_name}_{video_id}.pkl"), 'rb'))

                print(f"For {tool_name}_{video_id} pose seq is {len(alignment_params['tip_poses'])} full is {len(alignment_params['full_tip_poses']) if 'full_tip_poses' in alignment_params else 'None'}")
                # reward = 3
                # if tool_name == 'spade':
                #     alignment_params['tip_poses'] = alignment_params['full_tip_poses']
                #     pickle.dump(alignment_params, open(name_base.format(tool_name, video_id), 'wb'))
                avg_reward = 0
                for seed in seeds:
                    reward, _ = repete_tool_traj_from_alignment(alignment_params,
                                                                visualize=True,
                                                                # visualize=False,
                                                                tool_name=tool_name,
                                                                # add_end_steps=0,
                                                                # verbose=True,
                                                                rate=60 if tool_name == 'spade' and video_id >= 4 else 24
                                                                )
                    avg_reward += reward / len(seeds)
                # ['Tool', 'Video id', 'Description', 'Seed', 'Reward (avg 5 runs)']
                if extended_cols:
                    result_list.append([tool_name, video_id, k + 1, 'Follow alignment',
                                        alignment_params['trajectory_scale'], seeds, avg_reward])
                else:
                    result_list.append([tool_name, video_id, 'Follow alignment', seed, reward])

save_results(result_list, results_file, extended_cols=extended_cols)