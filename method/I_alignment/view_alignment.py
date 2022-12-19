import os
import pickle
from pyphysx_envs.envs import ToolEnv
from pyphysx_envs.utils import follow_tool_tip_traj
from alignment_utils import get_reward_to_track, get_sorted_alignments_from_folder
import argparse
import pathlib

parser = argparse.ArgumentParser()

parser.add_argument('-str_date', type=str, default='01_01_01', help='String date for folder for experiment tracking')
parser.add_argument('-tool', type=str, default='spade', help='Tool name to explore')
parser.add_argument('-seed', type=int, default=111, help='Seed for alignment')
parser.add_argument('-vid_id', type=int, default=1, help='ID of the video of interest')
options = parser.parse_args()

tool_name = options.tool
video_id = options.vid_id
reward_to_track_name = get_reward_to_track(tool_name)
which_alignment_show = 0

alignment_folder = f"{pathlib.Path(__file__).parent.parent.parent}/data/alignment/{options.str_date}/{tool_name}/video_{video_id}"
sorted_alignment_filenames = get_sorted_alignments_from_folder(alignment_folder, verbose=True)
if len(sorted_alignment_filenames) == 0:
    print(f"Folder {alignment_folder} contains 0 alignments")
    exit(10)
print(f"Visualizing {sorted_alignment_filenames[which_alignment_show]} file")

alignment_params = pickle.load(
    open(os.path.join(alignment_folder, sorted_alignment_filenames[which_alignment_show]), "rb"))
alignment_params['tool_init_position'] = alignment_params['tool_init_position'][0]
# print(alignment_params)
print(alignment_params.keys())
print(alignment_params['tool_init_position'])
# exit(10)
env = ToolEnv(scene_name=tool_name, tool_name=tool_name,
              render=True,
              return_rewads=True,
              add_spheres=True,
              use_simulate=False if tool_name == 'scythe' else True,
              nail_dim=((0.05, 0.05, 0.02), (0.01, 0.01, 0.2)),
              grass_patch_n=1,
              spheres_reward_weigth=1,
              threshold_cuting_vel=0.01,
              rate=60 if tool_name == 'spade' and video_id > 19 else 24,
              # rate=24,
              params=alignment_params,
              render_dict=dict(
                  # show_frames=True,
                  use_meshcat=True, open_meshcat=True, wait_for_open=True, render_to_animation=True,
                  animation_fps=24,
              )
              )

print(len(alignment_params['full_tip_poses']))
results = follow_tool_tip_traj(env, alignment_params['full_tip_poses'],  # alignment_params['tip_poses'],
                               get_reward_to_track(tool_name), return_last_step_id=False,
                               )
env.renderer.publish_animation()
