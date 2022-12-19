import tqdm
from method.utils import create_if_not_exist
from method.I_alignment.alignment_utils import *
from method.I_alignment.params_class import *
from pyphysx_envs.utils import follow_tool_tip_traj
import argparse
import pathlib

parser = argparse.ArgumentParser()

parser.add_argument('-video_data_path', type=str, default='None',
                    help='Path to output .pkl file from Zongmian Li method')
parser.add_argument('-str_date', type=str, default='01_01_01', help='String date for folder for experiment tracking')
parser.add_argument('-tool', type=str, default='spade', help='Tool name to explore')
parser.add_argument('-seed', type=int, default=111, help='Seed for alignment')
parser.add_argument('-vid_id', type=int, default=1, help='ID of the video of interest')
parser.add_argument('-fps', type=int, default=24, help='Video fps')
options = parser.parse_args()

# =============================================================================================================
tool_name = options.tool
i_video = options.vid_id
seed = options.seed
rate = options.fps

n_iter = 20000
double_check_n = 10
acceptable_success_n = 2
stop_serach_n = 50
repeate_last_pose_n = 0
trajectory_min_len = 6

measure_time = True
double_check_alignment = False
# =============================================================================================================
print(f"Running on {tool_name}_{i_video} with seed {seed}, save to {options.str_date}")
video_data_path = options.video_data_path
if video_data_path == 'None':
    video_data_path = f"{pathlib.Path(__file__).parent.parent.parent}/data/video_inputs/{tool_name}_{i_video}_s7.pkl"

save_dir_path = f"{pathlib.Path(__file__).parent.parent.parent}/data/alignment/{options.str_date}/{tool_name}/video_{i_video}"
save_file_base = "{}_scale{}_params_count_{}_smth_{}_{}"
# =============================================================================================================
create_if_not_exist(save_dir_path)

# (i) construct P distribution based on trajectory and environment
# Create env
env = create_tool_env(tool_name, double_check_alignment, rate)
rewards_to_track_name = get_reward_to_track(tool_name)
# Load trajectory from file
# scene_tool_length = np.linalg.norm(env.scene.tool.to_tip_transform[0])  # tool length from the simulation setup
scene_tool_length = env.scene.tool.tool_length  # tool length from the simulation setup

tool_length = 1 if scene_tool_length < 1e-10 else scene_tool_length
head, handle = load_tool_demonstration_extracted_from_video(video_data_path, offset_by_plane=True,
                                                            tool_length=tool_length)
# Construct parameters object
params_sampler = ParamsSampler(env, head, handle, seed)

found_alignments = 0
if measure_time:
    start_iter_time = time.time()
for i in tqdm.trange(n_iter):
    # (ii) sample simulation parameters and update env env
    sampled_params = params_sampler.sample_params()
    env.set_params(sampled_params)
    # (iii) follow trajectory and observe reward
    results = follow_tool_tip_traj(env,
                                   sampled_params['tip_poses'],
                                   rewards_to_track_name=rewards_to_track_name,
                                   add_zero_end_steps=0,
                                   verbose=double_check_alignment)
    reward_to_track = np.sum(results['total_reward_to_track_list'])
    sampled_params['full_tip_poses'] = sampled_params['tip_poses']
    sampled_params['tip_poses'] = results['tool_tip_pose_list']
    if double_check_alignment:
        env.renderer.publish_animation()
        exit(42)
    if reward_to_track > 0:
        traj_reward_list = []
        reward_to_track_list = []
        ids_reward_obtained = []
        for j in range(double_check_n):
            results = follow_tool_tip_traj(env, sampled_params['tip_poses'],
                                           rewards_to_track_name=rewards_to_track_name,
                                           return_last_step_id=True)
            if 0 <= results['last_step_id'] < trajectory_min_len:  # trajectory is too short
                continue
            reward_to_track = np.sum(results['total_reward_to_track_list'])
            traj_reward_list.append(np.sum(results['traj_follow_reward']))
            reward_to_track_list.append(reward_to_track)
            if reward_to_track > 0: ids_reward_obtained.append(results['last_step_id'])

        if len(ids_reward_obtained) >= acceptable_success_n:
            save_filename = f"{found_alignments:02d}_scale_{sampled_params['trajectory_scale']}_params_count_{len(ids_reward_obtained):02d}_reward_{round(np.max(reward_to_track_list), 2)}"
            save_params_to_pickle(sampled_params, ids_reward_obtained, repeate_last_pose_n,
                                  save_dir_path, save_filename)
            found_alignments += 1

    if found_alignments >= stop_serach_n:
        break
    if measure_time and (i == 0 or (i + 1) % 100 == 0):
        print_and_save_iter_time(start_iter_time, tool_name, i, save_dir_path)
