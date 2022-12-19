import vg
from method.I_alignment.human_demonstrations.load_mfv_outputs import *
from method.utils import create_if_not_exist
import numpy as np
from pyphysx_envs.envs import ToolEnv
import time
import pickle

def create_tool_env(tool_name, render, rate):
    """
    Creates tool-only environment with default setup.

    """
    return ToolEnv(scene_name=tool_name,
                   tool_name=tool_name,
                   path_spheres_n=2,
                   return_rewads=True,
                   use_simulate=False if tool_name == 'scythe' else True,
                   nail_dim=((0.05, 0.05, 0.02), (0.01, 0.01, 0.2)),
                   grass_patch_n=1,
                   threshold_cuting_vel=0.02,
                   render=render,
                   rate=rate,
                   add_spheres=True,
                   # add_spheres=False,
                   render_dict=dict(use_meshcat=True,
                                    open_meshcat=True,
                                    wait_for_open=True,
                                    render_to_animation=True,
                                    animation_fps=24,
                                    # show_frames=True,
                                    # frame_scale=0.1
                                    )
                   )

def load_tool_demonstration_extracted_from_video(filename, offset_by_plane=False, offset_by_minimum=False,
                                                 offset_by_z=0., tool_length=0.3):
    """
        The function first loads the human/tool trajectories extracted from the video.

        If offset_by_plane then trajectory of tool is translated by an absolute position of the mid-point between toes.
        Then trajectory is rotated into standard coord. frame with z upwards.
        If offset_by_minimum than minimum values from a trajectory are used to translate the trajectory.
        Finally, offset_by_z value is added to the z-coordinate.

        The handle path is modified to be at tool_length distance from head path.

        Return nx3, nx3 arrays
    """
    mfv_data = load_mfv_outputs(filename)
    keypoint_names = ["handle_end", "tool_head"]
    endpoint_positions = get_object_keypoint_trajectories(mfv_data, keypoint_names)

    if offset_by_plane:
        data = get_person_joint_trajectories(mfv_data, ['l_toes', 'r_toes'])
        center = np.median(np.mean(data, axis=1), axis=0)
        endpoint_positions -= center

    handle_end = endpoint_positions[:, 0, 0:3]
    tool_head = endpoint_positions[:, 1, 0:3]
    handle_end = vg.rotate(handle_end, np.array([1, 0, 0]), angle=-90.)
    tool_head = vg.rotate(tool_head, np.array([1, 0, 0]), angle=-90.)

    if offset_by_minimum:
        offset = np.min(np.min([handle_end, tool_head], axis=0), axis=0)
        handle_end -= offset
        tool_head -= offset

    handle_end[:, 2] += offset_by_z
    tool_head[:, 2] += offset_by_z

    v = handle_end - tool_head
    v = vg.normalize(v)
    handle_end = tool_head + v * tool_length

    return tool_head, handle_end

def print_and_save_iter_time(start_time, tool_name, i, folder="log"):
    print(f"{time.time() - start_time:.2f} sec for {i+1} iter")
    create_if_not_exist(folder, remove=False)
    f = open(os.path.join(folder, f"{tool_name}_{time.strftime('%Y-%m-%d')}.txt"), "a")
    f.write(f"{time.time() - start_time:.2f} sec for {i+1} iter")
    f.close()

def get_reward_to_track(tool_name):
    rewards_to_track = {'spade': ['spheres', 'box_displacement'],
                        'hammer': ['nail_hammered', 'overlaping_penalty'],
                        'scythe': ['cutted_grass']}
    return rewards_to_track[tool_name]

def save_params_to_pickle(params, ids_reward_obtained, repeate_last_pose_n, save_dir, filename):
    params['full_tip_poses'] = params['tip_poses'].copy()
    params['tip_poses'] = params['tip_poses'][:int(np.median(ids_reward_obtained))] + [
        params['tip_poses'][int(np.median(ids_reward_obtained)) - 1]] * repeate_last_pose_n
    pickle.dump(params, open(os.path.join(save_dir, filename), "wb"))

def get_sorted_alignments_from_folder(alignment_folder, verbose=False):
    alignment_filenames = [filename for filename in os.listdir(alignment_folder) if filename[0].isdigit()]
    alignment_paths = [os.path.join(alignment_folder, filename) for filename in alignment_filenames]
    alignment_filenames_splitted = [filename.split('_') for filename in alignment_filenames]
    if verbose:
        print(f"Amount of alignments = {len(alignment_paths)}")
        print(alignment_paths)
        print(alignment_filenames_splitted)
    # score_alignments = [10 * int(item[4]) + float(item[6]) + float(item[7]) for item in alignment_filenames_splitted]
    score_alignments = [10 * int(item[5]) + float(item[7]) for item in alignment_filenames_splitted]
    # score_alignments = [10 * int(item[4]) + float(item[6]) for item in alignment_filenames_splitted]
    return [x for _, x in sorted(zip(score_alignments, alignment_paths), reverse=True)]