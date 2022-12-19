from pyphysx import *
from pyphysx_envs.envs import ToolEnv, RobotEnv
import pathlib
import shutil
import os


def create_tool_env(tool_name, render, rate, alignment_params):
    """
    Creates tool-only environment with default setup.

    """
    return ToolEnv(scene_name=tool_name,
                   tool_name=tool_name,
                   path_spheres_n=0,  # len(alignment_params['tip_poses']),
                   return_rewads=True,
                   use_simulate=False if tool_name == 'scythe' else True,
                   nail_dim=((0.05, 0.05, 0.02), (0.01, 0.01, 0.2)),
                   grass_patch_n=1,
                   threshold_cuting_vel=0.02,
                   render=render,
                   rate=rate,
                   params=alignment_params,
                   add_spheres=True,
                   render_dict=dict(use_meshcat=True,
                                    open_meshcat=True,
                                    wait_for_open=True,
                                    render_to_animation=True,
                                    animation_fps=24,
                                    # show_frames=True
                                    ),
                   # debug_spheres_pos=alignment_params['tip_poses']
                   )


def default_robot_params_dict(tool_name):
    return {'grass_patch_n': 1,
            'spheres_reward_weigth': 1,
            'threshold_cuting_vel': 0.02,
            'use_simulate': False if tool_name == 'scythe' else True,
            'add_spheres': True,
            'obs_add_q': True,
            'obs_add_sand': False,
            'obs_add_grass_patch_location_0': False,
            'obs_add_goal_box': False,
            'obs_add_nail': False,
            'nail_dim': ((0.05, 0.05, 0.02), (0.01, 0.01, 0.2)),
            'path_spheres_n': 0,
            'velocity_violation_penalty': 0.01,
            'increase_velocity_penalty_factor': 0.0001,
            'increase_velocity_start_itr': 0,
            'action_l2_regularization': 1.e-7,
            'broken_joint_penalty': 0.01,
            }


def create_robot_env(tool_name, robot_name, render, rate,
                     alignment_params, add_steps_end,
                     demo_q, robot_init_pose, render_to_animation):
    """
    Creates robot environment with selected scene and default parameters

    """
    default_robot_params = default_robot_params_dict(tool_name)
    return RobotEnv(scene_name=tool_name,
                    tool_name=tool_name,
                    robot_name=robot_name,
                    rate=rate,
                    render=render,
                    demonstration_q=demo_q,
                    robot_pose=robot_init_pose,
                    params=alignment_params,
                    batch_T=len(demo_q) + add_steps_end,
                    render_dict=dict(use_meshcat=True,
                                     open_meshcat=True,
                                     wait_for_open=True,
                                     render_to_animation=render_to_animation,
                                     animation_fps=24,
                                     # show_frames=True
                                     ),
                    debug_spheres_pos=alignment_params['tip_poses'],
                    **default_robot_params
                    )


def get_reward_to_track(tool_name):
    rewards_to_track = {'spade': ['spheres'],
                        'hammer': ['nail_hammered', 'overlaping_penalty'],
                        # 'hammer': ['nail_hammered'],
                        'scythe': ['cutted_grass']}
    return rewards_to_track[tool_name]


def create_if_not_exist(dir, remove=True):
    save_dir_path = pathlib.Path(dir)
    if remove:
        shutil.rmtree(save_dir_path, ignore_errors=True)
    save_dir_path.mkdir(parents=True, exist_ok=True)


def get_default_robot_z(robot_name, tool_name):
    return -0.2 if robot_name == 'panda' and tool_name == 'spade' else 0.3 if robot_name == 'talos_arm' else 0


class LogText:
    def __init__(self, path='default_log.txt', mode='a'):
        self.path = path
        self.mode = mode

    def log(self, text):
        log_file = open(self.path, self.mode)
        log_file.write(text)
        log_file.close


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
