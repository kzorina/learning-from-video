import pickle
from pyphysx_envs.utils import follow_tool_tip_traj
from method.utils import create_tool_env, get_reward_to_track
import numpy as np

def repete_tool_traj_from_alignment(alignment_params, visualize=False, tool_name='spade', seed=None, verbose=False,
                                    stop_on_positive_reward=False, return_last_step_id=False,
                                    rate=24):
    if seed is not None:
        np.random.seed(seed)
    env = create_tool_env(tool_name, visualize, rate, alignment_params)
    results = follow_tool_tip_traj(env, alignment_params['tip_poses'],  # alignment_params['tip_poses'],
                                   get_reward_to_track(tool_name),
                                   return_last_step_id=return_last_step_id,
                                   add_zero_end_steps=5,
                                   verbose=verbose,
                                   stop_on_positive_reward=stop_on_positive_reward,
                                   default_start_height=0.4
                                   )
    alignment_params['tip_poses'] = results['tool_tip_pose_list']
    if visualize:
        env.renderer.publish_animation()
    if verbose:
        print(f"total reward = {results['total_reward_to_track_list']}")

    return results['total_reward_to_track_list'], alignment_params


if __name__ == '__main__':
    robot_name = 'panda'
    tool_name = 'hammer'
    video_id = 1
    # seed = 31

    alignment_filename = '/home/kzorina/Work/git_repos/learning-from-video-debug/data/alignment/debug_hammer_1/align_hammer_1.pkl'
    alignment_params = pickle.load(open(alignment_filename, "rb"))
    total_reward, new_alignment_params = repete_tool_traj_from_alignment(alignment_params,
                                    visualize=True,
                                    tool_name=tool_name,
                                    verbose=False,
                                    # add_end_steps=5,
                                    stop_on_positive_reward=True,
                                    # return_last_step_id=False,
                                    rate=24)
    print(total_reward)
    pickle.dump(new_alignment_params, open("../../data/debug/alignment_params.pkl", "wb"))
