from method.II_trajectory_optimization.I_repeate_tool_traj import repete_tool_traj_from_alignment
import pathlib
import pickle
import argparse

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('-results_file', type=str,
                        default=f'{pathlib.Path(__file__).parent.parent.parent}/data/final_results.csv',
                        help='String name of the file where result should be saved.')
arg_parser.add_argument('-tool_names', nargs='+', default=['spade'], help='For which tools to evaluate')
arg_parser.add_argument('-video_ids', type=int, nargs='+', default=[1], help='For which videos to evaluate')
arg_parser.add_argument('-seeds', type=int, nargs='+', default=[42], help='Seeds to use for each run of evaluation')
arg_parser.add_argument('-fps', type=int, default=24, help='Frequency of demonstration')
arg_parser.add_argument('--visualize', dest='visualize', action='store_true')
arg_parser.add_argument('--verbose', dest='verbose', action='store_true')
args = arg_parser.parse_args()
# del arg_parser
# print(args)
results_file = args.results_file
tool_names = args.tool_names
video_ids = args.video_ids
seeds = args.seeds
verbose = args.verbose
visualize = args.visualize
fps = args.fps

result_list = []
for tool_name in tool_names:
    for video_id in video_ids:
        pretrain_dir = f"{pathlib.Path(__file__).parent.parent.parent}/data/pretrain_components"
        alignment_path = pathlib.PurePath(pretrain_dir, f"alignment/{tool_name}/video_{video_id}/alignment.pkl")
        avg_reward = 0
        if pathlib.Path(alignment_path).is_file():
            alignment_params = pickle.load(open(alignment_path, 'rb'))

            for seed in seeds:
                reward, _ = repete_tool_traj_from_alignment(
                    alignment_params.copy(), tool_name=tool_name, rate=fps, seed=seed,
                    stop_on_positive_reward=False if tool_name == 'spade' else True,
                    visualize=visualize if seed == seeds[0] else False, verbose=verbose)
                avg_reward += reward / len(seeds)
        else:
            print(f"file {alignment_path} not found!")
        if verbose:
            print(f"Final reward = {round(avg_reward, 2)}")
        result_list.append([tool_name, video_id, 'Follow alignment', alignment_params['trajectory_scale'], seeds,
                            avg_reward])
print(result_list)
