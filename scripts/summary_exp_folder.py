import pathlib
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-exp_folder', type=str, default='train_ppo', help='String date for experiments folder')

options = parser.parse_args()
experiment_folder = options.exp_folder

root_experiments_folder = f"{pathlib.Path(__file__).parent.parent}/data/{experiment_folder}"
exp_ids = [path.name for path in pathlib.Path(root_experiments_folder).iterdir() if path.is_dir()]
exp_ids.sort()
params_of_interest = ['alignment_path']
for exp_id in exp_ids:
    exp_id_folder = pathlib.PurePath(root_experiments_folder, exp_id)
    # print([path.name[4:-4] for path in pathlib.Path(exp_id_folder).iterdir() if 'itr' in path.name])
    # exit()
    itr_candidates = [int(path.name[4:-4]) for path in pathlib.Path(exp_id_folder).iterdir() if 'itr' in path.name]
    max_iter = 0 if len(itr_candidates) == 0 else np.max(itr_candidates)
    params = json.load(open(pathlib.PurePath(exp_id_folder, 'params.json'), 'rb'))
    # print(params)
    # print(params.keys())
    print(f"""Exp {params['run_ID']} for {params['tool']} has {max_iter} iter 
        with params of interest: [{[params[name] for name in  params_of_interest]}] """)

