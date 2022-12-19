import numpy as np
import pathlib
import argparse
import shutil
import json


parser = argparse.ArgumentParser()
parser.add_argument('-exp_folder', type=str, default='train_ppo', help='String date for experiments folder')
options = parser.parse_args()
experiment_folder = options.exp_folder

run_id_list = [32, 33, 34, 35, 36]
root_experiments_folder = f"{pathlib.Path(__file__).parent.parent}/data/{experiment_folder}"
save_parent_folder = f"{pathlib.Path(__file__).parent.parent}/data/pretrain_components/final_policy"
for run_id in run_id_list:
    exp_id_folder = pathlib.PurePath(root_experiments_folder, f"run_{run_id}")
    itr_candidates = [int(path.name[4:-4]) for path in pathlib.Path(exp_id_folder).iterdir() if 'itr' in path.name]
    max_iter = 0 if len(itr_candidates) == 0 else np.max(itr_candidates)
    src = pathlib.PurePath(exp_id_folder, f"itr_{max_iter}.pkl")

    params = json.load(open(pathlib.PurePath(exp_id_folder, 'params.json'), 'rb'))
    save_folder = pathlib.PurePath(save_parent_folder, params['tool'], params['alignment_path'].split('/')[-2])
    save_folder.mkdir(parents=True, exist_ok=True)
    dst = pathlib.PurePath(save_folder, 'policy.pkl')
    shutil.copyfile(src, dst)


