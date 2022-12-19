import pandas as pd
import os


def get_results_columns():
    return ['Tool', 'Video id', 'Description', 'Seed', 'Reward (avg 5 runs)']

def get_ext_results_columns():
    return ['Tool', 'Video id', 'k try', 'Description', 'Scale', 'Seed', 'Reward']

def save_results(results_list, results_path, extended_cols=False):
    results_df = pd.DataFrame(results_list)
    results_df.columns = get_ext_results_columns() if extended_cols else get_results_columns()
    if os.path.exists(results_path):
        prev_results_df = pd.read_csv(results_path)
        prev_results_df = prev_results_df[get_ext_results_columns() if extended_cols else get_results_columns()]
        # delete all repeting rows
        for row in results_list:
            prev_results_df.drop(prev_results_df[(prev_results_df['Tool'] == row[0]) & (
                    prev_results_df['Video id'] == row[1]) & (prev_results_df['Description'] == row[2]
                                                              )].index, inplace=True)
        results_df = prev_results_df.append(results_df)
        results_df.to_csv(results_path)
    else:
        results_df.to_csv(results_path)
    print(results_df)
