import pandas as pd

csv_file = '../data/results.csv'
tools_list = ['spade', 'scythe', 'hammer']

for tool_name in tools_list:
    for video_id in range(1, 6):
        reward = 0

