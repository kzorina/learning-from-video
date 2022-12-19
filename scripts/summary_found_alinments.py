import pathlib
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-str_date', type=str, default='01_01_01', help='String date for alignment folder')
parser.add_argument('-robot', type=str, default='panda', help='Robot name')

options = parser.parse_args()
experiment_group_name = options.str_date

root_alignment_folder = f"{pathlib.Path(__file__).parent.parent}/data/alignment/{experiment_group_name}"
tools = [path.name for path in pathlib.Path(root_alignment_folder).iterdir() if path.is_dir()]
# print(tools)
count_alignments = {}
for tool_name in tools:
    count_alignments[tool_name] = {}
    tool_alignment_folder = pathlib.PurePath(root_alignment_folder, tool_name)
    # print(tool_alignment_folder)
    vid_ids = [path.name for path in pathlib.Path(tool_alignment_folder).iterdir() if path.is_dir()]
    # print(vid_ids)
    for vid_id in vid_ids:
        video_alignment_folder = pathlib.PurePath(tool_alignment_folder, vid_id, options.robot)
        # print(video_alignment_folder)
        # print(len([path for path in pathlib.Path(video_alignment_folder).iterdir()]) // 4)
        if pathlib.Path(video_alignment_folder).exists():
            count_alignments[tool_name][vid_id] = len([path for path in pathlib.Path(video_alignment_folder).iterdir()]) // 4
        else:
            count_alignments[tool_name][vid_id] = 0
print(count_alignments)
