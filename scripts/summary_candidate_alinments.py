import pathlib
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-str_date', type=str, default='01_01_01', help='String date for alignment folder')
parser.add_argument('-robot', type=str, default='panda', help='Robot name')

options = parser.parse_args()
experiment_group_name = options.str_date

root_alignment_folder = f"{pathlib.Path(__file__).parent.parent}/data/alignment/{experiment_group_name}"
tools = [path.name for path in pathlib.Path(root_alignment_folder).iterdir() if path.is_dir()]
count_alignments = {}
for tool_name in tools:
    count_alignments[tool_name] = {}
    tool_alignment_folder = pathlib.PurePath(root_alignment_folder, tool_name)
    vid_ids = [path.name for path in pathlib.Path(tool_alignment_folder).iterdir() if path.is_dir()]
    for vid_id in vid_ids:
        fol = pathlib.PurePath(tool_alignment_folder, vid_id)
        align_files = [path.name for path in pathlib.Path(fol).iterdir() if path.name[:2].isdigit()]
        count_alignments[tool_name][vid_id] = len(align_files)
print(count_alignments)
