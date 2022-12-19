# Getting reference 3D motion and contact forces from MFV outputs

### Loading MFV datafile
Use the following Python scirpts to load the data file output by MFV.
```python
from load_mfv_outputs import *

data_path = "path/to/datafile.pkl" # Update the path
mfv_data = load_mfv_outputs(data_path)
print(mfv_data.keys())
```

### Getting person joint positions from the data file
The MFV estimator uses a human rig of 24 joints (see `mfv_person_joints.txt` for joint names and joint ids).
You can get the estimated 3D trajectories of any joints using the following scripts:
```python
joint_names = ["l_fingers", "r_fingers", "l_shoulder", "pelvis"] # Customize this line
hand_positions = get_person_joint_trajectories(mfv_data, joint_names)
```

### Getting object keypoiint positions from the data file
Similarly, for getting object keypoint positions, checkout `mfv_object_keypoints.txt` for keypoint names and run the following scripts:
```python
keypoint_names = ["handle_end", "tool_head"] # Customize this line
endpoint_positions = get_object_keypoint_trajectories(mfv_data, keypoint_names)
```
