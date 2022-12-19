import numpy as np
from method.I_alignment.human_demonstrations.io_utils import load_pkl, load_parameters_from_txt
import os


def load_mfv_outputs(data_path, load_contact_forces=True):
    '''
    Load reference 3D motion and contact forces output by the MFV (motion and forces from video) estimator.
    -
    Terms explaned:
      nf: number of video frames
      fps: frame rate
      nq_person: length of person configuration vector
      nq_object: lenght of object configuration vector
    -
    Example:
      data_path = "/path/to/MFV/output.pkl"
      mfv_data = load_mfv_outputs(data_path)
    '''
    data = load_pkl(data_path)
    mfv_data = dict()
    mfv_data["fps"] = data["fps"]

    # Get object configuration (nq_object x nf array)
    # nq_object is of length 7 (xyz+quaternion)
    config_object = data["config_object"]  # nf x nq_object matrix
    mfv_data["config_object"] = config_object.T.getA()

    # Get 3D trajectories of the 2 object endpoints (nf x 2 x 3 array)
    # Handle end: first endpoint
    # Tool head: second endpoint
    object_3d_keypoints = data["keypoint_3d_positions_object"]  # 6 x nf matrix
    mfv_data["object_3d_keypoints"] = object_3d_keypoints.T.getA().reshape((-1, 2, 3))

    # Get person configuration (nf x nq_person array)
    # nq_person is of length 99, which includes 23 spherical joints
    # represented by unit quaternions, plus 1 free flyer represented
    # by xyz+quaternion
    config_person = data["config_person"]  # 99 x nf matrix
    mfv_data["config_person"] = config_person.T.getA()

    # Get 3D trajectories of all 24 person joints (nf x 24 x 3 array)
    person_3d_joints = data["joint_3d_positions_person"]  # 72 x nf matrix
    mfv_data["person_3d_joints"] = person_3d_joints.T.getA().reshape((-1, 24, 3))

    # Get contact states (nf x 24 array) and contact mapping (1D array of size 24)
    mfv_data["contact_states"] = data["contact_states"].T
    mfv_data["contact_mapping"] = data["contact_mapping"]

    # Get Contact forces (nf x 2 x 6 array)
    # These are 6D contact forces, i.e. 3D linear force + 3D moment
    if load_contact_forces:
        objcet_contact_forces = data["object_contact_forces"]
        mfv_data["object_contact_forces"] = objcet_contact_forces.T.getA().reshape((-1, 2, 6))

    return mfv_data


def get_person_joint_trajectories(mfv_data, joint_names):
    '''
    This funciton outputs a array of size nf x len(joint_names) x 3 indicating the 3D trajectories of the input joint names.
    -
    Inputs:
      mfv_data: a data dict output load_mfv_outputs()
      joint_names: a list of joint names. Checkout mfv_joint_names.txt for accepted joint names.
    -
    Example:
      mfv_data = load_mfv_outputs(data_path)
      joint_names = ["l_fingers", "r_fingers"]
      hand_positions = get_person_joint_trajectories(mfv_data, joint_names)
    '''
    dir_path = os.path.dirname(os.path.realpath(__file__))
    joint_names_dict = load_parameters_from_txt(dir_path + "/mfv_person_joints.txt")
    joint_ids = [joint_names_dict[name][0] for name in joint_names]
    joint_ids = np.array(joint_ids)
    return mfv_data["person_3d_joints"][:, joint_ids, :]


def get_object_keypoint_trajectories(mfv_data, keypoint_names):
    '''
    This funciton outputs a array of size nf x len(keypoint_names) x 3 indicating the 3D trajectories of the input keypoint names.
    -
    Inputs:
      mfv_data: a data dict output load_mfv_outputs()
      keypoint_names: a list of keypoint names. Checkout mfv_object_keypoints.txt for accepted keypoint names.
    -
    Example:
      mfv_data = load_mfv_outputs(data_path)
      keypoint_names = ["handle_end", "tool_head"]
      endpoint_positions = get_object_keypoint_trajectories(mfv_data, keypoint_names)
    '''
    dir_path = os.path.dirname(os.path.realpath(__file__))
    keypoint_names_dict = load_parameters_from_txt(dir_path + "/mfv_object_keypoints.txt")
    keypoint_ids = [keypoint_names_dict[name][0] for name in keypoint_names]
    keypoint_ids = np.array(keypoint_ids)
    return mfv_data["object_3d_keypoints"][:, keypoint_ids, :]
