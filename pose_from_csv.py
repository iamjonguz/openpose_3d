
import numpy as np
import pandas as pd
 
from csv import reader

def read_csv(path_to_csv):

    with open(path_to_csv) as read_obj:
        frame = pd.read_csv(read_obj)
    return frame

def create_poses(frame=None):
    all_frames = []
    for i in range(1, len(frame.index)-1):
        pose_frame = []
        for c in frame:
            pose_frame.append(frame[c][i])
        all_frames.append(pose_frame)

    return all_frames

def create_poses_one_frame(frame=None):
    all_frames = []
    pose_frame = []
    for c in frame:
        pose_frame.append(frame[c][0])
    all_frames.append(pose_frame)
    return all_frames


def create_nodes(pose_matrix):

    all_poses = []
    for pose in pose_matrix:
        one_pose = []
      
        for i in range(0, len(pose), 3):
            one_pose.append((pose[i], 300*pose[i+2], -1*pose[i+1]))
        all_poses.append(np.array(one_pose)/80)
    
    return all_poses

def poses(path_to_csv):
    f = read_csv(path_to_csv)
    pm = create_poses(f)
    return create_nodes(pm)


if __name__ == "__main__":
    print(poses('scripts/original.csv'))