import numpy as np

def create_matching_keypoints_vp(keypoints):
    '''
    Take out the necessary keypoints from VideoPose
    '''

    matching_keypoints = []
    for pose in keypoints:

        hip = pose[0]
        l_hip = pose[4]
        l_knee = pose[5]
        l_foot = pose[6]
        r_hip = pose[1]
        r_knee = pose[2]
        r_foot = pose[3]
        l_shoulder = pose[11]
        l_elbow = pose[12]
        l_hand = pose[13]
        r_shoulder = pose[14]
        r_elbow = pose[15]
        r_hand = pose[16]

        matching_keypoints.append([hip, l_hip, l_knee, l_foot, r_hip, r_knee, r_foot, l_shoulder,
                                l_elbow, l_hand, r_shoulder, r_elbow, r_hand])
        
    return np.array(matching_keypoints)

def create_matching_keypoints_op(keypoints):
    '''
    Take out the necessary keypoints from VideoPose
    '''

    matching_keypoints = []
    for pose in keypoints:

        hip = pose[0]
        l_hip = pose[12]
        l_knee = pose[13]
        l_foot = pose[14]
        r_hip = pose[9]
        r_knee = pose[10]
        r_foot = pose[11]
        l_shoulder = pose[5]
        l_elbow = pose[6]
        l_hand = pose[7]
        r_shoulder = pose[2]
        r_elbow = pose[3]
        r_hand = pose[4]

        matching_keypoints.append([hip, l_hip, l_knee, l_foot, r_hip, r_knee, r_foot, l_shoulder,
                                l_elbow, l_hand, r_shoulder, r_elbow, r_hand])
        
    return np.array(matching_keypoints)


def compare(VideoPose_kp, OpenPose_kp):
    VideoPose_kp = np.delete(VideoPose_kp, 0, axis=0)

    mpjpe = 0
    for vp_frame, op_frame in zip(VideoPose_kp, OpenPose_kp):
        dist = 0
        for vp_kp, op_kp in zip(vp_frame, op_frame):
            euc_dist = np.linalg.norm(vp_frame-op_frame)
            dist += euc_dist
        dist = dist / 13
        mpjpe +=dist

    mpjpe = mpjpe / len(VideoPose_kp)




vp = np.load('test_data/VideoPose.npy')
op = np.load('test_data/OpenPose.npy')

matching_kp_vp = create_matching_keypoints_vp(vp)
print(len(matching_kp_vp))

matching_kp_op = create_matching_keypoints_op(op)
print(len(matching_kp_op))

compare(matching_kp_vp, matching_kp_op)