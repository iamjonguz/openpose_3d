import numpy as np
import cv2

import cv2
import pandas as pd

from os import listdir
from os.path import isfile, join

import sys

sys.path.append('../')
from drawing.drawing import draw_stick_figure_op, draw_stick_figure_vp

depth_table = np.load('depth_table.npy')


def rolling_median(data):

    # Convert to panda frame with all keypoints on same row
    frame = []
    for pose in data:
        tmp = np.empty(1)
        for kp in pose:
            tmp = np.concatenate([tmp, kp])        
        frame.append(tmp)
    
    df = pd.DataFrame(frame)
    rdf = df.rolling(window=7).median()
    
    # Convert back to original format
    matrix = rdf.values
    seq = []
    for row in matrix:
        r = []
        for i in range(1, len(row), 3):
            kp = [row[i], row[i+1], row[i+2]]
            r.append(kp)
        seq.append((r))

    return np.array(seq)
    


def get_depth_from_table(keypoints_3d):
        '''
        Will from raw depth data pick the best depth frame from a table containg premade depth frames. 

        It will compare each frame in the frame table with recorded depth frame, using euclidian distance. 
        The frame from the table with the least distance to the recorded frame will be chosen. 
        '''

        depth = []
        for kp in keypoints_3d:
            depth.append(kp[2])

        # Big arbitrary number
        current_best_dist = 1000
        ind = 0

        for i,dl in enumerate(depth_table):

            # Euclidian distance
            dist = np.linalg.norm(np.array(depth)-np.array(dl))

            if dist < current_best_dist:
                current_best_dist = dist
                ind = i

        for kp, d in zip(keypoints_3d, depth_table[ind]):
            kp[2] = d

        return keypoints_3d


def process_depth(sequence_of_poses):

    processed_sequence_of_poses = []
    for keypoints_3d in sequence_of_poses:
        occ_data = keypoints_3d[0]
        keypoints_3d = np.delete(keypoints_3d, 0, axis=0)
        if occ_data[1] == True:
            keypoints_3d = get_depth_from_table(keypoints_3d)

        processed_sequence_of_poses.append(keypoints_3d)

    return processed_sequence_of_poses


def create_video(path_to_images="../data/processed/images/", save_as="animations.mp4"):
    img_array = []
    onlyfiles = [f for f in listdir(path_to_images) if isfile(join(path_to_images, f))]
    sorted_files = sorted(onlyfiles, key=lambda x: int(x.split('.')[0][5:]))

    for filename in sorted_files:
        img_array.append(cv2.imread(path_to_images + filename))

    image = img_array[0]
    size = (image.shape[1],image.shape[0])
    
    out = cv2.VideoWriter(f'../data/processed/animations/{save_as}', 0x00000021, 10, size)

    for img in img_array:
        out.write(img) 
    out.release()
        

def create_animation_images(op_poses, vp_poses, path_to_images):

    onlyfiles = [f for f in listdir(path_to_images) if isfile(join(path_to_images, f))]
    sorted_files = sorted(onlyfiles, key=lambda x: int(x.split('.')[0][5:]))
    i = 0

    for img, op_pose, vp_pose in zip(sorted_files, op_poses, vp_poses):
        p1 = draw_stick_figure_op(op_pose, -90, -50)
        p2 = draw_stick_figure_vp(vp_pose, -90, -90)
        img = cv2.imread(path_to_images + img)
     
        combined_img = np.hstack((img, p1, p2))

        cv2.imwrite(f'../data/processed/images/image{i}.jpg', combined_img)
        i+=1

def create_animation_images_op(op_poses, path_to_images):

    onlyfiles = [f for f in listdir(path_to_images) if isfile(join(path_to_images, f))]
    sorted_files = sorted(onlyfiles, key=lambda x: int(x.split('.')[0][5:]))
    i = 0

    for img, op_pose in zip(sorted_files, op_poses):
        p1 = draw_stick_figure_op(op_pose, -90, -50)
        img = cv2.imread(path_to_images + img)
     
        combined_img = np.hstack((img, p1))

        cv2.imwrite(f'../data/processed/images/image{i}.jpg', combined_img)
        i+=1


def start_processing():
    op = np.load('data/unprocessed/keypoints_unprocessed.npy')
    op = process_depth(op)
    op = rolling_median(op)

    vp = np.load('data/processed/VideoPose.npy')
    create_animation_images(op, vp, 'data/unprocessed/images/keypoints/')

    #np.save('processed_data.npy', f)

    create_video()   


if __name__ == '__main__':
    
    '''
    
    op = np.load('../data/unprocessed/keypoints_unprocessed.npy')
    op = process_depth(op)
    op = rolling_median(op)

    vp = np.load('../data/processed/VideoPose.npy')
    create_animation_images(op, vp,'../data/unprocessed/images/keypoints/')

    #np.save('processed_data.npy', f)
    '''
    create_video("../data/processed/images/", "animation2.mp4")