# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import json
import numpy as np

import time

from cam import VideoRecorder

from depth_table import create_depth_table

from k_nearest import knn_classifier


dir_path = os.path.dirname(os.path.realpath(__file__))
if platform == "win32":
    # Change these variables to point to the correct folder (Release/x64 etc.)
    sys.path.append(dir_path + '/../../python/openpose/Release')
    os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
    
    # Import from a lib file
    import pyopenpose as op


class OpenPose:

    def __init__(self):

        self.vr = VideoRecorder()
        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "../../../models/"

        # Used to get keypoints 
        params['part_candidates'] = True

        # Max number of people for pose estimation
        params['number_people_max'] = 1

        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.datum = op.Datum()
        self.opWrapper.start()

        # Just loaded from code, need a better solution
        self.depth_info = create_depth_table()
        self.prev_depth = [1.8]*25

        self.knn = knn_classifier('dataset/occlusion_data.csv')

    def record_estimation_sequence(self):
        '''
        Will record a sequence of frames and for each frame estimate the poses in 3D.

        Each frame will be saved as a picture together with a json file containing
        information about keypoints. 
        '''
        frame_number = 0
        time.sleep(3)
        while True:
            img_name = f'img{frame_number}.jpg'
            _, img = self.estimate_3d_picture(recording=True, frame_number=frame_number, img_name=img_name)
            cv2.imwrite(f'img/{img_name}', img)
            frame_number+=1

    def estimate_3d_picture(self, recording=False, frame_number=None, img_name=None, use_table_data=True):
        '''
        Will estimate the pose in 3D, by first estimating the 2D poses and then getting the 3D poses with the help 
        of the 3D data from the 3D camera. 

        Returns the estimated 3D keypoints and the picture of the estimation. 
        '''

        # Picture taken from Intel Realsense Camera
        color_pic, depth_data = self.vr.record_frame()
        
        self.datum.cvInputData = color_pic
        self.opWrapper.emplaceAndPop([self.datum])

        keypoints_2d = self.datum.poseKeypoints[0]

        keypoints_depth = self.get_depth_from_camera(keypoints_2d, depth_data)

        estimated_keypoints_3d = self.create_json(keypoints_2d, keypoints_depth, save=recording, frame_number=frame_number, img_name=img_name)

        return estimated_keypoints_3d, color_pic

    def get_depth_from_camera(self, kp, depth):
        '''
        Will return the raw depth data from the camera.
        ''' 
        depth_list = []
         
        # Getting the depths of the estimated keypoints
        for i, k in enumerate(kp):
            dist = depth.get_distance(k[0], k[1])
            if dist == 0 and self.prev_depth[i] != 0:
                dist = self.prev_depth[i]
            depth_list.append(dist)

        self.prev_depth = depth_list
        return depth_list

    def estimate_depth_from_table(self, depth_list):
        '''
        Will from raw depth data pick the best depth frame from a table containg premade depth frames. 

        It will compare each frame in the frame table with recorded depth frame, using euclidian distance. 
        The frame from the table with the least distance to the recorded frame will be chosen. 
        '''

        current_best_dist = 0
        ind = 0

        for i,dl in enumerate(self.depth_info):

            # Euclidian distance
            dist = np.linalg.norm(np.array(depth_list)-np.array(dl))

            if dist > current_best_dist:
                current_best_dist = dist
                ind = i
        
        return self.depth_info[ind]

    def create_json(self, pose_keypoints, depth, save=False, img_name=None, frame_number=None, path='img'):
        '''
        Will translate the data into more suitable data for printing. If save=true it will 
        save a json file with the keypoint information
        '''
        kp_dict = {}
        kp_list = []



        for ind, kp in enumerate(pose_keypoints):
            kp_dict[str(ind)] = [kp[0].item(), kp[1].item(), depth[ind]]
            kp_list += kp_dict[str(ind)]

        
        kp_list = np.array(kp_list).reshape(1, -1)
        if self.knn.predict(kp_list)[0]:
            print('occlusion')

            depth = self.estimate_depth_from_table(depth)
            
            for ind, kp in enumerate(pose_keypoints):
                kp_dict[str(ind)] = [kp[0].item(), kp[1].item(), depth[ind]]
        else:
            print('no occlusion')
         

        data = {}

        data['img_url'] = img_name
        data['frame'] = frame_number
        data['keypoints'] = kp_dict

        if save:
            with open(f'{path}/img{frame_number}.json', 'w', encoding='utf-8') as output:
                json.dump(data, output, ensure_ascii=False, indent=4)

        return data


if __name__ == "__main__":
    op = OpenPose()
    op.record_estimation_sequence()
