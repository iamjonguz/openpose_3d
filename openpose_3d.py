# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import time
import pyrealsense2 as rs
from os import listdir
from os.path import isfile, join
from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt

from drawing.drawing import draw_keypoints

from knn.k_nearest import create_knn


dir_path = os.path.dirname(os.path.realpath(__file__))

if platform == "win32":
    # Change these variables to point to the correct folder (Release/x64 etc.)
    sys.path.append(dir_path + '/../../python/openpose/Release')
    os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
    
    # Import from a lib file
    import pyopenpose as op


class OpenPose:

    def __init__(self):

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

        self.intrinsics = self.get_camera_intrinsics()

        self.pipeline = rs.pipeline()
        self.config = rs.config() 

        self.knn = create_knn('knn/files_combined.npy')


    def estimate_sequence(self):

        self.profile = self.pipeline.start()

        time.sleep(3)

        final_keypoints = []
        frame_number = 0
        try:
            while True:           
                keypoints_3d = self.estimate_3d_picture(frame_number)
                final_keypoints.append(keypoints_3d)
                frame_number+=1
        finally:      
            with open('data/unprocessed/keypoints_unprocessed.npy', 'wb') as f1:
                np.save(f1, final_keypoints) 


    def estimate_3d_picture(self, frame_number):
            '''
            Will estimate the pose in 3D, by first estimating the 2D poses and then getting the 3D poses with the help 
            of the 3D data from the 3D camera. 
            '''

            frames = self.pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_image = np.asarray(frames.get_color_frame().get_data())

            cv2.imwrite(f"data/unprocessed/images/original/image{frame_number}.jpg", color_image)

            self.datum.cvInputData = color_image
            self.opWrapper.emplaceAndPop([self.datum])
            keypoints_2d = self.datum.poseKeypoints[0]
            keypoints_2d = self.filter_keypoints(keypoints_2d)
            keypoints_3d = self.get_depth_from_frame(keypoints_2d, depth_frame)
            keypoints_3d = self.translate_to_camera_space(keypoints_3d)
            keypoints_3d = self.make_hip_root(keypoints_3d)

            if self.detect_occlusion(keypoints_3d):
                text = 'Occlusion'
                keypoints_3d.insert(0, [True,True,True])

            else:
                keypoints_3d.insert(0, [False,False,False])
                text = 'No occlusion'

            # ------------------------- Drawing ----------------------------
            color_image = draw_keypoints(color_image, keypoints_2d, frame_number, text)

            cv2.imshow('Frame', color_image)
        
            # Mark as training data. True = Occlusion, False = No occlusion
            #keypoints_3d.insert(0, [False,False,False])

            cv2.waitKey(1)
            cv2.imwrite(f"data/unprocessed/images/keypoints/image{frame_number}.jpg", color_image)
            return keypoints_3d


    def filter_keypoints(self, keypoints_2d):
        '''
        Filter keypoints from foot and head just leaving one keypoint for each of them.
        '''
        filtered_kp = []
        for i, kp in enumerate(keypoints_2d):
            if i == 19 or i == 20 or i == 21 or i == 22 or i == 23 or i == 24 or i == 15 or i == 16 or i == 17 or i == 18 :
                pass
            else:
                filtered_kp.append(kp)

        return filtered_kp


    def get_depth_from_frame(self, keypoints_2d, depth_frame):
        frame_with_depth = []
        for kp in keypoints_2d:
            try:
                depth = depth_frame.get_distance(kp[0], kp[1])
                a = [kp[0], kp[1], depth]
                frame_with_depth.append(a)
           
            # Openpose estimates keypoints to be out of the picture
            except IndexError:
                frame_with_depth.append([0, 0, 0])


        return frame_with_depth

    def translate_to_camera_space(self, keypoints_3d):
        
        translated_keypoints = []

        for kp in keypoints_3d:
            translated_keypoints.append(rs.rs2_deproject_pixel_to_point(self.intrinsics, [kp[0], kp[1]], kp[2]))

        return translated_keypoints


    def make_hip_root(self, keypoints):      

        root = keypoints[8] # hip

        new_kp = []

        for kp in keypoints:
            tmp_x = kp[0] - root[0]
            tmp_y = kp[1] - root[1]
            tmp_z = kp[2] - root[2]
        
            new_kp.append([tmp_x, tmp_y, tmp_z])

        return new_kp


    def get_camera_intrinsics(self):
        '''
        A way to get the camera intrinsics as an object to be used for 
        translation to camera space. 
        '''
        pipe = rs.pipeline()
        profile = pipe.start()
        frames = pipe.wait_for_frames()
        depth_frame =  frames.get_depth_frame()
        return depth_frame.profile.as_video_stream_profile().get_intrinsics()

    
    def detect_occlusion(self, frame_with_depth):

        flatten_list = []
        for kp in frame_with_depth:
            flatten_list += kp

        flatten_list = np.array(flatten_list).reshape(1, -1)
        occ = self.knn.predict(flatten_list)
        return occ


if __name__ == "__main__":
    op = OpenPose()
    op.estimate_sequence()

