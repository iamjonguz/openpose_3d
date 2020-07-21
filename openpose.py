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

        self.depth_info = create_depth_table()


    def record_estimation_sequence(self):
        frame_number = 0
        time.sleep(3)
        while True:
            img_name = f'img{frame_number}.jpg'
            frame, img = self.estimate_3d_picture(recording=True, frame_number=frame_number, img_name=img_name)
            cv2.imwrite(f'img/{img_name}', img)
            frame_number+=1

    def estimate_3d_picture(self, color_pic=None, depth=None, recording=False, frame_number=None, img_name=None):
        
        # Picture taken from Intel Realsense Camera
        color_pic, depth = self.vr.record_frame()
        
        self.datum.cvInputData = color_pic
        self.opWrapper.emplaceAndPop([self.datum])

        kp = self.datum.poseKeypoints[0]

        d = self.estimate_depth(kp, depth)
        kp_object = self.create_json(kp, d, save=recording, frame_number=frame_number, img_name=img_name)

        # Used for printing
        if recording == False:
            a = kp_object['keypoints']
            kp_list = []
            for k in a: 
                kp_list.append((a[k][0], 300*a[k][2], -1*a[k][1]))
            return np.array(kp_list)
        else:
            return kp_object, color_pic


    def similarity(self, l1, l2):
        '''
        Will compare two depth frames. For each value in in the depth list it compare to a frame from the saved data.
        It does this by adding up all the similarites for each keypoint and take the mean value out of that. 
        '''
        tmp = 0

        for i,j in zip(l1,l2):

            if i == 0 or j == 0:
                tmp += 0
            elif i > j:
                tmp += j/i
            else:
                tmp += i/j

        return tmp/len(l1)


    def estimate_depth(self, kp, depth):
        '''
        An extremly ugly and temporary solution to find the best depth match. For each frame estimation it will 
        loop through all saved valid frames and find the best match. Need a better solution for this. 

        Lookup, maybe using hashes? 
        '''
        depth_list = []
        current_best_sim = 0
        ind = 0

        for k in kp:
            depth_list.append(depth.get_distance(k[0], k[1]))

        #print(depth_list)

        for i,dl in enumerate(self.depth_info):
            sim = self.similarity(depth_list, dl)

            if sim > current_best_sim:
                current_best_sim = sim
                ind = i
        
        print(current_best_sim)

        return self.depth_info[ind]
        #return depth_list


    # Working
    def create_json(self, pose_keypoints, depth, save=False, img_name=None, frame_number=None):

        kp_dict = {}

        for ind, kp in enumerate(pose_keypoints):
            kp_dict[str(ind)] = [kp[0].item(), kp[1].item(), depth[ind]]
        
        data = {}
         # Should be the name of the image so they can get linked together 
        data['img_url'] = img_name
        data['frame'] = frame_number
        data['keypoints'] = kp_dict
            

        if save:
            with open(f'img/img{frame_number}.json', 'w', encoding='utf-8') as output:
                json.dump(data, output, ensure_ascii=False, indent=4)

        return data


if __name__ == "__main__":
    op = OpenPose()
    op.record_estimation_sequence()
