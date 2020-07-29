
import sklearn
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn import linear_model, preprocessing

from openpose import OpenPose

import numpy as np

import json
import cv2
import pandas as pd
import os

import time


def create_json_knn(pose_keypoints, frame_number, occlusion):
    '''
    Will translate the data into more suitable data for printing. If save=true it will 
    save a json file with the keypoint information
    '''
    kp_dict = {}

    kp_list = []

    for ind, kp in enumerate(pose_keypoints):
        kp_list += pose_keypoints[kp]

    data = {}
        # Should be the name of the image so they can get linked together 
    
    data['occlusion'] = occlusion
    data['keypoints'] = kp_list
        
    with open(f'dataset/json/occluded_test_data{frame_number}.json', 'w', encoding='utf-8') as output:
        json.dump(data, output, ensure_ascii=False, indent=4)

    return data

def create_dataset_knn(num_of_frames):
    op = OpenPose()
    time.sleep(3)
    
    frame_number = 0
    for i in range(num_of_frames):
        keypoints, image = op.estimate_3d_picture(use_table_data=False)
        cv2.imwrite(f'dataset/img/occluded_test_data{frame_number}.jpg', image)
        create_json_knn(keypoints['keypoints'], frame_number, True)
        
        print(f'Frame with number {frame_number} created...')
        
        frame_number += 1
        time.sleep(1)
        

def create_csv_knn(path_to_dir='dataset/json'):
    json_dir = os.listdir(path_to_dir)

    json_files = []

    for files in json_dir:
        with open(f'{path_to_dir}/{files}', 'r') as f:
            js = json.load(f)
            tmp = js['keypoints']
            tmp.insert(0, js['occlusion'])
            
        json_files.append(tmp)
        

    columns = ['occlusion', 'kp_0_x', 'kp_0_y', 'kp_0_z', 'kp_1_x','kp_1_y','kp_1_z',
             'kp_2_x', 'kp_2_y', 'kp_2_z', 'kp_3_x','kp_3_y','kp_3_z',
              'kp_4_x', 'kp_4_y', 'kp_4_z', 'kp_5_x','kp_5_y','kp_5_z',
               'kp_6_x', 'kp_6_y', 'kp_6_z', 'kp_7_x','kp_7_y','kp_7_z',
                'kp_8_x', 'kp_8_y', 'kp_8_z', 'kp_9_x','kp_9_y','kp_9_z',
                 'kp_10_x', 'kp_10_y', 'kp_10_z', 'kp_11_x','kp_11_y','kp_11_z',
                  'kp_12_x', 'kp_12_y', 'kp_12_z', 'kp_13_x','kp_13_y','kp_13_z',
                   'kp_14_x', 'kp_14_y', 'kp_14_z', 'kp_15_x','kp_15_y','kp_15_z',
                    'kp_16_x', 'kp_16_y', 'kp_16_z', 'kp_17_x','kp_17_y','kp_17_z',
                     'kp_18_x', 'kp_18_y', 'kp_18_z', 'kp_19_x','kp_19_y','kp_19_z',
                      'kp_20_x', 'kp_20_y', 'kp_20_z', 'kp_21_x','kp_21_y','kp_21_z',
                       'kp_22_x', 'kp_22_y', 'kp_22_z', 'kp_23_x','kp_23_y','kp_23_z',
                        'kp_24_x', 'kp_24_y', 'kp_24_z']
    
    return pd.DataFrame(json_files, columns=columns)
            

def knn_classifier(path_to_data):
    data = pd.read_csv(path_to_data)
    y = data['occlusion'].values
    X = data.drop(columns=['occlusion'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    
    return knn

def test_dir(path_to_dir, knn):
    json_dir = os.listdir(path_to_dir)

    for f in json_dir:
        with open(path_to_dir + f, 'r') as js:
            data = json.load(js)
        data = data['keypoints']
        data = np.array(data).reshape(1, -1)

        print(knn.predict(data))

def test_file(path_to_dir, knn):
 
    with open(path_to_dir , 'r') as js:
        data = json.load(js)
    data = data['keypoints']
    data = np.array(data).reshape(1, -1)

    print(knn.predict(data))



if __name__ == '__main__': 
    #create_dataset_knn(20)
    
    '''
    csv_frame = create_csv_knn()
    csv_frame.to_csv('dataset/occlusion_data.csv', index=False)
    '''
    
    knn = knn_classifier('dataset/occlusion_data.csv')
    test_dir('dataset/json/test/', knn)
    
    '''
    with open('dataset/json/no_occlusion0.json', 'r') as f:
        data = json.load(f)
        kp = data['keypoints']
    kp = np.array(kp).reshape(1,-1)
    print(knn.predict(kp))
    '''
