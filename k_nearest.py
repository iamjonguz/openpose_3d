
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
        
    with open(f'dataset/json/no_occlusion{frame_number}.json', 'w', encoding='utf-8') as output:
        json.dump(data, output, ensure_ascii=False, indent=4)

    return data

def create_dataset_knn(num_of_frames):
    op = OpenPose()
    time.sleep(3)
    
    frame_number = 0
    for i in range(num_of_frames):
        keypoints, image = op.estimate_3d_picture(use_table_data=False)
        cv2.imwrite(f'dataset/img/occlusion{frame_number}.jpg', image)
        create_json_knn(keypoints['keypoints'], frame_number, False)
        frame_number += 1
        

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
            

def classifier():
    data = pd.read_csv('dataset/occlusion_data.csv')
    y = data['occlusion'].values
    X = data.drop(columns=['occlusion'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    return knn



if __name__ == '__main__': 
    #create_dataset_knn(1)
    #csv_frame = create_csv_knn()
    #csv_frame.to_csv('dataset/occlusion_data.csv', index=False)
    #create_dataset_knn()
    classifier()
    



'''

    # Should be false
    test1 = np.array([364.6435241699219,61.863250732421875,1.571000099182129,352.913818359375,157.1379852294922,1.5320000648498535,286.2947998046875,158.41360473632812,1.5470000505447388,248.4745635986328,256.3169860839844,1.5430001020431519,223.7196807861328,359.4208679199219,1.4300000667572021,424.6938171386719,158.40370178222656,1.5670000314712524,453.42010498046875,257.6430969238281,1.5410001277923584,475.5384826660156,367.2441101074219,1.377000093460083,354.21484375,369.8530578613281,1.3250000476837158,308.53875732421875,368.55133056640625,1.3470001220703125,0.0,0.0,0.0,0.0,0.0,0.0,399.8479309082031,368.55474853515625,1.3420000076293945,0.0,0.0,0.0,0.0,0.0,0.0,350.23065185546875,57.92469024658203,1.5890001058578491,375.09649658203125,57.94723129272461,1.5940001010894775,329.3614501953125,82.7198486328125,1.6380001306533813,389.400634765625,78.86785125732422,1.65500009059906,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    test1 = test1.reshape(1,-1)
    print(knn.predict(test1))

    # Should be true
    test2 = np.array([351.6131591796875,77.47216796875,1.5540000200271606,347.6965026855469,159.75082397460938,1.5390000343322754,286.3214416503906,157.1085968017578,1.537000060081482,256.3587951660156,180.62413024902344,0.0,269.352783203125,123.19905090332031,1.0170000791549683,409.0121765136719,162.31268310546875,1.4930000305175781,433.7969970703125,187.12857055664062,1.252000093460083,442.8963928222656,150.56655883789062,1.0180000066757202,341.18072509765625,362.048583984375,1.2630000114440918,296.7549743652344,362.0479431152344,1.2940000295639038,0.0,0.0,0.0,0.0,0.0,0.0,390.73974609375,367.25958251953125,1.284000039100647,0.0,0.0,0.0,0.0,0.0,0.0,338.58209228515625,67.02762603759766,1.5600000619888306,362.03094482421875,65.72915649414062,1.5630000829696655,317.6455383300781,78.7811508178711,1.6150001287460327,380.240966796875,70.98588562011719,1.6010000705718994,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    test2 = test2.reshape(1, -1)
    print(knn.predict(test2))


    # true
    test3 = np.array([0.0,0.0,0.0,338.4911193847656,164.980712890625,1.6150001287460327,397.263427734375,166.27297973632812,1.6030000448226929,0.0,0.0,0.0,0.0,0.0,0.0,275.846923828125,164.9821319580078,1.6050000190734863,206.7079620361328,161.02963256835938,1.3420000076293945,298.1108093261719,97.05966186523438,1.2370001077651978,339.8497009277344,358.11578369140625,1.3360000848770142,382.9352722167969,358.1142883300781,1.3690000772476196,392.06829833984375,476.9074401855469,1.3300000429153442,0.0,0.0,0.0,296.74786376953125,355.5003967285156,1.377000093460083,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,372.47613525390625,99.68994140625,1.2100000381469727,312.4356994628906,95.79426574707031,1.2300000190734863,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    test3 = test3.reshape(1, -1)
    print(knn.predict(test3))

    # false
    test4 = np.array([362.0655517578125,63.159889221191406,1.5630000829696655,354.20001220703125,161.06321716308594,1.5500000715255737,284.9996337890625,159.75460815429688,1.5690001249313354,245.88150024414062,256.34112548828125,1.5240000486373901,211.93812561035156,358.0848083496094,1.4030001163482666,424.6795654296875,162.3463134765625,1.5830000638961792,453.4302978515625,256.3169860839844,1.5560001134872437,483.43304443359375,359.3943786621094,1.3820000886917114,351.6036376953125,371.1614990234375,1.315000057220459,307.2152099609375,369.8698425292969,1.3440001010894775,0.0,0.0,0.0,0.0,0.0,0.0,395.9596252441406,371.164794921875,1.349000096321106,0.0,0.0,0.0,0.0,0.0,0.0,349.02178955078125,56.57143020629883,1.5740001201629639,373.77923583984375,56.62651824951172,1.5870001316070557,328.08355712890625,78.81453704833984,1.6290000677108765,389.462646484375,76.15839385986328,1.65500009059906,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    test4 = test4.reshape(1, -1)
    print(knn.predict(test4))

    with open('dataset/json/no_occlusion0.json', 'r') as f:
        js = json.load(f)
        test5 = (js['keypoints'])

    test5 = np.array(test5).reshape(1, -1)
    print(knn.predict(test5))



'''