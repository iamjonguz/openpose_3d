import json
import csv
from csv import reader
import os
import natsort
import pandas as pd
import scipy
from scipy import signal
import numpy as np

path_to_dir='../img'

def json_to_csv():
    json_list = []
    img_dir = os.listdir(path_to_dir)
    img_dir = list(filter(lambda x: x[-1] == 'n', img_dir))
    img_dir = natsort.natsorted(img_dir)
    for files in img_dir:
        if files.endswith('.json'):
            with open(f'../img/{files}', 'r') as f:
                flatten_dir = {}
                js = json.load(f)

                for kp in js['keypoints']:
                    flatten_dir[f'kp_{kp}_x'] = js['keypoints'][str(kp)][0]
                    flatten_dir[f'kp_{kp}_y'] = js['keypoints'][str(kp)][1]
                    flatten_dir[f'kp_{kp}_z'] = js['keypoints'][str(kp)][2]

                json_list.append(flatten_dir)

    columns = ['kp_0_x', 'kp_0_y', 'kp_0_z', 'kp_1_x','kp_1_y','kp_1_z',
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

    return pd.DataFrame(json_list, columns=columns)

def rolling_mean():

    with open('original.csv', 'r') as read_obj:

        csv_reader = pd.read_csv(read_obj)
        col = []

        for i in csv_reader:
            rolling_windows = csv_reader[i].rolling(10)
            col.append(rolling_windows.mean())

        a = list(zip(col[0], col[1], col[2],col[3],col[4],col[5],col[6],col[7],col[8],col[9],col[10],col[11],col[12],col[13],col[14],col[15],col[16],col[17],col[18],col[19],col[20],
                col[21], col[22],col[23],col[24],col[25],col[26],col[27],col[28],col[29],col[30],col[31],col[32],col[33],col[34],col[35],col[36],col[37],col[38],col[39],col[40],
                col[41], col[42],col[43],col[44],col[45],col[46],col[47],col[48],col[49],col[50],col[51],col[52],col[53],col[54],col[55],col[56],col[57],col[58],col[59],col[60],
                col[61], col[62],col[63],col[64],col[65],col[66],col[67],col[68],col[69],col[70],col[71],col[72],col[73],col[74]))

        columns = ['kp_0_x', 'kp_0_y', 'kp_0_z', 'kp_1_x','kp_1_y','kp_1_z',
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


        frame = pd.DataFrame(a, columns=columns)
        return frame

if __name__ == "__main__":    

    a = json_to_csv()
    a.to_csv('../scripts/original.csv', index=False)
    frame = rolling_mean()
    frame.to_csv('../scripts/rolling_mean.csv', index=False)

