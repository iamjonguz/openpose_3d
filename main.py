
'''

Main file for the system
Following commands are available...
...
...
...

USE ARGPARSER TO DO THIS

1. Run standard Openpose
2. Run live 3D-estimation
3. Record video sequence & create pose data
4. Create training data for KNN
    - Record a sequence that you can label as occluded or nonoccluded
5. Run recorded demo
    - Openpose not required

'''

import sys


arg = sys.argv[1]
print(arg)

if arg == 'openpose_2d':
    import openpose_2d
elif arg == 'openpose_3d':
    from openpose_3d import OpenPose
    from csv_scripts.json_to_csv import create_csv
    from animation.pyqt import animation
    op = OpenPose()

    try:
        op.record_estimation_sequence()
    except:
        print('Recording stopped.')
        create_csv()
        animation()
elif arg == 'animation':
    from animation.pyqt import animation
    animation()

