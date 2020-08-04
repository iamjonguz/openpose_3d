
import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../')


import json
import pyqtgraph as pg

import cv2 

from cam import VideoRecorder

import time

from openpose_3d import OpenPose
from csv_scripts.pose_from_csv import poses

class GraphDrawer(pg.GraphicsWindow):

    def __init__(self, from_file, path_to_csv, img_path):
        '''
        Will initiate the graph and draw the first frame
        '''
        
        super(GraphDrawer, self).__init__(title='Pose estimation')

        self.resize(1000,800)

        self.frame_number = 0
        self.from_file = from_file
        self.poses = poses(path_to_csv)
        self.img_path = img_path

        if not self.from_file:
            try:
                self.op = OpenPose()
            except RuntimeError:
                print('Connect a camera!')

        # Keypoint connections
        self.connection = [
            [4, 3], [3, 2], [2, 1], [7, 6], [6, 5],
            [5, 1], [17, 15], [15, 0], [0, 16], [16, 18],
            [0, 1], [1, 8], [23, 22], [22, 11], [11, 24],
            [11, 10], [10, 9], [9, 8], [20, 19], [19, 14],
            [14, 21], [14, 13], [13, 12], [12, 8]
        ]

        # ------------------------------------------ 3D graph ------------------------------------------
        self.window = gl.GLViewWidget()
        
        # Uncomment to see the axes
        
        '''
        gx = gl.GLGridItem()
        gy = gl.GLGridItem()
        gz = gl.GLGridItem()

        gx.rotate(90, 0, 1, 0)
        gy.rotate(90, 1, 0, 0)
      
        gx.translate(-10, 0, 0) 
        gy.translate(0, -10, 0) 
        gz.translate(0, 0, -10)

        self.window.addItem(gx) 
        self.window.addItem(gy) 
        self.window.addItem(gz) 
        '''
 
        # Keypoints either loaded from file, or straight from the camera
        if self.from_file:
            keypoints = self.poses[self.frame_number]
        else:
            keypoints = self.op.estimate_3d_picture()

        self.points = gl.GLScatterPlotItem(
                pos=keypoints,
                color=pg.glColor((0, 255, 0)),
                size=10
            )

        self.window.addItem(self.points)
        self.lines = {}
        for n, pts in enumerate(self.connection):
            try:
                self.lines[n] = gl.GLLinePlotItem(
                    pos=np.array([keypoints[pts[0]], keypoints[pts[1]]]),
                    color=self.get_color(n), 
                    width=2,
                    antialias= True
                )
                self.window.addItem(self.lines[n])
            except IndexError:
                print('That points was not found.')    


        # ------------------------------------------ Image printing ------------------------------------------
        img = cv2.imread(self.img_path + f'img{self.frame_number}.jpg')
              
        self.p1 = pg.ImageItem()
        self.p1.setImage(img)
        vb = pg.ViewBox()
        vb.addItem(self.p1)

        gv = pg.GraphicsView()
        gv.setCentralItem(vb)

        layoutgb = QtGui.QGridLayout()
        self.setLayout(layoutgb)
        layoutgb.addWidget(gv, 0, 0, -1, 1)
        layoutgb.addWidget(self.window, 0, 1, 1, 1)

        gv.sizeHint = lambda: pg.QtCore.QSize(100, 100)
        self.window.sizeHint = lambda: pg.QtCore.QSize(100, 100)
        self.window.setSizePolicy(gv.sizePolicy()) 
        self.window.setCameraPosition(distance=15, elevation=20)


    def update(self):
        self.frame_number +=1

        img = cv2.imread(self.img_path + f'img{self.frame_number}.jpg')
        self.p1.setImage(img)

        try: 
            if self.from_file:
                time.sleep(0.05)
                keypoints = self.poses[self.frame_number]
            else:
                keypoints = self.op.estimate_3d_picture()

            # All the keypoints that is not found will be set to none instead of 0. 
            # This will make it so that the bad keypoints are not printed
            for i, p in enumerate(keypoints):
                if p[2] > -0.2:
                    keypoints[i] = None

            self.points.setData(pos=keypoints)

            # Creating lines between keypoints
            for n, pts in enumerate(self.connection):
                tmp = []
                for p in pts:  
                    tmp.append(keypoints[p])
                        
                self.lines[n].setData(
                    pos=np.array(tmp)
                )

        except IndexError:
            self.exit_program()

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec()

    def animation(self, frametime=10):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(frametime)
        self.start()

    def get_color(self, connection):
        '''
        Returns various colors for the different keypoints
        '''
        if connection == 0: return pg.glColor(235, 235, 52)
        elif connection == 1: return pg.glColor(235, 174, 52)
        elif connection == 2: return pg.glColor(235, 128, 52)
        elif connection == 3: return pg.glColor(159, 235, 52)
        elif connection == 4: return pg.glColor(119, 235, 52)
        elif connection == 5: return pg.glColor(192, 235, 52)
        elif connection == 6: return pg.glColor(235, 52, 204)
        elif connection == 7: return pg.glColor(235, 52, 122)
        elif connection == 8: return pg.glColor(168, 52, 235)
        elif connection == 9: return pg.glColor(122, 52, 235)
        elif connection == 10: return pg.glColor(235, 52, 79)
        elif connection == 11: return pg.glColor(235, 52, 52)
        elif connection == 12: return pg.glColor(52, 235, 198)
        elif connection == 13: return pg.glColor(52, 232, 235)
        elif connection == 14: return pg.glColor(52, 205, 235)
        elif connection == 15: return pg.glColor(52, 186, 235)
        elif connection == 16: return pg.glColor(52, 235, 192)
        elif connection == 17: return pg.glColor(52, 235, 162)
        elif connection == 18: return pg.glColor(52, 153, 235)
        elif connection == 19: return pg.glColor(52, 128, 235)
        elif connection == 20: return pg.glColor(52, 104, 235)
        elif connection == 21: return pg.glColor(52, 79, 235)
        elif connection == 22: return pg.glColor(52, 52, 235)
        elif connection == 23: return pg.glColor(92, 52, 235)
        else: return pg.glColor(0, 0, 255)

    def exit_program(self):
        if not self.from_file:
            self.op.vr.pipe.stop()
        sys.exit(0)

def animation(demo=False, csv_path='/../data/csv_files/rolling_mean.csv', img_path='/../data/recorded_sequences/'):
    app = QtGui.QApplication(sys.argv)
    if demo:
        csv_path = '/../data/demo/rolling_mean.csv'
        img_path = '/../data/demo/'
    g = GraphDrawer(True, dir_path + csv_path, dir_path + img_path)
    
    try:
        g.animation()
    finally:
        g.exit_program()
    

def one_pic():
    app = QtGui.QApplication(sys.argv) 
    g = GraphDrawer(True, dir_path + '/../data/csv_files/original.csv')
    g.start()    



if __name__ == "__main__":
    one_pic()