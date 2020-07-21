
import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
import sys

import json
import pyqtgraph as pg

from cam import VideoRecorder

import time

from openpose import OpenPose
from pose_from_csv import poses

class GraphDrawer:

    def __init__(self, from_file=False, path_to_csv=None):
        self.frame_number = 0
        self.from_file = from_file
        self.poses = poses(path_to_csv)
        self.app = QtGui.QApplication(sys.argv)
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('3D estimation')
        self.window.setGeometry(0, 110, 1200, 700)
        self.window.setCameraPosition(distance=30, elevation=12)
        self.window.show()

        try:
            self.op = OpenPose()
        except RuntimeError:
            print('Try to connect a device!')

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

        self.connection = [
            [4, 3], [3, 2], [2, 1], [7, 6], [6, 5],
            [5, 1], [17, 15], [15, 0], [0, 16], [16, 18],
            [0, 1], [1, 8], [23, 22], [22, 11], [11, 24],
            [11, 10], [10, 9], [9, 8], [20, 19], [19, 14],
            [14, 21], [14, 13], [13, 12], [12, 8]
        ]

 
        if self.from_file:
            verts = self.poses[self.frame_number]
        else:
            verts = self.op.estimate_3d_picture()

        for i, p in enumerate(verts):
            if p[2] > -0.2:
                verts[i] = None

        self.points = gl.GLScatterPlotItem(
                pos=verts,
                color=pg.glColor((0, 255, 0)),
                size=10
            )

        self.window.addItem(self.points)
        self.lines = {}
        for n, pts in enumerate(self.connection):
            try:
                self.lines[n] = gl.GLLinePlotItem(
                    pos=np.array([verts[pts[0]], verts[pts[1]]]),
                    color=self.get_color(n), 
                    width=2,
                    antialias= True
                )
                self.window.addItem(self.lines[n])
            except IndexError:
                print('That points was not found.')        

    def update(self):
        self.frame_number +=1

        try:
            if self.from_file:
                time.sleep(0.1)
                k = self.poses[self.frame_number]
            else:
                k = self.op.estimate_3d_picture()

            for i, p in enumerate(k):
                if p[2] > -0.2:
                    k[i] = None

            self.points.setData(pos=k)

            for n, pts in enumerate(self.connection):
                tmp = []
                for p in pts:  
                    tmp.append(k[p])
                        
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

    def picture(self):
        time.sleep(2)
        self.update()
        self.start()

    def get_color(self, connection):
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
        self.op.vr.pipe.stop()
        sys.exit(0)


if __name__ == "__main__":
    g = GraphDrawer(True, 'scripts/original.csv')
    g.start()
    '''
    try:
        g.animation()
    finally:
        g.exit_program()
    '''