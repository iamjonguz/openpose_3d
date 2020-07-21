import pyrealsense2 as rs
import numpy as np
import cv2

import time
import sys

class VideoRecorder:

    def __init__(self):
        self.pipe = rs.pipeline()
        self.config = rs.config()
        self.profile = self.pipe.start()     

    def record_frame(self, record=None):
        d = {}
        frames = self.pipe.wait_for_frames()
        d['col'] = np.asarray(frames.get_color_frame().get_data()) 
        d['depth'] = frames.get_depth_frame()
        
        return d['col'], d['depth']
        
if __name__ == "__main__":
    vr = VideoRecorder()
