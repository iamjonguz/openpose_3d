## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import time

class PhotoHandler:

    # Not working properly. How does wait_for_frames work??
    def record_video(self):

        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
            
        pipeline.start(config)
        
        try:
            video_seq = []
            while True:
                d = {}
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()

                color_frame = frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())
                
                # If not here the depth will not get calculated in draw_graphs somehow...    
                print(depth_frame.get_distance(200,200))
                
                d['col'] = color_image
                d['depth'] = depth_frame

                video_seq.append(d)

                cv2.waitKey(0)
            
        finally:
            pipeline.stop()
            return video_seq

    def take_picture(self):

        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)

        try:
            input('Press enter to take photo (4 seconds delay): ')
            time.sleep(4)

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # If not here the depth will not get calculated in draw_graphs somehow...    
            print(depth_frame.get_distance(200,200))
        
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

        finally:

            # Stop streaming
            pipeline.stop()
            return color_image, depth_frame

if __name__ == "__main__":
    ph = PhotoHandler()
    frames = ph.record_video()
    print(len(frames))
    