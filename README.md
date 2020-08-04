This is the repo for my bachelor thesis.

<b>Short description</b></br>
Openpose is a system for estimating human keypoints from video streams, pictures etc.

This thesis investigates the possibilities for how you can with the help of a 3D camera, in this case Intel Realsense create a system that can estimate human poses in 3D.

Approach:
1. Estimate keypoints in 2D using Openpose.
2. Use those keypoints together with depth values retrived from the 3D camera to get the depth of each keypoints. 
  1. Retrieve raw 3D data from the 3D camera.
  2. Use k-nearest neighbours to figure out if the current pose is self-occluded or not.
    - If self-occluded, depth data from some keypoints most be fixed. To do so there is a table containing valid combinations of depth data.
      The system will try to retrieve the best matching combination.
    - If no self-occlusion the system can just use raw data from the camera and OpenPose. 
3. Each estimated frame will be saved as a json file together with frame image. 
4. There is a script that can be run to process the recorded data. It will apply a rolling median filter to remove jitter.

For this code to able to run you have to have Openpose built on your computer. I used this tutorial:
https://www.youtube.com/watch?v=QC9GTb6Wsb4&feature=youtu.be

Once it is installed this code should be placed in the folder "openpose/build/examples/tutorial_api_python".

This is just a temporary solution and will hopefully be fixed in the future. 

What is needed from Openpose is estimated keypoints, therefor, any libabry built for that purpose should be easy to integrate with this. 

There is a recorded demo that can be used without having the OpenPose system installed. 
To run it, clone this repo and type "python main.py demo". This will run a recorded and processed demo and show the result.


