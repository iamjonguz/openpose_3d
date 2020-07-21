This is the repo for my bachelor thesis.

<b>Short description</b></br>
Openpose is a system for estimating human keypoints from video streams, pictures etc.

This thesis investigates the possibilities for how you can with the help of a 3D camera, in this case Intel Realsense create a system that can estimate human poses in 3D.

Approach:
1. Estimate keypoints in 2D using Openpose.
2. Use those keypoints together with depth values retrived from the 3D camera to get the depth of each keypoints. 
3. Clean the data.
  - Remove noices using a rolling mean function. 
  - Match the depth frame with saved frames. These saved frames comes from a set of frames where bad data from self-occlusion has been replaced with estimated new
  frames that do not contain self-occlusions. 
4. Use this systen to create alot of 3D poses. These poses will be label by Openpose. 
5. Use this data to retrain the Openpose network (using transer learning) with the hope of getting it estimate 3D posese without the help of the 3D camera. 


For this code to able to run you have to have Openpose built on your computer. I used this tutorial:
https://www.youtube.com/watch?v=QC9GTb6Wsb4&feature=youtu.be

Once it is installed this code should be placed in the folder "openpose/build/examples/tutorial_api_python".

This is just a temporary solution and will hopefully be fixed in the future. 

What is needed from Openpose is estimated keypoints, therefor, any libabry built for that purpose should be easy to integrate with this. 
