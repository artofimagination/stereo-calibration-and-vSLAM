# Stereo camera calibration and vSLAM
Repo to produce stereo depth map, do mapping and navigation

# Content
* [Stereo Calibration tab](#stereo-calibration-tab)<br>
* [Stereo Calibration tips](#stereo-calibration-tips)<br>
* [Block matching tab](#block-matching-tab)<br>
* [Block matching tips](#block-matching-tips)<br>
* [Point cloud visualization](#point-cloud-visualization)<br>
* [Calibration and block matching links](#sources-for-camera-calibration-and-depth-map-generation)<br>
* [Feature detection tab](#feature-detection-tab)<br>
* [Feature detection links](#sources-for-feature-detection-and-matching)<br>

# Intro
This repo is heavily under development.

At the moment only stereo camera calibration, depth map block matching configuration and feature detection is implemented.<br>
Added a small UI that allows generating calibration images and configuring block matching and feature detection parameters in a comfortable way.

Any contribution is appreciated.

## Install
The app was tested on Ubuntu 20.04 with Python 3.8. Can't guarantie it works on windows.<br>
To install dependencies run<br> 
```pip install -r requirements.txt```<br>
```sudo apt install python3-pyqt5.qtopengl```<br>
To run the app run ```python main.py```<br>


# Generic UI features
* **Save settings** - saves all UI control values into a settings npz file. Also saves feature specific info and files to locations those name is identical to the settings name. Saving the settings will generate a lastSaved folder/folder as well. this is useful for **Default load on startup**
* **Load settings** - loads saved UI control values and additional data/files
* **Default load on startup** - when the applciation starts it will immediately load the latest saved settings.
* **Swap lenses** - each tab has this feature. It allows the user to swap left and right cameras in case the app recognized them in the wrong order
* **Select camera indices** - each tab has this feature. It allows the user to custom select a camera device. It will exit the app if left and right indices are the same. Useful, when there are more than two cameras connected to the system.

# Stereo Calibration tab
It allows the user to generate chessboard calibration images and create the calibrate the stereo camera for later depth map streaming.
For this project I bought two Logitec C270 webcameras and glued them in a solidish cardboard box.

![calibration](https://github.com/artofimagination/stereo-vSLAM/blob/master/resources/ReadmeImg2.png)

## UI use
### Buttons
"**Start**" - when the application is started no use mode is setup. To start video streaming in calibration mode, press 'Start'<br>
"**Take image**" - to generate calibration images, the need to be capture. Press 'n' or 'Take image' to store an image showing the chessboard pattern<br>
"**Process**" - when enough chessboard images are taken, by pressing 'Process' it will start calibrating the cameras and create the final calibration data<br>

### Special modes
Taking teh calibration images can happen in two ways<br>
"**Simple mode**" - in this case, the user just presses 'Take image'/'n' until enough images are created.<br>
"**Advanced mode**" - in this case, whenever a chessboard image is capture, the code will run stereo calibration on the existing set of images and will generate the rms. This basically shows, how well the 3D points are matching the 2D points during the projection. If it is over a threshold the image pair wil be thrown away.<br>
As more images are taken, the rms, may increases and the fix RMS threshold makes it impossible to take new images, for this reason I added an increment of threshold every time, when an image is taken successfully. In order to stop getting the RMS out of control, there is a max limit the RMS can be increased to.
 * **Parameters:**<br>
      - _calib_image_index_ - index of the last taken image. It is useful, when the calibration image generation is interrupted (the app is closed) and the user wants to continue where he left.
      - _rms_limit_ - represents the current RMS limit, if the calibration rms is larger than this, when a new image pair is taken, the pair is discarded
      - _increment_ - represents the step of RMS incerement after each successfull image pairs
      - _max_rms_ - represents the maximum allowed RMS

"**Ignore existing image data**" - ignores the existing chessboard.npz information when trying to calibrate the sensors. Note: the existing calibration images are not deleted or ignored, only the generated chessboard cornes data, image, object points.

### Resolution
Performance can be quite different with different resolution, hence I added the feature of changing it. Note: Blok Matching resolution MUST be the same as the calibration for appropriate results.

### Save/Load settings
When settigns are saved, also the calibration images are stored in calibImages folder in the appropriate left/right sensor folder under the folder name identical to the settings name.

# Stereo calibration tips
First thing first, I am NOT an image/video processing specialist, there are more advanced people to tell you the right answer, see **Sources for camera calibration and depth map generation**. This project is just a tiny Minimum Viable Product for a larger project I try to learn more about.<br>
My tips are based on personal experience.
 * If you have a DIY stereo camera, make sure, they are aligned and focused as much as possible. During my first attempt, one camera had a different pitch and I was very surprised when the depth map was a nonsense.
 * If you choose the baseline distance (distance between the lenses) too small, it will not have a too good depth detection (To large isn't good either). I had first some 60 mm, which seemed to be a bit small, so I cahnged it to 75 in the final design. 
 * It is very important to have a very accurate chessboard pattern. I think it cannot be stressed enough. I had many failed attempts, because there were slight bumps in my paper made pattern stick on the wall. Probably a solid material pattern is preferable.
 * Do not use square shaped pattern, column and row count shall not be equal
 * Probably less of a concern, but have a fix stand of your camera when taking pictures, shakyness can be problematic, especially if your cameras are not synced, like mine.
 * Make sure the left and right cameras are not swapped, use the UI to swap them if they are.
 * According to the wise people on the internet RMS value below 0.5 is acceptable. I tried to set it between 0.1-0.3
 * In **Advanced mode** I usually set 0.13-0.15 start RMS limit, with 0.005 increment and max RMS of 0.27-0.3
 * Take many pictures, from different angles and different distances. I have got low RMS images sets in the range of 300 mm - 2000 mm of distance from teh chessboard pattern. I usually took 30-50 images

# Block matching tab
This tab allows the user to run block matching on the stereo image stream and generate depth map for further use. It has a UI interface that helps to configure the BM for better results.

![block matching](https://github.com/artofimagination/stereo-vSLAM/blob/master/resources/ReadmeImg1.png)

## UI use
### Block matching configurator
The majority of the following description is copied from this [OpenCV answer](https://answers.opencv.org/question/182049/pythonstereo-disparity-quality-problems/)

**Block matching type** - the app supports configuring both Block Matching (BM) and Semi-Global Block Matching (SGBM), check the links below if you want to learn more about each, also google is your friend
 * **Common Parameters**
      - _Minimum disparity_ - is the smallest disparity value to search for. Use smaller values to look for scenes which include objects at infinity, and larger values for scenes near the cameras. Negative minDisparity can be useful if the cameras are intentionally cross-eyed, but you wish to calculate long-range distances. Setting this to a sensible value reduces interference from out of scene areas and reduces unneeded computation.
      - _Number of disparity_ - is the range of disparities to search over. It is effectively your scene's depth of field setting. Use smaller values for relatively shallow depth of field scenes, and large values for deep depth-of-field scenes. Setting this to a sensible value reduces interference from out of scene areas and reduces unneeded computation.
      - _Block size_ - is the dimension (in pixels on a side) of the block which is compared between the left and right images. Setting this to a sensible value reduces interference from out of scene areas and reduces unneeded computation.
      - _Uniqueness ratio_ - used in filtering the disparity map before returning to reject small blocks. May reduce noise.
      - _Speckle windows size, speckle range, disparity 12 max diff_ - used in filtering the disparity map before returning, looking for areas of similar disparity (small areas will be assumed to be noise and marked as having invalid depth information). These reduces noise in disparity map output.
 * **BM parameters**
      - _Texture threshold_ - used in filtering the disparity map before returning. May reduce noise.
      - _prefilter type/size/cap_ - used in filtering the input images before disparity computation. These may improve noise rejection in input images.
      - _smallerBlockSize_ - in theory this should reduce noise, but I couldn't produce any effect
 * **SGBM parameters**
      - _P1, P2_ - used in filtering the disparity map before returning to reject small blocks. May reduce noise.

### Point cloud visualization
I've got two solutions
 * **Using OpenGL (pyqt compatible)**<br>
 It is a more sophisicated tool which allows high performance rendering. See the [original example](https://github.com/SoonminHwang/py-pointcloud-viewer). I adapted it [here](https://github.com/artofimagination/stereo-calibration-and-vSLAM/blob/master/componentExamples/main_pointCloudGLVisualizer.py) to draw depthmap numpy arrays.
 * **Using pyqtgraphs**<br>
It is a simple visualizer tool, that requires the depth map as an input and will generate a 3D point cloud representation. Check the original [original scatterplot example](https://gist.github.com/markjay4k/da2f55e28514be7160a7c5fbf95bd243).<br>
There is also a [minimum example](https://github.com/artofimagination/stereo-calibration-and-vSLAM/blob/master/componentExamples/main_pointCloudVisualizer.py) if you are only interested in this widget and not the whole app. It is pyqt compatible.<br>
Note: it uses pyqtgraph scatterplot. It is not a very fast way to represent, but good for initial development.
 * **Parameters**
      - _fov_ - sets the field of view
      - _samplingRatio_ - allows the user to change how many points to show. For example selecting 10, will show every 10th point only. Improves performance. For me > 200 settings was reasonable fast.

### Resolution
Performance can be quite different with different resolution, hence I added the feature of changing it. Note: Blok Matching resolution MUST be the same as the calibration for appropriate results.

# Block matching tips
Since I am not really knowledgable in the area I basically did a lots of trial and error. tweaked each parameter to see what they result in the depth map. I feel, that if you get the calibration right, then block matching configuration with default values, will already show promosiing values.

* My config was working the best with
  - SGBM
  - Min disparity: 7
  - Num disparity: 112
  - Block size: 5
  - Uniqueness ratio: 4
  - Speckle window size: 1
  - Speckle range: 1
  - Disp12MaxDiff: 1
  - P1: 99
  - P2: 999

* Make sure the left and right cameras are not swapped, use the UI to swap them if they are.
* I noticed that P1, and P2 had a major effect on the speckle reduction.
* Check the link below, those guys had different setups. My values are generally an average of the other documented attempts
* If you get nonsense or extreme amount of speckle and very little depth map with roughly similar config, then most likely your lenses are not well aligned or your chesspattern is not the most accurate


# Sources for camera calibration and depth map generation<br>
https://github.com/SoonminHwang/py-pointcloud-viewer<br>
https://github.com/mmatl/pyrender/issues/14<br>
https://answers.opencv.org/question/182049/pythonstereo-disparity-quality-problems/<br>
https://becominghuman.ai/stereo-3d-reconstruction-with-opencv-using-an-iphone-camera-part-i-c013907d1ab5<br>
https://github.com/OmarPadierna/3DReconstruction<br>
https://learnopencv.com/depth-perception-using-stereo-camera-python-c/<br>
https://gist.github.com/aarmea/629e59ac7b640a60340145809b1c9013#file-2-calibrate-py<br>

# Feature detection tab
In order to get visual odometry working, features have to be detected and matched on two consecutive images. This allows to calculate navigation data based on only a few points of the images instead of every single point.<br>
Again not an expert here. I copy pasted and adapted most of the cde [from here](https://github.com/FoamoftheSea/KITTI_visual_odometry). Note: This repo is a great tutorial for overall stereo calibration and vSLAM.<br>
The tab shows the matches of two images captured by the same lense at different time.

![feature detection](https://github.com/artofimagination/stereo-vSLAM/blob/master/resources/ReadmeImg3.png)

## UI use
There are only a few parameters present. In order to connect to the cameras and produce the matching, press **Start**
 * **Parameters**
     - _Feature detector_ - allows the user to change the detector algorithm between sift, orb and surf. Note surf is not supported with BF matcher in this app. It will terminate with an exception.
     - _Feature matcher_ - allows the user to change the detector matching algorithm ("BF" Brute force and FLANN)

# Sources for feature detection and matching
https://www.youtube.com/watch?v=2GJuEIh4xGo<br>
https://github.com/FoamoftheSea/KITTI_visual_odometry<br>
https://scialert.net/fulltext/?doi=itj.2009.250.262<br>
