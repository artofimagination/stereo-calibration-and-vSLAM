# stereo-vSLAM
Repo to produce stereo depth map and do mapping a navigation

This repo is heavily under development.

At the moment only stereo camera calibration, depth map block matching configuration is implemented.<br>
Added a small UI that allows generating calibration images and configuring block matching parameters in a comfortable way.

# Generic UI features
* **Save settings** - saves all UI control values into a settings npz file. Also saves feature specific info and files to locations those name is identical to the settings name. Saving the settings will generate a lastSaved folder/folder as well. this is useful for **Default load on startup**
* **Load settings** - loads saved UI control values and additional data/files
* **Default load on startup** - when the applciation starts it will immediately load the latest saved settings.

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
 * If you have a DIY stereo camera, make sure, they are aligned as much as possible. During my first attempt, one camera had a different pitch and I was very surprised when the depth map was a nonsense.
 * If you choose the baseline distance (distance between the lenses) too small, it will not have a too good depth detection (To large isn't good either). I had first some 60 mm, which seemed to be a bit small, so I cahnged it to 75 in the final design. 
 * It is very important to have a very accurate chessboard pattern. I think it cannot be stressed enough. I had many failed attempts, because there were slight bumps in my paper made pattern stick on the wall. Probably a solid material pattern is preferable.
 * Do not use square shaped pattern, column and row count shall not be equal
 * Probably less of a concern, but have a fix stand of your camera when taking pictures, shakyness can be problematic, especially if your cameras are not synced, like mine.
 * According to the wise people on the internet RMS value below 0.5 is acceptable. I tried to set it between 0.1-0.3
 * In **Advanced mode** I usually set 0.13-0.15 start RMS limit, with 0.005 increment and max RMS of 0.27-0.3
 * Take many pictures, from different angles and different distances. I have got low RMS images sets in the range of 300 mm - 2000 mm of distance from teh chessboard pattern. I usually took 30-50 images


# Sources for camera calibration and depth map generation:<br>
https://answers.opencv.org/question/182049/pythonstereo-disparity-quality-problems/<br>
https://becominghuman.ai/stereo-3d-reconstruction-with-opencv-using-an-iphone-camera-part-i-c013907d1ab5<br>
https://github.com/OmarPadierna/3DReconstruction<br>
https://learnopencv.com/depth-perception-using-stereo-camera-python-c/<br>
https://gist.github.com/aarmea/629e59ac7b640a60340145809b1c9013#file-2-calibrate-py<br>
