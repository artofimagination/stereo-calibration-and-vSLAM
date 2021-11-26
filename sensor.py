import cv2
import numpy as np
from calibration import CALIBRATION_RESULT


def decompose_projection_matrix(p):
    '''
    Shortcut to use cv2.decomposeProjectionMatrix(),
    which only returns k, r, t, and divides
    t by the scale, then returns it as a vector with shape (3,) (non-homogeneous)

    Arguments:
    p -- projection matrix to be decomposed

    Returns:
    k, r, t -- intrinsic matrix, rotation matrix, and 3D translation vector

    '''
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = (t / t[3])[:3]

    return k, r, t


def generateDepthMapMask(depthImage, leftImage):
    firstNonMaxIndex = 0
    for i, pixel in enumerate(depthImage[4]):
        if pixel < depthImage.max():
            firstNonMaxIndex = i
            break
    mask = np.zeros(depthImage.shape[:2], dtype=np.uint8)
    ymax = depthImage.shape[0]
    xmax = depthImage.shape[1]
    cv2.rectangle(mask, (firstNonMaxIndex, 0), (xmax, ymax), (255), thickness=-1)
    return mask


# Draws the specified lines on the input images.
def drawLines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, _ = img1.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


# This class implements the camera sensor handling and image outputing
# It can output normal stream or the depth map after block matching
# There is also an option to draw the epipolar lines
# on the undistorted image stream
class Sensor():
    REMAP_INTERPOLATION = cv2.INTER_LINEAR

    def __init__(self):
        self.running = False

        # Stereo block matching parameters
        self.bmType = 0
        self.min_disp = 0
        self.num_disp = 128
        self.blockSize = 5
        self.uniquenessRatio = 1
        self.speckleWindowSize = 2
        self.speckleRange = 80
        self.disp12MaxDiff = 180
        self.preFilterCap = 63
        self.preFilterSize = 5
        self.preFilterType = 0
        self.smallerBlockSize = 0
        self.textureThreshold = 0
        self.drawEpipolar = False
        self.camera_width = 640
        self.camera_height = 480

        # List of video devices found in the OS (/dev/video*).
        self.sensor_indices = list()
        self.sensor_devices = list()
        # Left camera capture instance
        self.left = None
        # Right camera capture instance
        self.right = None
        # Left camera index, that refers to the appropriate video device.
        self.leftIndex = 2
        # Right camera index, that refers to the appropriate video device.
        self.rightIndex = 1

    # Detects all available video devices in the OS.
    def detectSensors(self):
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f'Camera index available: {i}')
                self.sensor_indices.append(i)
                self.sensor_devices.append(cap)

        if len(self.sensor_indices) < 2:
            raise Exception("Not all sensors are accessible")

    # Initializes and starts the sensors
    # If there are more than 2 sensors
    # make sure the camera indices belong to the correct sensor.
    def startSensors(self):
        print("Starting sensors...")
        if self.running is False:
            self.left = self.sensor_devices[self.leftIndex]
            self.right = self.sensor_devices[self.rightIndex]

            # Increase the resolution
            self.left.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.left.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.right.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.right.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

            # Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
            self.left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.right.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

            (left, right) = self.captureFrame()
            if left is None or right is None:
                raise Exception("No frames, quitting app. Try to start again,\
usually the camera modules are ready by the second attempt")

            self.running = True
            print(f"Resolution is {self.camera_width}x{self.camera_height}")
            print("Sensors running...")
        else:
            print("Sensors already running...")

    # Restart sensors.
    def restartSensors(self):
        print("Restarting sensors...")
        self.running = False
        self.releaseVideoDevices()
        self.detectSensors()
        self.startSensors()

    # Capture frame in an efficient way.
    def captureFrame(self):
        # Grab both frames first,
        # then retrieve to minimize latency between cameras
        if not self.left.grab() or not self.right.grab():
            print("No more frames")
            return (None, None)

        _, leftFrame = self.left.retrieve()
        _, rightFrame = self.right.retrieve()
        return (leftFrame, rightFrame)

    # Loads the calibration file generated in the calibration tab.
    def loadCalibFile(self):
        try:
            calibration = np.load(CALIBRATION_RESULT, allow_pickle=False)
        except IOError as e:
            raise Exception(
                  f"Calibration file is missing. \
Please run calibration first: {e}")
        return calibration

    def releaseVideoDevices(self):
        for dev in self.sensor_devices:
            dev.release()
        self.sensor_devices.clear()
        self.sensor_indices.clear()

    # Opens the frame stream with cv2 library calls (No Qt UI is used)
    def openCVShow(self):
        calibration = self.loadCalibFile()
        while True:
            (leftFrame, rightFrame, depthMap, _) =\
                self.createDepthMap(calibration)
            numpy_horizontal_concat = np.concatenate(
                (cv2.resize(leftFrame, (640, 480)),
                 cv2.resize(rightFrame, (640, 480))),
                axis=1)
            cv2.imshow('Stereo image', numpy_horizontal_concat)
            cv2.imshow('Depth image', cv2.resize(depthMap, (640, 480)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.releaseVideoDevices()
        cv2.destroyAllWindows()

    # Undistorts the camera images, using the calibration data.
    def undistort(self, calibration, leftFrame, rightFrame):
        leftMapX = calibration["leftMapX"]
        leftMapY = calibration["leftMapY"]
        rightMapX = calibration["rightMapX"]
        rightMapY = calibration["rightMapY"]
        fixedLeft = cv2.remap(
            leftFrame,
            leftMapX,
            leftMapY,
            self.REMAP_INTERPOLATION)
        fixedRight = cv2.remap(
            rightFrame,
            rightMapX,
            rightMapY,
            self.REMAP_INTERPOLATION)
        return fixedLeft, fixedRight

    # Generates depth map using block matching, or semi global block matching.
    # TODO move block matching in its own class for clarity.
    def createDepthMap(self, calibration, leftFrame, rightFrame):
        imageSize = tuple(calibration["imageSize"])
        leftROI = tuple(calibration["leftROI"])
        rightROI = tuple(calibration["rightROI"])

        # TODO: Why these values in particular?
        # TODO: Try applying brightness/contrast/gamma
        # adjustments to the images
        if self.bmType == 0:
            stereoMatcher = cv2.StereoBM_create()
            stereoMatcher.setROI1(leftROI)
            stereoMatcher.setROI2(rightROI)
            stereoMatcher.setPreFilterType(self.preFilterType)
            stereoMatcher.setPreFilterSize(self.preFilterSize)
            stereoMatcher.setPreFilterCap(self.preFilterCap)
            stereoMatcher.setSmallerBlockSize(self.smallerBlockSize)
            stereoMatcher.setTextureThreshold(self.textureThreshold)
        else:
            stereoMatcher = cv2.StereoSGBM_create()
            stereoMatcher.setP1(8 * 1 * self.blockSize ** 2)
            stereoMatcher.setP2(32 * 1 * self.blockSize ** 2)
            stereoMatcher.setMode(cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        stereoMatcher.setMinDisparity(self.min_disp)
        stereoMatcher.setNumDisparities(self.num_disp)
        stereoMatcher.setBlockSize(self.blockSize)

        stereoMatcher.setSpeckleRange(self.speckleRange)
        stereoMatcher.setSpeckleWindowSize(self.speckleWindowSize)
        stereoMatcher.setDisp12MaxDiff(self.disp12MaxDiff)
        stereoMatcher.setUniquenessRatio(self.uniquenessRatio)

        leftHeight, leftWidth = leftFrame.shape[:2]
        rightHeight, rightWidth = rightFrame.shape[:2]
        if (leftWidth, leftHeight) != imageSize:
            print("Left camera has different size than the calibration data")
            return (leftFrame, rightFrame, None)

        if (rightWidth, rightHeight) != imageSize:
            print("Right camera has different size than the calibration data")
            return (leftFrame, rightFrame, None)

        (leftFrame, rightFrame) = self.undistort(calibration, leftFrame, rightFrame)

        if self.drawEpipolar is True:
            (leftFrame, rightFrame) = self.calculateEpipolarLine(
                leftFrame, rightFrame)

        grayLeft = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)

        depth = stereoMatcher.compute(grayLeft, grayRight)
        # Adding some smoothing to increase image quality.
        depth = cv2.medianBlur(depth, 5).astype(np.float32) / 16
        k_left, _, t_left = decompose_projection_matrix(calibration["leftProjection"])
        _, _, t_right = decompose_projection_matrix(calibration["rightProjection"])
        f = k_left[0][0]

        # Calculate baseline of stereo pair
        b = t_right[0] - t_left[0]

        # Avoid instability and division by zero
        depth[depth == 0.0] = 0.1
        depth[depth == -1.0] = 0.1

        # Make empty depth map then fill with depth
        depth_map = np.ones(depth.shape)
        depth_map = f * b / depth

        return leftFrame, rightFrame, depth_map

    # Calculates the epippolar lines for visualization purposes.
    def calculateEpipolarLine(self, leftFrame, rightFrame):
        epipolarFrameLeft = leftFrame.copy()
        epipolarFrameRight = rightFrame.copy()
        # sift = cv2.SIFT() causes crash during calling detectAndCompute
        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(leftFrame, None)
        kp2, des2 = sift.detectAndCompute(rightFrame, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        pts1 = []
        pts2 = []

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
        # We select only inlier points
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

        # Find epilines corresponding to points in right image (second image)
        # and drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        img5, img6 = drawLines(
            epipolarFrameLeft, epipolarFrameRight, lines1, pts1, pts2)
        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = drawLines(
            epipolarFrameRight, epipolarFrameLeft, lines2, pts2, pts1)
        return (epipolarFrameLeft, epipolarFrameRight)
