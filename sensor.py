import sys

import cv2
import numpy as np
from calibration import CALIBRATION_RESULT, CAMERA_WIDTH, CAMERA_HEIGHT


def drawLines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, _ = img1.shape
    # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


class Sensor():
    REMAP_INTERPOLATION = cv2.INTER_LINEAR
    DEPTH_VISUALIZATION_SCALE = 2048

    def __init__(self):
        self.running = False

        # Stereo block matching parameters
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
        self.left = None
        self.right = None

    def startSensors(self):
        print("Starting sensors...")
        if self.running is False:
            indices = []
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f'Camera index available: {i}')
                    indices.append(i)
                cap.release()

            if len(indices) < 2:
                print("Not all sensors are accessible")
                sys.exit(1)
            self.left = cv2.VideoCapture(indices[0])
            self.right = cv2.VideoCapture(indices[1])

            # Increase the resolution
            self.left.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

            # Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
            self.left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.right.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.running = True
            print("Sensors running...")
        else:
            print("Sensors already running...")

    def captureFrame(self):
        # Grab both frames first,
        # then retrieve to minimize latency between cameras
        if not self.left.grab() or not self.right.grab():
            print("No more frames")
            return (None, None, None)

        _, leftFrame = self.left.retrieve()
        _, rightFrame = self.right.retrieve()
        return (leftFrame, rightFrame)

    def loadCalibFile(self):
        try:
            calibration = np.load(CALIBRATION_RESULT, allow_pickle=False)
        except IOError as e:
            raise Exception(
                  f"Calibration file is missing. \
Please run calibration first: {e}")
        return calibration

    def openCVShow(self):
        calibration = self.loadCalibFile()
        while True:
            (leftFrame, rightFrame, depthMap) =\
                self.createDepthMap(calibration)
            numpy_horizontal_concat = np.concatenate(
                (cv2.resize(leftFrame, (640, 480)),
                 cv2.resize(rightFrame, (640, 480))),
                axis=1)
            cv2.imshow('Stereo image', numpy_horizontal_concat)
            cv2.imshow('Depth image', cv2.resize(depthMap, (640, 480)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.left.release()
        self.right.release()
        cv2.destroyAllWindows()

    def createDepthMap(self, calibration):
        imageSize = tuple(calibration["imageSize"])
        leftMapX = calibration["leftMapX"]
        leftMapY = calibration["leftMapY"]
        leftROI = tuple(calibration["leftROI"])
        rightMapX = calibration["rightMapX"]
        rightMapY = calibration["rightMapY"]
        rightROI = tuple(calibration["rightROI"])

        # TODO: Why these values in particular?
        # TODO: Try applying brightness/contrast/gamma
        # adjustments to the images
        stereoMatcher = cv2.StereoBM_create()
        stereoMatcher.setROI1(leftROI)
        stereoMatcher.setROI2(rightROI)

        stereoMatcher.setPreFilterType(self.preFilterType)
        stereoMatcher.setPreFilterSize(self.preFilterSize)
        stereoMatcher.setPreFilterCap(self.preFilterCap)

        stereoMatcher.setMinDisparity(self.min_disp)
        stereoMatcher.setNumDisparities(self.num_disp)
        stereoMatcher.setBlockSize(self.blockSize)

        stereoMatcher.setSpeckleRange(self.speckleRange)
        stereoMatcher.setSpeckleWindowSize(self.speckleWindowSize)
        stereoMatcher.setDisp12MaxDiff(self.disp12MaxDiff)

        stereoMatcher.setTextureThreshold(self.textureThreshold)
        stereoMatcher.setUniquenessRatio(self.uniquenessRatio)
        stereoMatcher.setSmallerBlockSize(self.smallerBlockSize)

        (leftFrame, rightFrame) = self.captureFrame()
        leftHeight, leftWidth = leftFrame.shape[:2]
        rightHeight, rightWidth = rightFrame.shape[:2]
        if (leftWidth, leftHeight) != imageSize:
            print("Left camera has different size than the calibration data")
            return (leftFrame, rightFrame, None)

        if (rightWidth, rightHeight) != imageSize:
            print("Right camera has different size than the calibration data")
            return (leftFrame, rightFrame, None)

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

        grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)

        # grayLeft =\
        #     cv2.fastNlMeansDenoising(
        #         grayLeft,
        #         dst=31,
        #         h=7,
        #         templateWindowSize=5,
        #         searchWindowSize=21)

        # grayRight =\
        #     cv2.fastNlMeansDenoising(
        #         grayRight,
        #         dst=31,
        #         h=7,
        #         templateWindowSize=5,
        #         searchWindowSize=21)

        # grayLeft =\
        #     cv2.medianBlur(grayLeft, 11)
        # grayRight =\
        #     cv2.medianBlur(grayRight, 11)

        depth = stereoMatcher.compute(grayLeft, grayRight)
        # depth =\
        #     cv2.medianBlur(depth, 5)
        depth =\
            cv2.GaussianBlur(depth, (9, 9), 0)
        depth = depth / self.DEPTH_VISUALIZATION_SCALE

        # Convert depth map to a format that can be accepted by the UI
        normalized_depth = cv2.normalize(
            depth,
            depth,
            0,
            255,
            norm_type=cv2.NORM_MINMAX)
        normalized_depth_color = cv2.cvtColor(
            normalized_depth.astype(np.uint8),
            cv2.COLOR_GRAY2RGB)

        return leftFrame, rightFrame, normalized_depth_color

    def calculateEpipolarLine(self, leftFrame, rightFrame):
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
            if m.distance < 0.8*n.distance:
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
        img5, img6 = drawLines(leftFrame, rightFrame, lines1, pts1, pts2)
        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = drawLines(rightFrame, leftFrame, lines2, pts2, pts1)
        return (leftFrame, rightFrame)
