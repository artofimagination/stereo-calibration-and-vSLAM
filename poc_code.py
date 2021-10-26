# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# This is the original Proof of Concept code that I copy pasted
# from various sources
# For more info on sources read README.md
# This code may not function entirely and is only kept for reference
# and is being properly reimplemented in the other files.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

from gui import MainWindow

import cv2
import sys
import time
import os
import traceback
import random
from pathlib import Path

# Calib modules
import numpy as np
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
import matplotlib.pyplot as plt

# GUI
from PyQt5.QtWidgets\
    import\
        QApplication, QGridLayout, QPushButton, QLabel, QWidget, QMainWindow, QSpinBox, QHBoxLayout
from PyQt5.QtCore import QThread, QTimer, QObject, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore

CHESSBOARD_SIZE = (9, 6)
CHESSBOARD_OPTIONS = (cv2.CALIB_CB_ADAPTIVE_THRESH |
        cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)

OBJECT_POINT_ZERO = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3),
        np.float32)
OBJECT_POINT_ZERO[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0],
        0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

OPTIMIZE_ALPHA = 0.25

TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30,
        0.001)

MAX_IMAGES = 64

leftImageDir = "calibImages/left"
rightImageDir = "calibImages/right"
outputFile = "output/calibration.npz"

LEFT_PATH = "calibImages/left/{:06d}.jpg"
RIGHT_PATH = "calibImages/right/{:06d}.jpg"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


REMAP_INTERPOLATION = cv2.INTER_LINEAR

DEPTH_VISUALIZATION_SCALE = 2048

# Function that Downsamples image x number (reduce_factor) of times.
def downsample_image(image, reduce_factor):
    for i in range(0, reduce_factor):
        # Check if image is color or grayscale
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape

        image = cv2.pyrDown(image, dstsize=(col//2, row // 2))
    return image


# Function to create point cloud file
def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')


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


class StereoCameraSensor():
    RIGHT = 0
    LEFT = 1

    MAX_SYNC_BUFFER = 1000

    def __init__(self):
        self.instances = []
        self.syncDelay_frame = 1  # Set through trial and error
        self.syncFrameBuffer = []
        # Create buffer for left and right camera
        self.syncFrameBuffer.append([])
        self.syncFrameBuffer.append([])
        self.calibStep = 0
        self.global_rms = 100
        self.global_index = 23
        self.rms_limit = 0.315
        self.increment = 0.01
        self.turnaround_limit = 0.31
        self.trendUp = True


        # stereoMatcher.setMinDisparity(4)
        # stereoMatcher.setNumDisparities(128)
        # stereoMatcher.setBlockSize(21)
        # stereoMatcher.setSpeckleRange(16)
        # stereoMatcher.setSpeckleWindowSize(45)
        # self.min_disp = 20
        # self.max_disp = 80
        # self.blockSize = 29
        # self.uniquenessRatio = 1
        # self.speckleWindowSize = 1
        # self.speckleRange = 0
        # self.disp12MaxDiff = 0
        # self.preFilterCap = 4
        # self.preFilterSize = 13
        # self.preFilterType = 0
        # self.smallerBlockSize = 23
        # self.textureThreshold = 10

        self.min_disp = 0
        self.max_disp = 128
        self.blockSize = 13
        self.uniquenessRatio = 1
        self.speckleWindowSize = 2
        self.speckleRange = 80
        self.disp12MaxDiff = 180
        self.preFilterCap = 63
        self.preFilterSize = 5
        self.preFilterType = 0
        self.smallerBlockSize = 0
        self.textureThreshold = 0

        indices = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f'Camera index available: {i}')
                indices.append(i)
            cap.release()

        # if len(indices) < 3:
        #     raise Exception("Not all cameras are found")

        # cap = cv2.VideoCapture(indices[0])
        # # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # if cap.isOpened():
        #     print(f'Camera index available: {indices[0]}')
        #     self.instances.append(cap)
        # else:
        #     print("Cannot open left camera")
        #     return

        # cap = cv2.VideoCapture(indices[1])
        # # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # if cap.isOpened():
        #     print(f'Camera index available: {indices[1]}')
        #     self.instances.append(cap)
        # else:
        #     print("Cannot open right camera")
        #     self.releaseCameras()
        #     return

    def releaseCameras(self):
        for instance in self.instances:
            instance.release()

    def _syncCameras(self, indexOfCameraToDelay, indexOfCameraOnTime):
        syncBufferLength = len(self.syncFrameBuffer[indexOfCameraToDelay])
        if syncBufferLength > self.MAX_SYNC_BUFFER:
            self.syncFrameBuffer[self.LEFT].pop(0)
            self.syncFrameBuffer[self.RIGHT].pop(0)
            return None
        syncDelay = self.syncDelay_frame
        if syncDelay > self.MAX_SYNC_BUFFER:
            syncDelay = self.MAX_SYNC_BUFFER

        frame0 = None
        frame1 = None
        if syncBufferLength >= syncDelay:
            leftDelay = 0
            rightDelay = syncDelay - 1
            if indexOfCameraToDelay == self.LEFT:
                leftDelay = syncDelay - 1
                rightDelay = 0

            frame0 = self.syncFrameBuffer[self.LEFT].pop(leftDelay)
            frame1 = self.syncFrameBuffer[self.RIGHT].pop(rightDelay)
        return (frame0, frame1)

    def showFrame(self, inWindow=True):
        frames = []
        for index, instance in enumerate(self.instances):
            instance.grab()

        for index, instance in enumerate(self.instances):
            _, frame = instance.retrieve()
            self.syncFrameBuffer[index].append(frame)

        (frame0, frame1) = self._syncCameras(self.LEFT, self.RIGHT)
        if frame0 is not None:
            frames.append(frame0)
        if frame1 is not None:
            frames.append(frame1)

        if inWindow and frame0 is not None and frame1 is not None:
            numpy_horizontal_concat = np.concatenate((frame0, frame1), axis=1)
            cv2.imshow('Stereo image', numpy_horizontal_concat)

        return frames

    def _saveFrameToJpg(self, cameraIndex, index):
        _, frame = self.instances[cameraIndex].read()
        cv2.imwrite(f'calibImages/frame{cameraIndex}_{index}.jpg', frame)

    def createCalibImages(self):
        # TODO: Use more stable identifiers
        left = cv2.VideoCapture(0)
        right = cv2.VideoCapture(2)

        # Increase the resolution
        left.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        # Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
        left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        frameId = 0

        # Grab both frames first, then retrieve to minimize latency between cameras
        while(True):
            if not (left.grab() and right.grab()):
                print("No more frames")
                break

            _, leftFrame = left.retrieve()
            _, rightFrame = right.retrieve()

            if leftFrame is not None and rightFrame is not None:
                numpy_horizontal_concat = np.concatenate((cv2.resize(leftFrame, (640,480)), cv2.resize(rightFrame, (640,480))), axis=1)
                cv2.imshow('Stereo image', numpy_horizontal_concat)
                ret = cv2.waitKey(1)
                if ret & 0xFF == ord('n'):
                    print(f"Store image {LEFT_PATH.format(self.global_index)}")
                    print(f"Store image {RIGHT_PATH.format(self.global_index)}")
                    cv2.imwrite(LEFT_PATH.format(self.global_index), leftFrame)
                    cv2.imwrite(RIGHT_PATH.format(self.global_index), rightFrame)
                    self.global_index += 1
                    break
                if ret & 0xFF == ord('q'):
                    left.release()
                    right.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)

        left.release()
        right.release()
        cv2.destroyAllWindows()

    def generateCalibrationImages(self):
        maxCount = 5
        fileIndex = 0
        for index in range(maxCount):
            print(f"Align the pattern in the center\
 with a different angle on it ({maxCount}/{index}) and press 'n'")
            while(True):
                self.showFrame()
                if cv2.waitKey(1) & 0xFF == ord('n'):
                    self._saveFrameToJpg(self.RIGHT, fileIndex)
                    self._saveFrameToJpg(self.LEFT, fileIndex)
                    fileIndex += 1
                    break

        for index in range(maxCount):
            print(f"Align the pattern in the top left\
 with a different angle on it ({maxCount}/{index}) and press 'n'")
            while(True):
                self.showFrame()
                if cv2.waitKey(1) & 0xFF == ord('n'):
                    self._saveFrameToJpg(self.RIGHT, fileIndex)
                    self._saveFrameToJpg(self.LEFT, fileIndex)
                    fileIndex += 1
                    break

        for index in range(maxCount):
            print(f"Align the pattern in the top right\
 with a different angle on it ({maxCount}/{index}) and press 'n'")
            while(True):
                self.showFrame()
                if cv2.waitKey(1) & 0xFF == ord('n'):
                    self._saveFrameToJpg(self.RIGHT, fileIndex)
                    self._saveFrameToJpg(self.LEFT, fileIndex)
                    fileIndex += 1
                    break

        for index in range(maxCount):
            print(f"Align the pattern in the bottom left\
 with a different angle on it ({maxCount}/{index}) and press 'n'")
            while(True):
                self.showFrame()
                if cv2.waitKey(1) & 0xFF == ord('n'):
                    self._saveFrameToJpg(self.RIGHT, fileIndex)
                    self._saveFrameToJpg(self.LEFT, fileIndex)
                    fileIndex += 1
                    break

        for index in range(maxCount):
            print(f"Align the pattern in the bottom right\
 with a different angle on it ({maxCount}/{index}) and press 'n'")
            while(True):
                self.showFrame()
                if cv2.waitKey(1) & 0xFF == ord('n'):
                    self._saveFrameToJpg(self.RIGHT, fileIndex)
                    self._saveFrameToJpg(self.LEFT, fileIndex)
                    fileIndex += 1
                    break

        return True

    def readImagesAndFindChessboards(self, imageDirectory):
        cacheFile = "{0}/chessboards.npz".format(imageDirectory)
        try:
            cache = np.load(cacheFile)
            print("Loading image data from cache file at {0}".format(cacheFile))
            return (list(cache["filenames"]), [], list(cache["objectPoints"]),
                    list(cache["imagePoints"]), tuple(cache["imageSize"]))
        except IOError:
            print("Cache file at {0} not found".format(cacheFile))

        print("Reading images at {0}".format(imageDirectory))
        imagePaths = glob.glob("{0}/*.jpg".format(imageDirectory))

        filenames = []
        objectPoints = []
        imagePoints = []
        imageSize = None

        for imagePath in sorted(imagePaths):
            image = cv2.imread(imagePath)
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            newSize = grayImage.shape[::-1]
            if imageSize is not None and newSize != imageSize:
                raise ValueError(
                        "Calibration image at {0} is not the same size as the others"
                        .format(imagePath))
            imageSize = newSize

            hasCorners, corners = cv2.findChessboardCorners(
                grayImage,
                CHESSBOARD_SIZE,
                None)

            if hasCorners:
                filenames.append(os.path.basename(imagePath))
                objectPoints.append(OBJECT_POINT_ZERO)
                cv2.cornerSubPix(grayImage, corners, (11, 11), (-1, -1),
                        TERMINATION_CRITERIA)
                imagePoints.append(corners)

            cv2.drawChessboardCorners(image, CHESSBOARD_SIZE, corners, hasCorners)
            cv2.imshow(imageDirectory, image)

            # Needed to draw the window
            cv2.waitKey(1)
        print("OOP",objectPoints)
        cv2.destroyWindow(imageDirectory)

        print("Found corners in {0} out of {1} images"
                .format(len(imagePoints), len(imagePaths)))

        np.savez_compressed(cacheFile,
                filenames=filenames, objectPoints=objectPoints,
                imagePoints=imagePoints, imageSize=imageSize)
        return filenames, imagePaths, objectPoints, imagePoints, imageSize

    def getCameraParams(self):
        (leftFilenames, allImagesLeft, leftObjectPoints, leftImagePoints, leftSize
                ) = self.readImagesAndFindChessboards(leftImageDir)
        (rightFilenames, allImagesRight, rightObjectPoints, rightImagePoints, rightSize
                ) = self.readImagesAndFindChessboards(rightImageDir)

        if leftSize != rightSize:
            print("Camera resolutions do not match")
            sys.exit(1)
        imageSize = leftSize

        filenames = list(set(leftFilenames) & set(rightFilenames))
        if (len(filenames) > MAX_IMAGES):
            print("Too many images to calibrate, using {0} randomly selected images"
                    .format(MAX_IMAGES))
            filenames = random.sample(filenames, MAX_IMAGES)
        filenamesSorted = sorted(filenames)
        print("Using these images:")
        print(filenamesSorted)

        # throw away useless images.
        for fileToDelete in allImagesLeft:
            keep = False
            for fileToKeep in filenamesSorted:
                fileToKeep = f"calibImages/left/{fileToKeep}"
                if fileToDelete == fileToKeep:
                    keep = True
                    continue
            if keep is False:
                print(f"Removing {fileToDelete}")
                os.remove(fileToDelete)

        for fileToDelete in allImagesRight:
            keep = False
            for fileToKeep in filenamesSorted:
                fileToKeep = f"calibImages/right/{fileToKeep}"
                if fileToDelete == fileToKeep:
                    keep = True
                    continue
            if keep is False:
                print(f"Removing {fileToDelete}")
                os.remove(fileToDelete)
        return (filenamesSorted, imageSize, leftFilenames, leftObjectPoints, leftImagePoints, rightFilenames, rightObjectPoints, rightImagePoints)

    def getMatchingObjectAndImagePoints(self, requestedFilenames,
        allFilenames, objectPoints, imagePoints):
        requestedFilenameSet = set(requestedFilenames)
        requestedObjectPoints = []
        requestedImagePoints = []

        for index, filename in enumerate(allFilenames):
            if filename in requestedFilenameSet:
                requestedObjectPoints.append(objectPoints[index])
                requestedImagePoints.append(imagePoints[index])

        return requestedObjectPoints, requestedImagePoints

    def calibrateSensor(self, filenames, imageSize, leftFilenames, leftObjectPoints, leftImagePoints,
                rightFilenames, rightObjectPoints, rightImagePoints):

        leftObjectPoints, leftImagePoints = self.getMatchingObjectAndImagePoints(filenames,
            leftFilenames, leftObjectPoints, leftImagePoints)
        rightObjectPoints, rightImagePoints = self.getMatchingObjectAndImagePoints(filenames,
                rightFilenames, rightObjectPoints, rightImagePoints)

        # TODO: Fix this validation
        # Keep getting "Use a.any() or a.all()" even though it's already used?!
        # if (leftObjectPoints != rightObjectPoints).all():
        #     print("Object points do not match")
        #     sys.exit(1)
        objectPoints = leftObjectPoints
        
        print("Calibrating left camera...")
        _, leftCameraMatrix, leftDistortionCoefficients, _, _ = cv2.calibrateCamera(
                objectPoints, leftImagePoints, imageSize, None, None)
        print("Calibrating right camera...")
        _, rightCameraMatrix, rightDistortionCoefficients, _, _ = cv2.calibrateCamera(
                objectPoints, rightImagePoints, imageSize, None, None)

        print("Calibrating cameras together...")
        (rms, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(
                objectPoints, leftImagePoints, rightImagePoints,
                leftCameraMatrix, leftDistortionCoefficients,
                rightCameraMatrix, rightDistortionCoefficients,
                imageSize, None, None, None, None,
                cv2.CALIB_FIX_INTRINSIC, TERMINATION_CRITERIA)
        print(f"RMS {rms} Global value {self.global_rms}, index {self.global_index} increment {self.increment} trendUp {self.trendUp}")
        if rms > self.rms_limit:
            print(f"RMS is too high (> {self.rms_limit}). Throwing  away current chessboard images...")
            self.global_index -= 1
            # os.remove("calibImages/left/{:06d}.jpg".format(self.global_index))
            # os.remove("calibImages/right/{:06d}.jpg".format(self.global_index))
            # print("RMS is too high (> 0.5). Throwing  away chessboard images...")
            # files_in_dir = glob.glob("calibImages/left/*.jpg")
            # for _file in files_in_dir:
            #     os.remove(_file)
            # files_in_dir = glob.glob("calibImages/right/*.jpg")
            # for _file in files_in_dir:
            #     os.remove(_file)
            # files_in_dir = glob.glob("calibImages/left/*.npz")
            # for _file in files_in_dir:
            #     os.remove(_file)
            # files_in_dir = glob.glob("calibImages/right/*.npz")
            # for _file in files_in_dir:
            #     os.remove(_file)
            # sys.exit(0)
        else:
            self.global_rms = rms
            if self.rms_limit < self.turnaround_limit:
                if self.trendUp and rms > self.turnaround_limit - 0.01:
                    print("Switch trend")
                    self.trendUp = False
                    self.increment /= -2.0

                self.rms_limit += self.increment

        print("Rectifying cameras...")
        # TODO: Why do I care about the disparityToDepthMap?
        (leftRectification, rightRectification, leftProjection, rightProjection,
                dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
                        leftCameraMatrix, leftDistortionCoefficients,
                        rightCameraMatrix, rightDistortionCoefficients,
                        imageSize, rotationMatrix, translationVector,
                        None, None, None, None, None,
                        cv2.CALIB_ZERO_DISPARITY, OPTIMIZE_ALPHA)

        print("Saving calibration...")
        leftMapX, leftMapY = cv2.initUndistortRectifyMap(
                leftCameraMatrix, leftDistortionCoefficients, leftRectification,
                leftProjection, imageSize, cv2.CV_32FC1)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(
                rightCameraMatrix, rightDistortionCoefficients, rightRectification,
                rightProjection, imageSize, cv2.CV_32FC1)

        np.savez_compressed(outputFile, imageSize=imageSize,
                leftMapX=leftMapX, leftMapY=leftMapY, leftROI=leftROI,
                rightMapX=rightMapX, rightMapY=rightMapY, rightROI=rightROI)

        cv2.destroyAllWindows()

    def createDepthMap2(self):
        calibration = np.load("./output/calibration.npz", allow_pickle=False)
        imageSize = tuple(calibration["imageSize"])
        leftMapX = calibration["leftMapX"]
        leftMapY = calibration["leftMapY"]
        leftROI = tuple(calibration["leftROI"])
        rightMapX = calibration["rightMapX"]
        rightMapY = calibration["rightMapY"]
        rightROI = tuple(calibration["rightROI"])

        CAMERA_WIDTH = 1280
        CAMERA_HEIGHT = 720

        # TODO: Use more stable identifiers
        left = cv2.VideoCapture(0)
        right = cv2.VideoCapture(2)

        # Increase the resolution
        left.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        # Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
        left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        # TODO: Why these values in particular?
        # TODO: Try applying brightness/contrast/gamma adjustments to the images
        stereoMatcher = cv2.StereoBM_create()
        stereoMatcher.setMinDisparity(4)
        stereoMatcher.setNumDisparities(128)
        stereoMatcher.setBlockSize(21)
        stereoMatcher.setROI1(leftROI)
        stereoMatcher.setROI2(rightROI)
        stereoMatcher.setSpeckleRange(16)
        stereoMatcher.setSpeckleWindowSize(45)

        # Grab both frames first, then retrieve to minimize latency between cameras
        while(True):
            if not left.grab() or not right.grab():
                print("No more frames")
                break

            _, leftFrame = left.retrieve()
            leftHeight, leftWidth = leftFrame.shape[:2]
            _, rightFrame = right.retrieve()
            rightHeight, rightWidth = rightFrame.shape[:2]

            if (leftWidth, leftHeight) != imageSize:
                print("Left camera has different size than the calibration data")
                break

            if (rightWidth, rightHeight) != imageSize:
                print("Right camera has different size than the calibration data")
                break

            fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY, REMAP_INTERPOLATION)
            fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY, REMAP_INTERPOLATION)

            grayLeft = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
            grayRight = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)
            depth = stereoMatcher.compute(grayLeft, grayRight)


            numpy_horizontal_concat = np.concatenate((cv2.resize(leftFrame, (640,480)), cv2.resize(rightFrame, (640,480))), axis=1)
            cv2.imshow('Stereo image', numpy_horizontal_concat)
            cv2.imshow('Dpth image', cv2.resize(depth / DEPTH_VISUALIZATION_SCALE, (640,480)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        left.release()
        right.release()
        cv2.destroyAllWindows()

    def loadCalibFile(self):
        calibration = np.load("./output/calibration.npz", allow_pickle=False)
        self.imageSize = tuple(calibration["imageSize"])
        self.leftMapX = calibration["leftMapX"]
        self.leftMapY = calibration["leftMapY"]
        self.leftROI = tuple(calibration["leftROI"])
        self.rightMapX = calibration["rightMapX"]
        self.rightMapY = calibration["rightMapY"]
        self.rightROI = tuple(calibration["rightROI"])

        CAMERA_WIDTH = 1280
        CAMERA_HEIGHT = 720

        # TODO: Use more stable identifiers
        self.left = cv2.VideoCapture(0)
        self.right = cv2.VideoCapture(2)

        # Increase the resolution
        self.left.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        # Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
        self.left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    def createDepthMap(self):
        # TODO: Why these values in particular?
        # TODO: Try applying brightness/contrast/gamma adjustments to the images
        stereoMatcher = cv2.StereoBM_create()
        stereoMatcher.setPreFilterType(self.preFilterType)
        stereoMatcher.setPreFilterSize(self.preFilterSize)
        stereoMatcher.setPreFilterCap(self.preFilterCap)
        stereoMatcher.setMinDisparity(self.min_disp)
        stereoMatcher.setNumDisparities(self.max_disp)
        stereoMatcher.setBlockSize(self.blockSize)
        stereoMatcher.setROI1(self.leftROI)
        stereoMatcher.setTextureThreshold(self.textureThreshold)
        stereoMatcher.setUniquenessRatio(self.uniquenessRatio)
        stereoMatcher.setROI2(self.rightROI)
        stereoMatcher.setSpeckleRange(self.speckleRange)
        stereoMatcher.setSpeckleWindowSize(self.speckleWindowSize)
        stereoMatcher.setDisp12MaxDiff(self.disp12MaxDiff)
        stereoMatcher.setSmallerBlockSize(self.smallerBlockSize)

        pix_disp = QPixmap()
        pix0 = QPixmap()
        pix1 = QPixmap()
        # Grab both frames first, then retrieve to minimize latency between cameras
        if not self.left.grab() or not self.right.grab():
            print("No more frames")
            return (pix0, pix1, pix_disp)

        _, leftFrame = self.left.retrieve()
        leftHeight, leftWidth = leftFrame.shape[:2]
        _, rightFrame = self.right.retrieve()
        rightHeight, rightWidth = rightFrame.shape[:2]

        if (leftWidth, leftHeight) != self.imageSize:
            print("Left camera has different size than the calibration data")
            return (pix0, pix1, pix_disp)

        if (rightWidth, rightHeight) != self.imageSize:
            print("Right camera has different size than the calibration data")
            return (pix0, pix1, pix_disp)

        # leftFrame = downsample_image(leftFrame, 1)
        # rightFrame = downsample_image(rightFrame, 1)

        fixedLeft = cv2.remap(leftFrame, self.leftMapX, self.leftMapY, REMAP_INTERPOLATION)
        fixedRight = cv2.remap(rightFrame, self.rightMapX, self.rightMapY, REMAP_INTERPOLATION)

        grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
        # grayLeft =\
        #     cv2.fastNlMeansDenoising(grayLeft, dst=31, h=7, templateWindowSize=5, searchWindowSize=21)

        # grayRight =\
        #     cv2.fastNlMeansDenoising(grayRight, dst=31, h=7, templateWindowSize=5, searchWindowSize=21)

        # grayLeft =\
        #     cv2.medianBlur(grayLeft, 11)
        # grayRight =\
        #     cv2.medianBlur(grayRight, 11)
        depth = stereoMatcher.compute(grayLeft, grayRight)
        depth =\
            cv2.medianBlur(depth, 5)
        depth =\
            cv2.GaussianBlur(depth, (9, 9), 0)
        depth = depth / DEPTH_VISUALIZATION_SCALE

        normalized_depth = cv2.normalize(depth, depth, 0, 255, norm_type=cv2.NORM_MINMAX)
        normalized_depth_color = cv2.cvtColor(normalized_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        img = QImage(
            normalized_depth_color,
            normalized_depth_color.shape[1],
            normalized_depth_color.shape[0],
            QImage.Format_RGB888)
        pix_disp = QPixmap.fromImage(img)
        pix_disp = pix_disp.scaled(
            800,
            400,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation)
        leftFrame = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2RGB)
        img = QImage(
            grayLeft,
            leftFrame.shape[1],
            leftFrame.shape[0],
            QImage.Format_Indexed8)
        pix0 = QPixmap.fromImage(img)
        pix0 = pix0.scaled(
            800,
            600,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation)

        rightFrame = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2RGB)
        img = QImage(
            grayRight,
            rightFrame.shape[1],
            rightFrame.shape[0],
            QImage.Format_Indexed8)
        pix1 = QPixmap.fromImage(img)
        pix1 = pix1.scaled(
            800,
            600,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation)

        return (pix0, pix1, pix_disp)

    def calibrate(self):
        # Size is determined, (number of columns - 1, number of rows -1)
        chessboard_size = (9, 6)

        # Define arrays to save detected points
        obj_points_left = []  # 3D points in real world space
        img_points_left = []  # 3D points in image plane
        obj_points_right = []  # 3D points in real world space
        img_points_right = []  # 3D points in image plane

        # Prepare grid and points to display
        objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
        objp[:, :2] = \
            np.mgrid[
                0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        # read imagescalibration_
        p = Path('calibImages')
        filenamesLeft = [i for i in p.glob('**/frame0*.jpg')]
        filenamesRight = [i for i in p.glob('**/frame1*.jpg')]
        filenamesLeft = sorted(filenamesLeft, key=lambda i: int(i.stem.split("_")[1]))
        filenamesRight = sorted(filenamesRight, key=lambda i: int(i.stem.split("_")[1]))
        print(filenamesLeft)
        print(filenamesRight)

        # Iterate over images to find intrinsic matrix
        for i in range(len(filenamesLeft)):
            # Load image
            print(f"Checking {filenamesLeft[i]}")
            image = cv2.imread(str(filenamesLeft[i]))
            if image is None:
                print(f"Falied to load image {filenamesLeft[i]}")
                continue

            gray_image_left = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # cv2.startWindowThread()
            # cv2.namedWindow("grayImage")
            # cv2.imshow("grayImage", gray_image)
            # cv2.waitKey()
            print("Image loaded, Analizying...")
            # find chessboard corners
            ret, corners = cv2.findChessboardCorners(
                gray_image_left,
                chessboard_size,
                None)

            if ret is True:
                print("Chessboard detected!")
                print(filenamesLeft[i])
                # define criteria for subpixel accuracy
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001)
                # refine corner location (to subpixel accuracy)
                # based on criteria.
                cv2.cornerSubPix(
                    gray_image_left,
                    corners,
                    (5, 5),
                    (-1, -1),
                    criteria)
                obj_points_left.append(objp)
                img_points_left.append(corners)

                # Iterate over images to find intrinsic matrix
        for i in range(len(filenamesRight)):
            # Load image
            print(f"Checking {filenamesRight[i]}")
            image = cv2.imread(str(filenamesRight[i]))
            if image is None:
                print(f"Falied to load image {filenamesRight[i]}")
                continue

            gray_image_right = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            print("Image loaded, Analizying...")
            # find chessboard corners
            ret, corners = cv2.findChessboardCorners(
                gray_image_right,
                chessboard_size,
                None)

            if ret is True:
                print("Chessboard detected!")
                print(filenamesRight[i])
                # define criteria for subpixel accuracy
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001)
                # refine corner location (to subpixel accuracy)
                # based on criteria.
                cv2.cornerSubPix(
                    gray_image_right,
                    corners,
                    (5, 5),
                    (-1, -1),
                    criteria)
                obj_points_right.append(objp)
                img_points_right.append(corners)

        # Calibrate camera
        ret_left, K_left, dist_left, rvecs_left, tvecs_left = \
            cv2.calibrateCamera(
                obj_points_left,
                img_points_left,
                gray_image_left.shape[::-1],
                None,
                None)  # Save parameters into numpy file

        # Calibrate camera
        ret_right, K_right, dist_right, rvecs_right, tvecs_right = \
            cv2.calibrateCamera(
                obj_points_right,
                img_points_right,
                gray_image_right.shape[::-1],
                None,
                None)  # Save parameters into numpy file

        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                    100, 1e-5)

        flags = (cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST +
                 cv2.CALIB_SAME_FOCAL_LENGTH)

        (_, _, _, _, _, rotationMatrix, translationVector, _, _) =\
            cv2.stereoCalibrate(
              obj_points_right, img_points_left, img_points_right,
              K_left, dist_left,
              K_right, dist_right,
              gray_image_left.shape[::-1], None, None, None, None,
              cv2.CALIB_FIX_INTRINSIC, criteria)

        (leftRectification,
            rightRectification, leftProjection, rightProjection,
            dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
            K_left, dist_left,
            K_right, dist_right,
            gray_image_left.shape[::-1], rotationMatrix, translationVector,
            None, None, None, None, None,
            cv2.CALIB_ZERO_DISPARITY, flags)

        leftMapX, leftMapY = cv2.initUndistortRectifyMap(
                K_left, dist_left, leftRectification,
                leftProjection, gray_image_left.shape[::-1], cv2.CV_32FC1)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(
                K_right, dist_right, rightRectification,
                rightProjection, gray_image_left.shape[::-1], cv2.CV_32FC1)

        np.savez_compressed(
            "./cameraParams/calibration",
            imageSize=gray_image_left.shape[::-1],
            leftMapX=leftMapX, leftMapY=leftMapY, leftROI=leftROI,
            rightMapX=rightMapX, rightMapY=rightMapY, rightROI=rightROI)

        return True

    def generateStereoImage(self, frames):
        calibration = np.load("./output/calibration.npz", allow_pickle=False)
        imageSize = tuple(calibration["imageSize"])
        leftMapX = calibration["leftMapX"]
        leftMapY = calibration["leftMapY"]
        leftROI = tuple(calibration["leftROI"])
        rightMapX = calibration["rightMapX"]
        rightMapY = calibration["rightMapY"]
        rightROI = tuple(calibration["rightROI"])

        stereoMatcher = cv2.StereoBM_create()

        stereoMatcher.setMinDisparity(self.min_disp)
        stereoMatcher.setNumDisparities(self.max_disp)
        stereoMatcher.setBlockSize(self.blockSize)
        stereoMatcher.setSpeckleRange(self.speckleRange)
        stereoMatcher.setSpeckleWindowSize(self.speckleWindowSize)
        # num_disp = self.max_disp - self.min_disp
        # stereoMatcher = cv2.StereoSGBM_create(
        #     minDisparity=self.min_disp,
        #     numDisparities=num_disp,
        #     blockSize=self.blockSize,
        #     uniquenessRatio=self.uniquenessRatio,
        #     speckleWindowSize=self.speckleWindowSize,
        #     speckleRange=self.speckleRange,
        #     disp12MaxDiff=self.disp12MaxDiff,
        #     P1=8 * 3 * self.win_size ** 2,  # 8*3*win_size**2,
        #     P2=32 * 3 * self.win_size ** 2)  # 32*3*win_size**2)

        fixedLeft = cv2.remap(frames[0], leftMapX, leftMapY, cv2.INTER_LINEAR)
        fixedRight = cv2.remap(frames[1], rightMapX, rightMapY, cv2.INTER_LINEAR)

        grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
        depth = stereoMatcher.compute(grayLeft, grayRight)

        K = np.load('./cameraParams/K.npy')
        dist = np.load('./cameraParams/dist.npy')
        h, w = frames[0].shape[:2]

        # Get optimal camera matrix for better undistortion
        new_camera_matrix, roi =\
            cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        # Undistort images
        img_1_undistorted =\
            cv2.undistort(frames[0], K, dist, None, new_camera_matrix)
        img_2_undistorted =\
            cv2.undistort(frames[1], K, dist, None, new_camera_matrix)

        # Downsample each image 3 times (because they're too big)
        img_1_downsampled = downsample_image(img_1_undistorted, 0)
        img_2_downsampled = downsample_image(img_2_undistorted, 0)

        # Set disparity parameters
        # Note: disparity range is tuned according to specific parameters
        # obtained through trial and error.
        # min_disp = -1
        # max_disp = 63  # min_disp * 9
        num_disp = self.max_disp - self.min_disp  # Needs to be divisible by 16

        #stereo = cv2.StereoBM_create()
        # Create Block matching object.
        stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disp,
            numDisparities=num_disp,
            blockSize=self.blockSize,
            uniquenessRatio=self.uniquenessRatio,
            speckleWindowSize=self.speckleWindowSize,
            speckleRange=self.speckleRange,
            disp12MaxDiff=self.disp12MaxDiff,
            P1=8 * 3 * self.win_size ** 2,  # 8*3*win_size**2,
            P2=32 * 3 * self.win_size ** 2)  # 32*3*win_size**2)

        # Compute disparity map
        # print("\nComputing the disparity  map...")
        disparity_map = stereo.compute(img_1_downsampled, img_2_downsampled)

        # Show disparity map before generating 3D cloud
        # to verify that point cloud will be usable.
        # cv2.startWindowThread()
        # cv2.imshow('frame_stereo', disparity_map)
        # cv2.waitKey(1)
        # plt.imshow(disparity_map, 'gray')
        # plt.show()

        # # Generate  point cloud.
        # # print("\nGenerating the 3D map...")
        # # Get new downsampled width and height
        # h, w = img_2_downsampled.shape[:2]
        # # Load focal length.
        # focal_length = np.load('./cameraParams/FocalLength.npy')
        # # Perspective transformation matrix
        # # This transformation matrix is from the openCV documentation,
        # # didn't seem to work for me.
        # Q = np.float32([
        #     [1, 0, 0, -w / 2.0],
        #     [0, -1, 0, h / 2.0],
        #     [0, 0, 0, -focal_length],
        #     [0, 0, 1, 0]])
        # # This transformation matrix is derived from
        # # Prof. Didier Stricker's power point presentation on computer vision.
        # # https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
        # Q2 = np.float32([
        #     [1, 0, 0, 0],
        #     [0, -1, 0, 0],
        #     [0, 0, focal_length * 0.05, 0],
        #     # Focal length multiplication obtained experimentally.
        #     [0, 0, 0, 1]])  # Reproject points into 3D
        # points_3D = cv2.reprojectImageTo3D(disparity_map, Q2)

        # # Get color points
        # colors = cv2.cvtColor(img_1_downsampled, cv2.COLOR_BGR2RGB)

        # # Get rid of points with value 0 (i.e no depth)
        # mask_map = disparity_map > disparity_map.min()
        # # Mask colors and points.
        # output_points = points_3D[mask_map]
        # output_colors = colors[mask_map]
        # # Define name for output file
        # output_file = './output/reconstructed.ply'
        # # Generate point cloud
        # # print("\n Creating the output file... \n")
        # create_output(output_points, output_colors, output_file)
        return (depth, frames[0], frames[1])

    def calculateEpipolarLine(self, frames):
        # sift = cv2.SIFT() causes crash during calling detectAndCompute
        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(frames[0], None)
        kp2, des2 = sift.detectAndCompute(frames[1], None)

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
        img5, img6 = drawLines(frames[0], frames[1], lines1, pts1, pts2)
        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = drawLines(frames[1], frames[0], lines2, pts2, pts1)
        return frames


class BackendSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object, object, object)


class Backend(QObject):
    def __init__(self):
        super(Backend, self).__init__()

        self.signals = BackendSignals()
        self.stop = False
        self.cameras = StereoCameraSensor()

    def updateTextureThreshold(
          self,
          textureThreshold):
        self.cameras.textureThreshold = textureThreshold

    def updateSmallerBlockSize(
          self,
          smallerBlockSize):
        self.cameras.smallerBlockSize = smallerBlockSize

    def updatePreFilterType(
          self,
          preFilterType):
        self.cameras.preFilterType = preFilterType

    def updatePrefilterCap(
          self,
          preFilterCap):
        self.cameras.preFilterCap = preFilterCap

    def updatePrefilterSize(
          self,
          preFilterSize):
        self.cameras.preFilterSize = preFilterSize

    def updateMin_disp(
          self,
          min_disp):
        self.cameras.min_disp = min_disp

    def updateMax_disp(
          self,
          max_disp):
        self.cameras.max_disp = max_disp

    def updateBlockSize(
          self,
          blockSize):
        self.cameras.blockSize = blockSize

    def updateUniquenessRatio(
          self,
          uniquenessRatio):
        self.cameras.uniquenessRatio = uniquenessRatio

    def updateSpeckleWindowSize(
          self,
          speckleWindowSize):
        self.cameras.speckleWindowSize = speckleWindowSize

    def updateSpeckleRange(
          self,
          speckleRange):
        self.cameras.speckleRange = speckleRange

    def updateDisp12MaxDiff(
          self,
          disp12MaxDiff):
        self.cameras.disp12MaxDiff = disp12MaxDiff

    @QtCore.pyqtSlot()
    def run(self):
        # Retrieve args/kwargs here; and fire processing using them
        try:
            # frames = []
            # # frames.append(cv2.imread("./resources/frameLeft.jpg"))
            # # frames.append(cv2.imread("./resources/frameRight.jpg"))
            # self.cameras.calibrate()
            self.cameras.loadCalibFile()
            while(True):
                if self.stop:
                    break
            #     # if self.cameras.generateCalibrationImages():
            #     #     break

            #     frames = self.cameras.showFrame(inWindow=False)

            #     # frames = []
            #     # frames.append(cv2.imread("./resources/frameLeft.jpg"))
            #     # frames.append(cv2.imread("./resources/frameRight.jpg"))
            #     pix_disp = QPixmap()
            #     pix0 = QPixmap()
            #     pix1 = QPixmap()
            #     if len(frames) > 1:
            #         #time.sleep(3)
            #         # frames[0] =\
            #         #     cv2.medianBlur(frames[0], 9)
            #         # frames[1] =\
            #         #     cv2.medianBlur(frames[1], 9)
            #         # frames[0] =\
            #         #     cv2.fastNlMeansDenoisingColored(frames[0], None, 10, 10, 7, 21)

            #         # frames[1] =\
            #         #     cv2.fastNlMeansDenoisingColored(frames[1], None, 10, 10, 7, 21)
            #         # cv2.imwrite(f'resources/frameLeft.jpg', frames[0])
            #         # cv2.imwrite(f'resources/frameRight.jpg', frames[1])

            #         #frames = self.cameras.calculateEpipolarLine(frames)
            #         (disparity_map, frames[0], frames[1]) = self.cameras.generateStereoImage(frames)
            #         # disparity_map =\
            #         #     cv2.GaussianBlur(disparity_map, (15, 15), 0)
                #     img = QImage(
                #         disparity_map,
                #         int(2*disparity_map.shape[1]),
                #         disparity_map.shape[0],
                #         QImage.Format_Indexed8)
                #     pix_disp = QPixmap.fromImage(img)
                #     pix_disp = pix_disp.scaled(
                #         800,
                #         400,
                #         Qt.KeepAspectRatio,
                #         Qt.SmoothTransformation)

                #     img = QImage(
                #         frames[0],
                #         frames[0].shape[1],
                #         frames[0].shape[0],
                #         QImage.Format_RGB888)
                #     pix0 = QPixmap.fromImage(img)
                #     pix0 = pix0.scaled(
                #         800,
                #         600,
                #         Qt.KeepAspectRatio,
                #         Qt.SmoothTransformation)

                #     img = QImage(
                #         frames[1],
                #         frames[1].shape[1],
                #         frames[1].shape[0],
                #         QImage.Format_RGB888)
                #     pix1 = QPixmap.fromImage(img)
                #     pix1 = pix1.scaled(
                #         800,
                #         600,
                #         Qt.KeepAspectRatio,
                #         Qt.SmoothTransformation)

                # self.signals.result.emit(pix0, pix1, pix_disp)
                (pix0, pix1, pix_disp) = self.cameras.createDepthMap()
                self.signals.result.emit(pix0, pix1, pix_disp)
                # cv2.imshow('Stereo image', numpy_horizontal_concat)
                # cv2.imshow('Dpth image', cv2.resize(depth / DEPTH_VISUALIZATION_SCALE, (640,480)))
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                # cv2.destroyAllWindows()
        except Exception:
            self.cameras.left.release()
            self.cameras.right.release()
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.cameras.left.release()
            self.cameras.right.release()
            self.signals.finished.emit()  # Done


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.win_sizeUpdated = QtCore.pyqtSignal(int)
        self.min_dispUpdated = QtCore.pyqtSignal(int)
        self.max_dispUpdated = QtCore.pyqtSignal(int)
        self.blockSizeUpdated = QtCore.pyqtSignal(int)
        self.uniquenessRatioUpdated = QtCore.pyqtSignal(int)
        self.speckleWindowSizeUpdated = QtCore.pyqtSignal(int)
        self.speckleRangeUpdated = QtCore.pyqtSignal(int)
        self.disp12MaxDiffUpdated = QtCore.pyqtSignal(int)
        layout = QGridLayout()

        self.video0 = QLabel()
        self.video1 = QLabel()
        self.video_disp = QLabel()
        self.textureThreshold = QSpinBox()
        self.min_disp = QSpinBox()
        self.max_disp = QSpinBox()
        self.blockSize = QSpinBox()
        self.uniquenessRatio = QSpinBox()
        self.speckleWindowSize = QSpinBox()
        self.speckleRange = QSpinBox()
        self.disp12MaxDiff = QSpinBox()
        self.preFilterSize = QSpinBox()
        self.preFilterType = QSpinBox()
        self.preFilterCap = QSpinBox()
        self.smallerBlockSize = QSpinBox()

        layout.addWidget(self.video0, 0, 0, 1, 4)
        layout.addWidget(self.video1, 0, 4, 1, 4)
        layout.addWidget(self.video_disp, 1, 2, 1, 4)
        spinlayout = QHBoxLayout()
        spinlayout.addWidget(QLabel("textureThreshold"))
        spinlayout.addWidget(self.textureThreshold)
        spinlayout.addWidget(QLabel("min_disp"))
        spinlayout.addWidget(self.min_disp)
        spinlayout.addWidget(QLabel("max_disp"))
        spinlayout.addWidget(self.max_disp)
        spinlayout.addWidget(QLabel("blockSize"))
        spinlayout.addWidget(self.blockSize)
        spinlayout.addWidget(QLabel("uniquenessRatio"))
        spinlayout.addWidget(self.uniquenessRatio)
        spinlayout.addWidget(QLabel("speckleWindowSize"))
        spinlayout.addWidget(self.speckleWindowSize)
        spinlayout.addWidget(QLabel("speckleRange"))
        spinlayout.addWidget(self.speckleRange)
        spinlayout.addWidget(QLabel("disp12MaxDiff"))
        spinlayout.addWidget(self.disp12MaxDiff)
        spinLayoutWidget = QWidget()
        spinLayoutWidget.setLayout(spinlayout)
        layout.addWidget(spinLayoutWidget, 2, 0, 1, 8)
        spinlayout2 = QHBoxLayout()
        spinlayout2.addWidget(QLabel("smallerBlockSize"))
        spinlayout2.addWidget(self.smallerBlockSize)
        spinlayout2.addWidget(QLabel("preFilterType"))
        spinlayout2.addWidget(self.preFilterType)
        spinlayout2.addWidget(QLabel("preFilterCap"))
        spinlayout2.addWidget(self.preFilterCap)
        spinlayout2.addWidget(QLabel("preFilterSize"))
        spinlayout2.addWidget(self.preFilterSize)
        spinLayoutWidget2 = QWidget()
        spinLayoutWidget2.setLayout(spinlayout2)
        layout.addWidget(spinLayoutWidget2, 3, 0, 1, 8)

        w = QWidget()
        w.setLayout(layout)

        self.setCentralWidget(w)

        self.show()
        self.startBackend()

    def updateVideo(self, v0, v1, v_disp):
        self.video0.setPixmap(v0)
        self.video1.setPixmap(v1)
        self.video_disp.setPixmap(v_disp)

    def thread_complete(self):
        print("THREAD COMPLETE!")

    def startBackend(self):
        self.workerThread = QThread(self)
        self.worker = Backend()

        # Connect worker signals
        self.worker.signals.result.connect(self.updateVideo)
        self.worker.signals.finished.connect(self.thread_complete)

        # Init spin boxes
        self.textureThreshold.setRange(0, 10000)
        self.textureThreshold.setValue(self.worker.cameras.textureThreshold)
        self.min_disp.setRange(-1, 1000)
        self.min_disp.setSingleStep(1)
        self.min_disp.setValue(self.worker.cameras.min_disp)
        self.max_disp.setRange(16, 1000)
        self.max_disp.setSingleStep(16)
        self.max_disp.setValue(self.worker.cameras.max_disp)
        self.blockSize.setRange(5, 255)
        self.blockSize.setSingleStep(2)
        self.blockSize.setValue(self.worker.cameras.blockSize)
        self.smallerBlockSize.setRange(-1, 1000000)
        self.smallerBlockSize.setSingleStep(1)
        self.smallerBlockSize.setValue(self.worker.cameras.smallerBlockSize)
        self.uniquenessRatio.setRange(0, 1000)
        self.uniquenessRatio.setValue(self.worker.cameras.uniquenessRatio)
        self.speckleWindowSize.setRange(-1, 1000)
        self.speckleWindowSize.setValue(self.worker.cameras.speckleWindowSize)
        self.speckleRange.setRange(-1, 1000)
        self.speckleRange.setValue(self.worker.cameras.speckleRange)
        self.disp12MaxDiff.setRange(-1, 1000)
        self.disp12MaxDiff.setValue(self.worker.cameras.disp12MaxDiff)
        self.preFilterType.setRange(0, 1)
        self.preFilterType.setSingleStep(1)
        self.preFilterType.setValue(self.worker.cameras.preFilterType)
        self.preFilterSize.setRange(5, 255)
        self.preFilterSize.setSingleStep(2)
        self.preFilterSize.setValue(self.worker.cameras.preFilterSize)
        self.preFilterCap.setRange(1, 63)
        self.preFilterCap.setSingleStep(1)
        self.preFilterCap.setValue(self.worker.cameras.preFilterCap)

        # wire signals/slots
        self.textureThreshold.valueChanged.connect(self.worker.updateTextureThreshold)
        self.min_disp.valueChanged.connect(self.worker.updateMin_disp)
        self.max_disp.valueChanged.connect(self.worker.updateMax_disp)
        self.blockSize.valueChanged.connect(self.worker.updateBlockSize)
        self.uniquenessRatio.valueChanged.connect(
            self.worker.updateUniquenessRatio)
        self.speckleWindowSize.valueChanged.connect(
            self.worker.updateSpeckleWindowSize)
        self.speckleRange.valueChanged.connect(
              self.worker.updateSpeckleRange)
        self.disp12MaxDiff.valueChanged.connect(
              self.worker.updateDisp12MaxDiff)
        self.smallerBlockSize.valueChanged.connect(
              self.worker.updateSmallerBlockSize)
        self.preFilterType.valueChanged.connect(
              self.worker.updatePreFilterType)
        self.preFilterSize.valueChanged.connect(
              self.worker.updatePrefilterSize)
        self.preFilterCap.valueChanged.connect(
              self.worker.updatePrefilterCap)

        # Execute
        self.worker.moveToThread(self.workerThread)
        self.workerThread.finished.connect(self.worker.deleteLater)
        self.workerThread.started.connect(self.worker.run)
        self.workerThread.start()

    def sigint_handler(self):
        self.worker.stop = True
        self.workerThread.quit()
        self.workerThread.wait()
        QApplication.quit()


if __name__ == "__main__":
    # app = QApplication([])
    # window = MainWindow()
    # app.aboutToQuit.connect(window.sigint_handler)

    # sys.exit(app.exec_())
    worker = Backend()
    while True:
        worker.cameras.createCalibImages()
        (filenames, imageSize, leftFilenames, leftObjectPoints, leftImagePoints, rightFilenames, rightObjectPoints, rightImagePoints) = worker.cameras.getCameraParams()
        worker.cameras.calibrateSensor(filenames, imageSize, leftFilenames, leftObjectPoints, leftImagePoints, rightFilenames, rightObjectPoints, rightImagePoints)
    # time.sleep(3)
    #worker.cameras.createDepthMap2()
    # del worker
    # time.sleep(3)
    # app = QApplication([])
    # window = MainWindow()
    # app.aboutToQuit.connect(window.sigint_handler)

    # sys.exit(app.exec_())
