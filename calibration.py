import sys
import glob
import os

import cv2
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import QObject

# Stereo calibration result file
CALIBRATION_RESULT = "output/calibration.npz"

# Camera resolution
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


# Generates calibration images and data
# Calibrates the cameras and stores the result for future use.
class SensorCalibrator(QObject):
    framesUpdated = QtCore.pyqtSignal(object, object, object)
    rmsMessages = QtCore.pyqtSignal(object, object)
    rmsLimitUpdated = QtCore.pyqtSignal(float)
    calibImageIndexUpdated = QtCore.pyqtSignal(int)

    # Location where the chessboard images to be stored
    LEFT_IMAGE_DIR = "calibImages/left"
    RIGHT_IMAGE_DIR = "calibImages/right"
    LEFT_PATH = "{}/{:06d}.jpg"
    RIGHT_PATH = "{}/{:06d}.jpg"
    CALIB_PARAMS = "output/calibParams.npz"
    # Chessboard pattern size
    CHESSBOARD_SIZE = (9, 6)
    # Empty object point table.
    OBJECT_POINT_ZERO = np.zeros(
        (CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3),
        np.float32)
    OBJECT_POINT_ZERO[:, :2] = np.mgrid[
        0:CHESSBOARD_SIZE[0],
        0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    # Termination criteria for sub pixel corner search and stereo calibrate.
    TERMINATION_CRITERIA = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001)
    OPTIMIZE_ALPHA = 0.25

    def __init__(self):
        super(SensorCalibrator, self).__init__()
        # Last RMS value, that was within limits.
        self.last_accepted_rms = 100
        # Index of the last stored chess board calibration image.
        self.calib_image_index = 0
        # Accepted RMS limit, all images that would
        # increase the RMS over the limit are thrown away.
        self.rms_limit = 0.135
        # Increment of RMS limit change
        self.increment = 0.01
        # Maximum rms the rms_limit can be increased to.
        self.max_rms = 0.315

        if not os.path.isdir("calibImages"):
            os.mkdir("calibImages")

        if not os.path.isdir("output"):
            os.mkdir("output")

        self.advancedCalib = True
        self.ignoreExistingImageData = True

    # Saves current image pair and
    # gets chessboard corner data of all saved images.
    def saveAndProcessImage(self, leftFrame, rightFrame):
        left_path = self.LEFT_PATH.format(
            self.LEFT_IMAGE_DIR, self.calib_image_index)
        right_path = self.RIGHT_PATH.format(
            self.RIGHT_IMAGE_DIR, self.calib_image_index)
        print(
            f"Store image Left {left_path}")
        print(
            f"Store image Right {right_path}")
        cv2.imwrite(left_path, leftFrame)
        cv2.imwrite(right_path, rightFrame)
        self.calib_image_index += 1
        self.calibImageIndexUpdated.emit(self.calib_image_index)
        self.getCameraParams()

    # Tries to find chessboard corners for each image for per camera sensor
    def _readImagesAndFindChessboards(self, imageDirectory):
        cacheFile = "{0}/chessboards.npz".format(imageDirectory)
        if self.ignoreExistingImageData is False:
            try:
                cache = np.load(cacheFile)
                print(f"Loading image data from cache file at {cacheFile}")
                return (list(cache["filenames"]),
                        [], list(cache["objectPoints"]),
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
                        "Calibration image at {0} \
is not the same size as the others"
                        .format(imagePath))
            imageSize = newSize

            hasCorners, corners = cv2.findChessboardCorners(
                grayImage,
                self.CHESSBOARD_SIZE,
                None)

            if hasCorners:
                filenames.append(os.path.basename(imagePath))
                objectPoints.append(self.OBJECT_POINT_ZERO)
                cv2.cornerSubPix(
                    grayImage,
                    corners,
                    (11, 11),
                    (-1, -1),
                    self.TERMINATION_CRITERIA)
                imagePoints.append(corners)

            cv2.drawChessboardCorners(
                image,
                self.CHESSBOARD_SIZE,
                corners,
                hasCorners)
            if imageDirectory == self.LEFT_IMAGE_DIR:
                self.framesUpdated.emit(image, None, None)
            else:
                self.framesUpdated.emit(None, image, None)

        print("Found corners in {0} out of {1} images"
              .format(len(imagePoints), len(imagePaths)))

        np.savez_compressed(
            cacheFile,
            filenames=filenames,
            objectPoints=objectPoints,
            imagePoints=imagePoints,
            imageSize=imageSize)
        return filenames, imagePaths, objectPoints, imagePoints, imageSize

    # Gets chessboard corner data of all saved images for each camera.
    def getCameraParams(self):
        (leftFilenames,
         allImagesLeft,
         leftObjectPoints,
         leftImagePoints,
         leftSize
         ) = self._readImagesAndFindChessboards(self.LEFT_IMAGE_DIR)
        (rightFilenames,
         allImagesRight,
         rightObjectPoints,
         rightImagePoints,
         rightSize
         ) = self._readImagesAndFindChessboards(self.RIGHT_IMAGE_DIR)

        if leftSize != rightSize:
            print("Camera resolutions do not match")
            sys.exit(1)
        imageSize = leftSize

        filenames = list(set(leftFilenames) & set(rightFilenames))
        filenamesSorted = sorted(filenames)
        print("Using these images:")
        print(filenamesSorted)

        # throw away useless images.
        print("Removing images that did not have identifiable \
corners on any of the sensors...")
        for fileToDelete in allImagesLeft:
            keep = False
            for fileToKeep in filenamesSorted:
                fileToKeep = f"{self.LEFT_IMAGE_DIR}/{fileToKeep}"
                if fileToDelete == fileToKeep:
                    keep = True
                    continue
            if keep is False:
                print(f"Removing {fileToDelete}")
                os.remove(fileToDelete)

        for fileToDelete in allImagesRight:
            keep = False
            for fileToKeep in filenamesSorted:
                fileToKeep = f"{self.RIGHT_IMAGE_DIR}/{fileToKeep}"
                if fileToDelete == fileToKeep:
                    keep = True
                    continue
            if keep is False:
                print(f"Removing {fileToDelete}")
                os.remove(fileToDelete)

        np.savez_compressed(
            self.CALIB_PARAMS,
            filenames=filenamesSorted,
            imageSize=imageSize,
            leftFilenames=leftFilenames,
            leftObjectPoints=leftObjectPoints,
            leftImagePoints=leftImagePoints,
            rightFilenames=rightFilenames,
            rightObjectPoints=rightObjectPoints,
            rightImagePoints=rightImagePoints)

    def _getMatchingObjectAndImagePoints(
          self,
          requestedFilenames,
          allFilenames,
          objectPoints,
          imagePoints):
        requestedFilenameSet = set(requestedFilenames)
        requestedObjectPoints = []
        requestedImagePoints = []

        for index, filename in enumerate(allFilenames):
            if filename in requestedFilenameSet:
                requestedObjectPoints.append(objectPoints[index])
                requestedImagePoints.append(imagePoints[index])

        return requestedObjectPoints, requestedImagePoints

    # Check RMS value in advanced calibration image retrieving.
    # This will only store images that result in an overall calibration RMS
    # below the rms_limit.
    # The limit is gradually increasing until max rms has been reached
    # This allows to filter some of the bad images.
    def checkRMS(self, rms):
        if self.advancedCalib:
            message = f"\
    Advanced calibration:\n\
    Current RMS       {rms}\n\
    Last valid RMS    {self.last_accepted_rms}\n\
    Max RMS           {self.max_rms}\n\
    Current RMS limit {self.rms_limit}\n\
    RMS increment     {self.increment}\n\
    Image index       {self.calib_image_index}\n\n"
            if rms > self.rms_limit:
                message += f"   RMS is too high (> {self.rms_limit}). \
Throwing away current chessboard image pair..."
                self.calib_image_index -= 1
                self.calibImageIndexUpdated.emit(self.calib_image_index)
                os.remove(
                    "{}/{:06d}.jpg"
                    .format(self.LEFT_IMAGE_DIR, self.calib_image_index))
                os.remove(
                  "{}/{:06d}.jpg"
                  .format(self.RIGHT_IMAGE_DIR, self.calib_image_index))
            else:
                message += "    RMS is within limit, adding image..."
                self.last_accepted_rms = rms
                if self.rms_limit < self.max_rms:
                    self.rms_limit += self.increment
                    self.rmsLimitUpdated.emit(self.rms_limit)
            print(message)
            self.rmsMessages.emit(rms, message)
        else:
            self.rmsMessages.emit(rms, "Simple calibration mode enabled")

    def finalizingCalibration(
          self,
          leftCameraMatrix,
          leftDistortionCoefficients,
          rightCameraMatrix,
          rightDistortionCoefficients,
          imageSize,
          rotationMatrix,
          translationVector):
        print("Rectifying cameras...")
        # TODO: Why do I care about the disparityToDepthMap?
        (leftRectification,
         rightRectification,
         leftProjection,
         rightProjection,
         dispartityToDepthMap,
         leftROI,
         rightROI) = cv2.stereoRectify(
                        leftCameraMatrix, leftDistortionCoefficients,
                        rightCameraMatrix, rightDistortionCoefficients,
                        imageSize,
                        rotationMatrix, translationVector,
                        None, None, None, None, None,
                        cv2.CALIB_ZERO_DISPARITY,
                        self.OPTIMIZE_ALPHA)

        print("Saving calibration...")
        leftMapX, leftMapY = cv2.initUndistortRectifyMap(
                leftCameraMatrix,
                leftDistortionCoefficients,
                leftRectification,
                leftProjection,
                imageSize,
                cv2.CV_32FC1)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(
                rightCameraMatrix,
                rightDistortionCoefficients,
                rightRectification,
                rightProjection,
                imageSize,
                cv2.CV_32FC1)

        np.savez_compressed(
            CALIBRATION_RESULT,
            imageSize=imageSize,
            leftMapX=leftMapX,
            leftMapY=leftMapY,
            leftROI=leftROI,
            rightMapX=rightMapX,
            rightMapY=rightMapY,
            rightROI=rightROI)

    def calibrateSensor(self):
        filenames = []
        leftFilenames = []
        leftObjectPoints = []
        leftImagePoints = []
        rightFileNames = []
        rightObjectPoints = []
        rightImagePoints = []
        imageSize = None
        try:
            print(f"Loading calib params from file {self.CALIB_PARAMS}")
            calib_params = np.load(self.CALIB_PARAMS)
            filenames = list(calib_params["filenames"])
            leftFilenames = list(calib_params["leftFilenames"])
            leftObjectPoints = list(calib_params["leftObjectPoints"])
            leftImagePoints = list(calib_params["leftImagePoints"])
            rightFileNames = list(calib_params["rightFilenames"])
            rightObjectPoints = list(calib_params["rightObjectPoints"])
            rightImagePoints = list(calib_params["rightImagePoints"])
            imageSize = tuple(calib_params["imageSize"])
        except IOError:
            print("Calib params file file at {0} not found"
                  .format(self.CALIB_PARAMS))
            sys.exit(1)

        if len(calib_params["filenames"]) == 0:
            print("No valid calib images were produced for calibration")
            return

        leftObjectPoints, leftImagePoints =\
            self._getMatchingObjectAndImagePoints(
              filenames,
              leftFilenames,
              leftObjectPoints,
              leftImagePoints)
        rightObjectPoints, rightImagePoints =\
            self._getMatchingObjectAndImagePoints(
                filenames,
                rightFileNames,
                rightObjectPoints,
                rightImagePoints)

        objectPoints = leftObjectPoints

        print("Calibrating left camera...")
        _, leftCameraMatrix, leftDistortionCoefficients, _, _ =\
            cv2.calibrateCamera(
                objectPoints,
                leftImagePoints,
                imageSize,
                None,
                None)
        print("Calibrating right camera...")
        _, rightCameraMatrix, rightDistortionCoefficients, _, _ =\
            cv2.calibrateCamera(
                objectPoints,
                rightImagePoints,
                imageSize,
                None,
                None)

        print("Calibrating cameras together...")
        (rms, _, _, _, _, rotationMatrix, translationVector, _, _) =\
            cv2.stereoCalibrate(
                objectPoints,
                leftImagePoints,
                rightImagePoints,
                leftCameraMatrix,
                leftDistortionCoefficients,
                rightCameraMatrix,
                rightDistortionCoefficients,
                imageSize,
                None,
                None,
                None,
                None,
                cv2.CALIB_FIX_INTRINSIC,
                self.TERMINATION_CRITERIA)

        return (rms, (
          leftCameraMatrix,
          leftDistortionCoefficients,
          rightCameraMatrix,
          rightDistortionCoefficients,
          imageSize,
          rotationMatrix,
          translationVector))
