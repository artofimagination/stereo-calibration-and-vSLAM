import sys
import glob
import os

import cv2
import numpy as np

# Stereo calibration result file
CALIBRATION_RESULT = "output/calibration.npz"

# Camera resolution
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


class SensorCalibrator():
    # Location where the chessboard images to be stored
    LEFT_IMAGE_DIR = "calibImages/left"
    RIGHT_IMAGE_DIR = "calibImages/right"
    LEFT_PATH = "{LEFT_IMAGE_DIR}/{:06d}.jpg"
    RIGHT_PATH = "{RIGHT_IMAGE_DIR}/{:06d}.jpg"
    CALIB_PARAMS = "output/calibParams.npz"
    # Chessboard pattern size
    CHESSBOARD_SIZE = (9, 6)
    # Empty object point table.
    OBJECT_POINT_ZERO = np.zeros(
        (CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3),
        np.float32)
    # Termination criteria for sub pixel corner search and stereo calibrate.
    TERMINATION_CRITERIA = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001)
    OPTIMIZE_ALPHA = 0.25

    def __init__(self):
        # Last RMS value, that was within limits.
        self.last_accepted_rms = 100
        # Index of the last stored chess board calibration image.
        self.calib_image_index = 23
        # Accepted RMS limit, all images that would
        # increase the RMS over the limit are thrown away.
        self.rms_limit = 0.315
        # Increment of RMS limit change
        self.increment = 0.01
        # When the RMS value goes above this value, the
        # increment will change to decrement strictening the RMS condition
        self.turnaround_limit = self.rms_limit - 0.01
        # Determines whether it is tolerance extension or reduction phase
        self.trend_up = True

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

        # Grab both frames first, then retrieve
        # to minimize latency between cameras.
        while(True):
            if self.calib_image_index > self.MAX_IMAGES:
                print("Calibration image count limit reached")
                break

            if not (left.grab() and right.grab()):
                print("No more frames")
                break

            _, leftFrame = left.retrieve()
            _, rightFrame = right.retrieve()

            if leftFrame is not None and rightFrame is not None:
                numpy_horizontal_concat = np.concatenate(
                    (cv2.resize(leftFrame, (640, 480)),
                     cv2.resize(rightFrame, (640, 480))),
                    axis=1)
                cv2.imshow('Stereo image', numpy_horizontal_concat)
                ret = cv2.waitKey(1)
                if ret & 0xFF == ord('n'):
                    left_path = self.LEFT_PATH.format(self.calib_image_index)
                    right_path = self.RIGHT_PATH.format(self.calib_image_index)
                    print(
                        f"Store image Left {left_path}")
                    print(
                        f"Store image Right {right_path}")
                    cv2.imwrite(left_path, leftFrame)
                    cv2.imwrite(right_path, rightFrame)
                    self.calib_image_index += 1
                    break
                if ret & 0xFF == ord('q'):
                    left.release()
                    right.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)

        left.release()
        right.release()
        cv2.destroyAllWindows()

    def _readImagesAndFindChessboards(self, imageDirectory):
        cacheFile = "{0}/chessboards.npz".format(imageDirectory)
        try:
            cache = np.load(cacheFile)
            print(f"Loading image data from cache file at {cacheFile}")
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
            cv2.imshow(imageDirectory, image)

            # Needed to draw the window
            cv2.waitKey(1)

        cv2.destroyWindow(imageDirectory)

        print("Found corners in {0} out of {1} images"
              .format(len(imagePoints), len(imagePaths)))

        np.savez_compressed(
            cacheFile,
            filenames=filenames,
            objectPoints=objectPoints,
            imagePoints=imagePoints,
            imageSize=imageSize)
        return filenames, imagePaths, objectPoints, imagePoints, imageSize

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
            filenamesSorted=filenamesSorted,
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

    def calibrateSensor(self):
        try:
            print(f"Loading calib params from file {self.CALIB_PARAMS}")
            calib_params = np.load(self.CALIB_PARAMS)
        except IOError:
            print("Calib params file file at {0} not found"
                  .format(self.CALIB_PARAMS))
            sys.exit(1)

        leftObjectPoints, leftImagePoints =\
            self._getMatchingObjectAndImagePoints(
              calib_params["filenames"],
              calib_params["leftFilenames"],
              calib_params["leftObjectPoints"],
              calib_params["leftImagePoints"])
        rightObjectPoints, rightImagePoints =\
            self._getMatchingObjectAndImagePoints(
                calib_params["filenames"],
                calib_params["rightFilenames"],
                calib_params["rightObjectPoints"],
                calib_params["rightImagePoints"])

        objectPoints = leftObjectPoints

        print("Calibrating left camera...")
        _, leftCameraMatrix, leftDistortionCoefficients, _, _ =\
            cv2.calibrateCamera(
                objectPoints,
                calib_params["leftImagePoints"],
                calib_params["imageSize"],
                None,
                None)
        print("Calibrating right camera...")
        _, rightCameraMatrix, rightDistortionCoefficients, _, _ =\
            cv2.calibrateCamera(
                objectPoints,
                calib_params["rightImagePoints"],
                calib_params["imageSize"],
                None,
                None)

        print("Calibrating cameras together...")
        (rms, _, _, _, _, rotationMatrix, translationVector, _, _) =\
            cv2.stereoCalibrate(
                calib_params["objectPoints"],
                calib_params["leftImagePoints"],
                calib_params["rightImagePoints"],
                leftCameraMatrix,
                leftDistortionCoefficients,
                rightCameraMatrix,
                rightDistortionCoefficients,
                calib_params["imageSize"],
                None,
                None,
                None,
                None,
                cv2.CALIB_FIX_INTRINSIC,
                self.TERMINATION_CRITERIA)
        print(f"Current RMS {rms}, last valid RMS {self.global_rms}")
        print(f"MS increment {self.increment}, is trend up? {self.trendUp}")
        print(f"Image index {self.calib_image_index}")
        if rms > self.rms_limit:
            print(f"RMS is too high (> {self.rms_limit}).\
Throwing away current chessboard image pair...")
            self.calib_image_index -= 1
            os.remove(
                "{}/{:06d}.jpg"
                .format(self.LEFT_IMAGE_DIR, self.calib_image_index))
            os.remove(
              "{}/{:06d}.jpg"
              .format(self.RIGHT_IMAGE_DIR, self.calib_image_index))
        else:
            self.last_accepted_rms = rms
            if self.rms_limit < self.turnaround_limit:
                if self.trend_up and rms > self.turnaround_limit:
                    print("Switching to downwards trend...")
                    # Downwards trend is slower
                    self.trend_up = False
                    self.increment /= -2.0

                self.rms_limit += self.increment

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
                        calib_params["imageSize"],
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
                calib_params["imageSize"],
                cv2.CV_32FC1)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(
                rightCameraMatrix,
                rightDistortionCoefficients,
                rightRectification,
                rightProjection,
                calib_params["imageSize"],
                cv2.CV_32FC1)

        np.savez_compressed(
            CALIBRATION_RESULT,
            imageSize=calib_params["imageSize"],
            leftMapX=leftMapX,
            leftMapY=leftMapY,
            leftROI=leftROI,
            rightMapX=rightMapX,
            rightMapY=rightMapY,
            rightROI=rightROI)
