import traceback
import sys
from enum import Enum

from sensor import Sensor
from calibration import SensorCalibrator, loadCalibFile
from featureDetector import FeatureDetector
from motionEstimator import MotionEstimator
from blockMatching import BlockMatching, generateDepthMapMask
from mapping import FeatureMap

from PyQt5.QtCore import QObject, Qt
from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap

import cv2
import numpy as np


# Signals used by the backend.
class BackendSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    # sends live data to the gui
    # parameters
    # left camera image in QPixmap
    # right camera image in QPixmap
    # depth map image in QPixmap
    # depth map point cloud
    # feature matches image QPixmap
    # trajectory of our movement
    # feature map recorded during navigation.
    # transformation matrix
    framesSent = QtCore.pyqtSignal(
        QPixmap, QPixmap, QPixmap, object, QPixmap, object, object)
    updateCalibInfo = QtCore.pyqtSignal(object, object)
    updateFeatureInfo = QtCore.pyqtSignal(object)
    rmsLimitUpdated = QtCore.pyqtSignal(float)
    calibImageIndexUpdated = QtCore.pyqtSignal(int)
    cameraIndicesUpdated = QtCore.pyqtSignal(int, int, list)


# Represents the internal states.
class States(Enum):
    # Restarts the sensors with updated config
    UpdateSensorConfig = 0
    # SaveAndProcessImage (creating calibration image and camera info)
    SaveAndProcessImage = 1
    # Idle (spitting images on the UI)
    Idle = 2
    # Calibrate (generates the final calibration data)
    Calibrate = 3


## @enum Modes
#  @brief Describes the modes, the thread cna run in.
class Modes(Enum):
    # No mode is selected.
    NoMode = 0
    # Generating calibration data and images.
    Calibration = 1
    # Generates depth map using block matching
    BlockMatching = 2
    # Does feature extraction and matching
    FeatureDetection = 3
    # Estimates motion trajectory and orientation
    MotionEstimation = 4
    # Puts the point cloud and features in a map.
    Mapping = 5


## @class Backend
#  @brief Controls all business logic in a seperate thread from the UI.
#
#  Logic elements
#   - Generating calibration images and data
#   - Producing final calibration data
#   - Depth map generation using block matching
#   - Feature extraction and matching
#   - Motion Estimation
class Backend(QObject):
    def __init__(self):
        super(Backend, self).__init__()

        self.signals = BackendSignals()
        # Thread break guard condition, when true the thread finishes.
        self.stop = False
        # Sensor contains the camera handling, image retrieval
        # and image processing functions
        self.sensor = Sensor()
        self.sensor.detectSensors()

        # Responsible for creating calibration images and calibration matrices
        self.calibrator = SensorCalibrator()
        self.calibrator.framesUpdated.connect(self.updateVideoPixmap)
        self.calibrator.rmsMessages.connect(self.signals.updateCalibInfo)
        self.calibrator.rmsLimitUpdated.connect(self.signals.rmsLimitUpdated)
        self.calibrator.calibImageIndexUpdated.connect(
            self.signals.calibImageIndexUpdated)

        # Responsible for generating the depth map.
        self.blockMatching = BlockMatching()

        # Responsible to extract and match features from the images and depth map.
        self.featureDetector = FeatureDetector()
        self.featureDetector.infoStrUpdated.connect(self.signals.updateFeatureInfo)

        # Responsible to generate trajectory and orientation
        self.motionEstimator = MotionEstimator()
        # Responsible to store features and depth map in a point cloud.
        self.mapping = FeatureMap()

        # This number represents which frame to process. For example if it is set to 5
        # every 5th frame is being processed. This is to improve performance.
        # Must be >= 1.
        self.frameSkipCount = 1

        # Holds the current execution state
        self.state = States.Idle
        # Holds the current feature mode.
        self.mode = Modes.NoMode

    ###########################################
    # Below are the implementation of all slots connected to the gui
    # Note: they do not follow thread safety. Backend values are updated from UI thread
    # while potentially being accessed in the backend thread.
    ###########################################
    def updateTextureThreshold(self, textureThreshold):
        self.blockMatching.textureThreshold = textureThreshold

    def updateSmallerBlockSize(self, smallerBlockSize):
        self.blockMatching.smallerBlockSize = smallerBlockSize

    def updatePreFilterType(self, preFilterType):
        self.blockMatching.preFilterType = preFilterType

    def updatePrefilterCap(self, preFilterCap):
        self.blockMatching.preFilterCap = preFilterCap

    def updatePrefilterSize(self, preFilterSize):
        self.blockMatching.preFilterSize = preFilterSize

    def updateMin_disp(self, min_disp):
        self.blockMatching.min_disp = min_disp

    def updateNum_disp(self, num_disp):
        self.blockMatching.num_disp = num_disp

    def updateBlockSize(self, blockSize):
        self.blockMatching.blockSize = blockSize

    def updateUniquenessRatio(self, uniquenessRatio):
        self.blockMatching.uniquenessRatio = uniquenessRatio

    def updateSpeckleWindowSize(self, speckleWindowSize):
        self.blockMatching.speckleWindowSize = speckleWindowSize

    def updateSpeckleRange(self, speckleRange):
        self.blockMatching.speckleRange = speckleRange

    def updateDisp12MaxDiff(self, disp12MaxDiff):
        self.blockMatching.disp12MaxDiff = disp12MaxDiff

    def updateDrawEpipolar(self, draw):
        self.blockMatching.drawEpipolar = draw

    def updateBmType(self, bmType):
        self.blockMatching.bmType = bmType

    def updateResolution(self, index):
        if index == 0:
            self.sensor.camera_width = 640
            self.sensor.camera_height = 480
        elif index == 1:
            self.sensor.camera_width = 1280
            self.sensor.camera_height = 720
        self.state = States.UpdateSensorConfig

    def updateCalib_image_index(self, calib_image_index):
        if calib_image_index != self.calibrator.calib_image_index:
            self.calibrator.calib_image_index = calib_image_index

    def updateRms_limit(self, rms_limit):
        if rms_limit != self.calibrator.rms_limit:
            self.calibrator.rms_limit = rms_limit

    def updateMax_rms(self, max_rms):
        self.calibrator.max_rms = max_rms

    def updateIncrement(self, increment):
        self.calibrator.increment = increment

    def saveImage(self):
        self.state = States.SaveAndProcessImage

    def calibrateSensor(self):
        self.state = States.Calibrate

    def setIgnoreExistingImageData(self, value):
        self.calibrator.ignoreExistingImageData = value

    def enableAdvancedCalib(self, value):
        self.calibrator.advancedCalib = value

    def updateLeftCameraIndex(self, left):
        if left != self.sensor.leftIndex:
            self.sensor.leftIndex = left
            self.state = States.UpdateSensorConfig
            self.signals.cameraIndicesUpdated.emit(
                self.sensor.leftIndex,
                self.sensor.rightIndex,
                self.sensor.sensor_indices)

    def updateRightCameraIndex(self, right):
        if right != self.sensor.rightIndex:
            self.sensor.rightIndex = right
            self.state = States.UpdateSensorConfig
            self.signals.cameraIndicesUpdated.emit(
                self.sensor.leftIndex,
                self.sensor.rightIndex,
                self.sensor.sensor_indices)

    def swapSensorIndices(self):
        self.signals.cameraIndicesUpdated.emit(
            self.sensor.rightIndex,
            self.sensor.leftIndex,
            self.sensor.sensor_indices)

    def updateDrawMatches(self, value):
        self.featureDetector.drawMatches = value

    def updateFeatureDetector(self, value):
        self.featureDetector.featureDetector = value

    def updateFeatureMatcher(self, value):
        self.featureDetector.featureMatcher = value

    def updateFrameSkipCount(self, value):
        self.frameSkipCount = value

    def updateInlierLimit(self, value):
        self.motionEstimator.inliersLimit = value

    def updateMaxDepth(self, value):
        self.motionEstimator.maxDepth = value

    def updateReprojectionError(self, value):
        self.motionEstimator.reprojectionError = value

    def updateMatchDistanceThreshold(self, value):
        self.featureDetector.matchDistanceThreshold = value

    ###########################################
    # Slot implementation ends
    ###########################################

    # Returns the current sesnor indices and sensor index list.
    def getSensorIndices(self):
        return (self.sensor.leftIndex,
                self.sensor.rightIndex,
                self.sensor.sensor_indices)

    ## @brief Converts the incoming frames to QPixmap in order to visualize in Qt UI.
    #
    # @param leftFrame -- left camera frame
    # @param rightFrame -- right camera frame
    # @param depthMap -- computed depth map
    # @param imageMatches -- detected feature matches drawn on consecutive image pair.
    def updateVideoPixmap(self, leftFrame, rightFrame, depthMap, imageMatches):
        leftPix = QPixmap()
        rightPix = QPixmap()
        depthPix = QPixmap()
        matchesPix = QPixmap()
        if depthMap is not None:
            rgbDepth = cv2.cvtColor(
                depthMap.astype(np.uint8),
                cv2.COLOR_GRAY2RGB)

            img = QImage(
                rgbDepth,
                rgbDepth.shape[1],
                rgbDepth.shape[0],
                QImage.Format_RGB888)
            depthPix = QPixmap.fromImage(img)
            depthPix = depthPix.scaled(
                600,
                350,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation)

        if leftFrame is not None:
            leftFrame = cv2.cvtColor(
                leftFrame, cv2.COLOR_BGR2RGB)
            img = QImage(
                leftFrame,
                leftFrame.shape[1],
                leftFrame.shape[0],
                QImage.Format_RGB888)
            leftPix = QPixmap.fromImage(img)
            leftPix = leftPix.scaled(
                600,
                350,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation)

        if rightFrame is not None:
            rightFrame = cv2.cvtColor(
                rightFrame, cv2.COLOR_BGR2RGB)
            img = QImage(
                rightFrame,
                rightFrame.shape[1],
                rightFrame.shape[0],
                QImage.Format_RGB888)
            rightPix = QPixmap.fromImage(img)
            rightPix = rightPix.scaled(
                600,
                350,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation)

        if imageMatches is not None:
            imageMatches = cv2.cvtColor(
                imageMatches, cv2.COLOR_BGR2RGB)
            img = QImage(
                imageMatches,
                imageMatches.shape[1],
                imageMatches.shape[0],
                QImage.Format_RGB888)
            matchesPix = QPixmap.fromImage(img)
        return (leftPix, rightPix, depthPix, matchesPix)

    ## @brief Main entry point of the thread worker.
    #
    #  Handles different backend states and feature modes
    #  Modes:
    #    - Calibration
    #    - BlockMatching
    #    - Feature detection
    #    - Motion estimation
    #    - Mapping
    #  Each mode can have multiple internal states.
    @QtCore.pyqtSlot()
    def run(self):
        try:
            frame = None
            depth_pointcloud = None
            image_matches = None
            trajectory = None
            featureMap = None
            leftPix = QPixmap()
            rightPix = QPixmap()
            depthPix = QPixmap()
            matchesPix = QPixmap()
            frameSkipCount = 0
            calibrationFileLoaded = False

            self.sensor.startSensors()
            self.signals.cameraIndicesUpdated.emit(
                self.sensor.leftIndex,
                self.sensor.rightIndex,
                self.sensor.sensor_indices)

            while(True):
                if self.stop:
                    break

                if self.mode == Modes.Calibration:
                    if self.state == States.Idle:
                        frame = self.sensor.captureFrame()

                    if self.state == States.UpdateSensorConfig:
                        self.sensor.restartSensors()
                        self.state = States.Idle

                    if self.state == States.SaveAndProcessImage:
                        frame = self.sensor.captureFrame()
                        self.calibrator.saveAndProcessImage(frame)
                        if self.calibrator.advancedCalib:
                            rms, _ = self.calibrator.calibrateSensor()
                            self.calibrator.checkRMS(rms)
                        self.state = States.Idle

                    if self.state == States.Calibrate:
                        _, params = self.calibrator.calibrateSensor()
                        self.calibrator.finalizingCalibration(*params)
                        calibrationFileLoaded = False
                        self.state = States.Idle

                elif self.mode == Modes.BlockMatching:
                    if self.state == States.Idle:
                        if calibrationFileLoaded is False:
                            calibration = loadCalibFile()
                            calibrationFileLoaded = True
                        frame = self.sensor.captureFrame()
                        # Do processing only in every xth cycle.
                        if frameSkipCount == self.frameSkipCount:
                            (frame, depth_pointcloud) =\
                                self.blockMatching.createDepthMap(
                                    calibration, frame)
                            frameSkipCount = 0
                        frameSkipCount += 1

                    if self.state == States.UpdateSensorConfig:
                        self.sensor.restartSensors()
                        self.state = States.Idle

                elif self.mode == Modes.FeatureDetection:
                    if self.state == States.Idle:
                        frame = self.sensor.captureFrame()
                        self.featureDetector.set_frame(frame.leftImage)
                        (_, _, _, image_matches) =\
                            self.featureDetector.detect_features(None)

                elif self.mode == Modes.MotionEstimation:
                    if self.state == States.Idle:
                        if calibrationFileLoaded is False:
                            calibration = loadCalibFile()
                            calibrationFileLoaded = True
                        frame = self.sensor.captureFrame()

                        # Do processing only in every xth cycle.
                        if frameSkipCount == 1:
                            frameSkipCount = 0
                            (frame, depth_pointcloud) =\
                                self.blockMatching.createDepthMap(calibration, frame)
                            self.featureDetector.set_frame(frame.leftImage)
                            mask = generateDepthMapMask(depth_pointcloud, frame.leftImage)
                            (kp0, kp1, matches, image_matches) =\
                                self.featureDetector.detect_features(mask)
                            if kp0 is not None and kp1 is not None:
                                rmat, tvec, _, _, _ =\
                                    self.motionEstimator.estimate_motion(
                                        matches,
                                        kp0,
                                        kp1,
                                        calibration["leftCameraMatrix"],
                                        depth_pointcloud)
                                trajectory, _ = self.motionEstimator.calculate_trajectory(
                                    rmat,
                                    tvec)
                        frameSkipCount += 1

                    if self.state == States.UpdateSensorConfig:
                        self.sensor.restartSensors()
                        self.state = States.Idle
                elif self.mode == Modes.Mapping:
                    if self.state == States.Idle:
                        if calibrationFileLoaded is False:
                            calibration = loadCalibFile()
                            calibrationFileLoaded = True
                        frame = self.sensor.captureFrame()

                        # Do processing only in every xth cycle.
                        if frameSkipCount == 1:
                            frameSkipCount = 0
                            (frame, depth_pointcloud) =\
                                self.blockMatching.createDepthMap(calibration, frame)
                            self.featureDetector.set_frame(frame.leftImage)
                            mask = generateDepthMapMask(depth_pointcloud, frame.leftImage)
                            (kp0, kp1, matches, image_matches) =\
                                self.featureDetector.detect_features(mask)
                            if kp0 is not None and kp1 is not None:
                                rmat, tvec, img0, img1, object_points =\
                                    self.motionEstimator.estimate_motion(
                                        matches,
                                        kp0,
                                        kp1,
                                        calibration["leftCameraMatrix"],
                                        depth_pointcloud)
                                trajectory, worldTransform =\
                                    self.motionEstimator.calculate_trajectory(
                                        rmat,
                                        tvec)
                                if object_points is not None:
                                    self.mapping.init_landmarks(
                                        object_points,
                                        worldTransform)
                                featureMap = self.mapping.get_pointcloud()
                        frameSkipCount += 1

                (leftPix, rightPix, depthPix, matchesPix) = self.updateVideoPixmap(
                    frame.leftImage, frame.rightImage, depth_pointcloud, image_matches)
                self.signals.framesSent.emit(
                    leftPix,
                    rightPix,
                    depthPix,
                    depth_pointcloud,
                    matchesPix,
                    trajectory,
                    featureMap)
        except Exception:
            self.sensor.releaseVideoDevices()
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.sensor.releaseVideoDevices()
            self.signals.finished.emit()  # Done
