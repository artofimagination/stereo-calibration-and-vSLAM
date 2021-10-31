import traceback
import sys
from enum import Enum

from sensor import Sensor
from calibration import SensorCalibrator

from PyQt5.QtCore import QObject, Qt
from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap

import cv2


# Signals used by the backend.
class BackendSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    framesSent = QtCore.pyqtSignal(object, object, object)
    updateCalibInfo = QtCore.pyqtSignal(object, object)
    rmsLimitUpdated = QtCore.pyqtSignal(float)
    calibImageIndexUpdated = QtCore.pyqtSignal(int)


class States(Enum):
    # Restarts the sensors with updated config
    UpdateSensorConfig = 0
    # SaveAndProcessImage (creating calibration image and camera info)
    SaveAndProcessImage = 1
    # Idle (spitting images on the UI)
    Idle = 2
    # Calibrate (generates the final calibration data)
    Calibrate = 3
    # BlockMatching (generates depth map using block matching)
    BlockMatching = 4


# Controls all business logic in a seperate thread from the UI.
# Logic elements
# Generating calibration images and data
# Producing final calibration data
# Depth map generation using block matching
class Backend(QObject):
    def __init__(self):
        super(Backend, self).__init__()

        self.signals = BackendSignals()
        # Thread break guard condition, when true the thread finishes.
        self.stop = False
        # Sensor contains the camera handling, image retrieval
        # and image processing functions
        self.sensor = Sensor()
        # Responsible for creating calibration images and calibration matrices
        self.calibrator = SensorCalibrator()
        self.calibrator.framesUpdated.connect(self.updateVideoPixmap)
        self.calibrator.rmsMessages.connect(self.signals.updateCalibInfo)
        self.calibrator.rmsLimitUpdated.connect(self.signals.rmsLimitUpdated)
        self.calibrator.calibImageIndexUpdated.connect(
            self.signals.calibImageIndexUpdated)

        # Current left camera frame
        self.leftFrame = None
        # Current right camera frame
        self.rightFrame = None
        self.state = States.Idle

        # Current left camera frame converted to QPixmap for UI displaying.
        self.leftPix = QPixmap()
        # Current right camera frame converted to QPixmap for UI displaying.
        self.rightPix = QPixmap()
        # Current depth map frame converted to QPixmap for UI displaying.
        self.depthPix = QPixmap()

    ###########################################
    # Below are the implementation of all slots connected to the gui
    ###########################################
    def updateTextureThreshold(
          self,
          textureThreshold):
        self.sensor.textureThreshold = textureThreshold

    def updateSmallerBlockSize(
          self,
          smallerBlockSize):
        self.sensor.smallerBlockSize = smallerBlockSize

    def updatePreFilterType(
          self,
          preFilterType):
        self.sensor.preFilterType = preFilterType

    def updatePrefilterCap(
          self,
          preFilterCap):
        self.sensor.preFilterCap = preFilterCap

    def updatePrefilterSize(
          self,
          preFilterSize):
        self.sensor.preFilterSize = preFilterSize

    def updateMin_disp(
          self,
          min_disp):
        self.sensor.min_disp = min_disp

    def updateNum_disp(
          self,
          num_disp):
        self.sensor.num_disp = num_disp

    def updateBlockSize(
          self,
          blockSize):
        self.sensor.blockSize = blockSize

    def updateUniquenessRatio(
          self,
          uniquenessRatio):
        self.sensor.uniquenessRatio = uniquenessRatio

    def updateSpeckleWindowSize(
          self,
          speckleWindowSize):
        self.sensor.speckleWindowSize = speckleWindowSize

    def updateSpeckleRange(
          self,
          speckleRange):
        self.sensor.speckleRange = speckleRange

    def updateDisp12MaxDiff(
          self,
          disp12MaxDiff):
        self.sensor.disp12MaxDiff = disp12MaxDiff

    def updateP1(self, p1):
        self.sensor.P1 = p1

    def updateP2(self, p2):
        self.sensor.P2 = p2

    def updateDrawEpipolar(self, draw):
        self.sensor.drawEpipolar = draw

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

    def updateBmType(self, bmType):
        self.sensor.bmType = bmType

    def saveImage(self):
        self.state = States.SaveAndProcessImage

    def calibrateSensor(self):
        self.state = States.Calibrate

    def setIgnoreExistingImageData(self, value):
        self.calibrator.ignoreExistingImageData = value

    def enableAdvancedCalib(self, value):
        self.calibrator.advancedCalib = value

    ###########################################
    # Slot implementation ends
    ###########################################

    # Converts the incoming frames to QPixmap in order to visualize in Qt UI.
    def updateVideoPixmap(self, leftFrame, rightFrame, depthMap):
        if depthMap is not None:
            img = QImage(
                depthMap,
                depthMap.shape[1],
                depthMap.shape[0],
                QImage.Format_RGB888)
            self.depthPix = QPixmap.fromImage(img)
            self.depthPix = self.depthPix.scaled(
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
            self.leftPix = QPixmap.fromImage(img)
            self.leftPix = self.leftPix.scaled(
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
            self.rightPix = QPixmap.fromImage(img)
            self.rightPix = self.rightPix.scaled(
                600,
                350,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation)
        self.signals.framesSent.emit(
                  self.leftPix, self.rightPix, self.depthPix)

    # Main entry point.
    # Handles different backend states
    #   UpdateSensorConfig (restarts the sensors if their config has changed)
    #   Idle (spitting images on the UI)
    #   SaveAndProcessImage (creating calibration image and camera info)
    #   Calibrate (generates the final calibration data)
    #   BlockMatching (generates depth map using block matching)
    @QtCore.pyqtSlot()
    def run(self):
        self.sensor.startSensors()
        calibrationFileLoaded = False
        try:
            while(True):
                if self.stop:
                    break

                if self.state == States.UpdateSensorConfig:
                    self.sensor.restartSensors()
                    self.state = States.Idle

                if self.state == States.Idle:
                    (self.leftFrame, self.rightFrame) =\
                        self.sensor.captureFrame()
                    self.updateVideoPixmap(
                        self.leftFrame, self.rightFrame, None)

                if self.state == States.SaveAndProcessImage:
                    self.calibrator.saveAndProcessImage(
                        self.leftFrame, self.rightFrame)
                    if self.calibrator.advancedCalib:
                        rms, _ = self.calibrator.calibrateSensor()
                        self.calibrator.checkRMS(rms)
                    self.state = States.Idle

                if self.state == States.Calibrate:
                    _, params = self.calibrator.calibrateSensor()
                    self.calibrator.finalizingCalibration(*params)
                    self.state = States.Idle

                if self.state == States.BlockMatching:
                    if calibrationFileLoaded is False:
                        calibration = self.sensor.loadCalibFile()
                        calibrationFileLoaded = True
                    (leftFrame, rightFrame, depthMap) =\
                        self.sensor.createDepthMap(calibration)
                    self.updateVideoPixmap(leftFrame, rightFrame, depthMap)
                else:
                    calibrationFileLoaded = False

                self.signals.framesSent.emit(
                    self.leftPix, self.rightPix, self.depthPix)
        except Exception:
            self.sensor.left.release()
            self.sensor.right.release()
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.sensor.left.release()
            self.sensor.right.release()
            self.signals.finished.emit()  # Done
