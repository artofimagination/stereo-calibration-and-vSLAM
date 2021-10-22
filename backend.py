import traceback
import sys

from sensor import Sensor

from PyQt5.QtCore import QObject, Qt
from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap

import cv2


class BackendSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object, object, object)


class Backend(QObject):
    def __init__(self):
        super(Backend, self).__init__()

        self.signals = BackendSignals()
        self.stop = False
        self.sensor = Sensor()

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

    def updateMax_disp(
          self,
          max_disp):
        self.sensor.max_disp = max_disp

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

    def sendVideoToUI(self, leftFrame, rightFrame, depthMap):
        pix_disp = QPixmap()
        pix0 = QPixmap()
        pix1 = QPixmap()
        if depthMap is not None:
            img = QImage(
                depthMap,
                depthMap.shape[1],
                depthMap.shape[0],
                QImage.Format_RGB888)
            pix_disp = QPixmap.fromImage(img)
            pix_disp = pix_disp.scaled(
                800,
                400,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation)

        if leftFrame is not None:
            img = QImage(
                leftFrame,
                leftFrame.shape[1],
                leftFrame.shape[0],
                QImage.Format_RGB888)
            pix0 = QPixmap.fromImage(img)
            pix0 = pix0.scaled(
                800,
                600,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation)

        if rightFrame is not None:
            img = QImage(
                rightFrame,
                rightFrame.shape[1],
                rightFrame.shape[0],
                QImage.Format_RGB888)
            pix1 = QPixmap.fromImage(img)
            pix1 = pix1.scaled(
                800,
                600,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation)
        self.signals.result.emit(pix0, pix1, pix_disp)

    @QtCore.pyqtSlot()
    def run(self):
        # Retrieve args/kwargs here; and fire processing using them
        try:
            self.sensor.startSensors()
            calibration = self.sensor.loadCalibFile()
            while(True):
                if self.stop:
                    break

                (leftFrame, rightFrame, depthMap) =\
                    self.sensor.createDepthMap(calibration)
                self.sendVideoToUI(leftFrame, rightFrame, depthMap)
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
