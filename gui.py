from backend import Backend

from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QMainWindow, QGridLayout
from PyQt5.QtWidgets import QSpinBox, QLabel, QHBoxLayout
from PyQt5.QtWidgets import QWidget, QApplication


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
        self.textureThreshold.setValue(self.worker.sensor.textureThreshold)
        self.min_disp.setRange(-1, 1000)
        self.min_disp.setSingleStep(1)
        self.min_disp.setValue(self.worker.sensor.min_disp)
        self.max_disp.setRange(16, 1000)
        self.max_disp.setSingleStep(16)
        self.max_disp.setValue(self.worker.sensor.max_disp)
        self.blockSize.setRange(5, 255)
        self.blockSize.setSingleStep(2)
        self.blockSize.setValue(self.worker.sensor.blockSize)
        self.smallerBlockSize.setRange(-1, 1000000)
        self.smallerBlockSize.setSingleStep(1)
        self.smallerBlockSize.setValue(self.worker.sensor.smallerBlockSize)
        self.uniquenessRatio.setRange(0, 1000)
        self.uniquenessRatio.setValue(self.worker.sensor.uniquenessRatio)
        self.speckleWindowSize.setRange(-1, 1000)
        self.speckleWindowSize.setValue(self.worker.sensor.speckleWindowSize)
        self.speckleRange.setRange(-1, 1000)
        self.speckleRange.setValue(self.worker.sensor.speckleRange)
        self.disp12MaxDiff.setRange(-1, 1000)
        self.disp12MaxDiff.setValue(self.worker.sensor.disp12MaxDiff)
        self.preFilterType.setRange(0, 1)
        self.preFilterType.setSingleStep(1)
        self.preFilterType.setValue(self.worker.sensor.preFilterType)
        self.preFilterSize.setRange(5, 255)
        self.preFilterSize.setSingleStep(2)
        self.preFilterSize.setValue(self.worker.sensor.preFilterSize)
        self.preFilterCap.setRange(1, 63)
        self.preFilterCap.setSingleStep(1)
        self.preFilterCap.setValue(self.worker.sensor.preFilterCap)

        # wire signals/slots
        self.textureThreshold.valueChanged.connect(
            self.worker.updateTextureThreshold)
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
        print("Exiting app through GUI")
        QApplication.quit()
