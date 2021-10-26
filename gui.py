import os
import numpy as np
import glob
import shutil
from pathlib import Path

from backend import Backend, States

from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QMainWindow, QGridLayout, QAction
from PyQt5.QtWidgets import QSpinBox, QLabel, QHBoxLayout, QFileDialog
from PyQt5.QtWidgets import QWidget, QApplication, QCheckBox
from PyQt5.QtWidgets import QPushButton, QTabWidget, QVBoxLayout
from PyQt5.QtWidgets import QDoubleSpinBox


class CalibWidget(QWidget):
    takeImageTriggered = QtCore.pyqtSignal()

    def keyPressEvent(self, event):
        super(CalibWidget, self).keyPressEvent(event)
        if event.key() == QtCore.Qt.Key_N:
            self.takeImageTriggered.emit()


class MainWindow(QMainWindow):
    SETTINGS_DIR = "settings"

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.workerThread = QThread(self)
        self.worker = Backend()
        self.worker.signals.framesSent.connect(self.updateVideo)
        self.worker.signals.finished.connect(self.thread_complete)
        self.worker.signals.error.connect(self.sigint_handler)
        self.mode = "none"
        mainLayout = QGridLayout()

        self._createMenu()

        calibratorLayout = self._createCalibrationUI()
        self.calibratorLayoutWidget = CalibWidget()
        self.calibratorLayoutWidget.setLayout(calibratorLayout)

        bmConfiguratorLayout = self._createBlockMatchingConfiguratorUI()
        bmConfiguratorLayoutWidget = QWidget()
        bmConfiguratorLayoutWidget.setLayout(bmConfiguratorLayout)
        tabwidget = QTabWidget()
        tabwidget.addTab(self.calibratorLayoutWidget, "Sensor calibration")
        tabwidget.addTab(
            bmConfiguratorLayoutWidget, "Block Matching Configurator")

        self._initUIElements()
        mainLayout.addWidget(tabwidget, 0, 0)
        mainWidget = QWidget()
        mainWidget.setLayout(mainLayout)
        self.setCentralWidget(mainWidget)

        if not os.path.isdir(self.SETTINGS_DIR):
            os.mkdir(self.SETTINGS_DIR)

        self.show()

    def _saveValues(self, settingsName):
        np.savez_compressed(
            settingsName,
            bm_textureThreshold=self.textureThreshold.value(),
            bm_min_disp=self.min_disp.value(),
            bm_num_disp=self.num_disp.value(),
            bm_blocksize=self.blockSize.value(),
            bm_uniquenessRatio=self.uniquenessRatio.value(),
            bm_speckleWindowSize=self.speckleWindowSize.value(),
            bm_speckleRange=self.speckleRange.value(),
            bm_disp12MaxDiff=self.disp12MaxDiff.value(),
            bm_preFilterType=self.preFilterType.value(),
            bm_preFilterSize=self.preFilterSize.value(),
            bm_preFilterCap=self.preFilterCap.value(),
            bm_smallerBlockSize=self.smallerBlockSize.value(),
            cal_calib_image_index=self.calib_image_index.value(),
            cal_rms_limit=self.rms_limit.value(),
            cal_advanced=self.advanced.isChecked(),
            cal_ignoreExitingImageData=self
                                       .ignoreExistingImageData
                                       .isChecked(),
            cal_rms_increment=self.increment.value(),
            cal_max_rms=self.max_rms.value())

        settingsPath = Path(settingsName).with_suffix('')
        settingsPath = settingsPath.stem

        files = glob.glob(f"calibImages/left/{str(settingsPath)}/*")
        for f in files:
            os.remove(f)

        files = glob.glob(f"calibImages/right/{str(settingsPath)}/*")
        for f in files:
            os.remove(f)

        leftDirectory = f"calibImages/left/{str(settingsPath)}"
        if not os.path.isdir(leftDirectory):
            os.mkdir(leftDirectory)

        files = glob.glob("calibImages/left/*")
        for f in files:
            if not os.path.isdir(f):
                shutil.copy(f, leftDirectory)

        rightDirectory = f"calibImages/right/{str(settingsPath)}"
        if not os.path.isdir(rightDirectory):
            os.mkdir(rightDirectory)

        files = glob.glob("calibImages/right/*")
        for f in files:
            if not os.path.isdir(f):
                shutil.copy(f, rightDirectory)

    def saveSettings(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(
            self,
            "QFileDialog.getSaveFileName()", "", "npz (*.npz)",
            options=options)
        if fileName:
            self._saveValues(fileName)
            self._saveValues(f"{self.SETTINGS_DIR}/lastSaved.npz")

    def _setLoadedValues(self, settingsName):
        files = glob.glob("calibImages/left/*")
        for f in files:
            if not os.path.isdir(f):
                os.remove(f)

        files = glob.glob("calibImages/right/*")
        for f in files:
            if not os.path.isdir(f):
                os.remove(f)

        settings = np.load(settingsName)
        self.textureThreshold.setValue(settings["bm_textureThreshold"])
        self.min_disp.setValue(settings["bm_min_disp"])
        self.num_disp.setValue(settings["bm_num_disp"])
        self.blockSize.setValue(settings["bm_blocksize"])
        self.uniquenessRatio.setValue(settings["bm_uniquenessRatio"])
        self.speckleWindowSize.setValue(
            settings["bm_speckleWindowSize"])
        self.speckleRange.setValue(settings["bm_speckleRange"])
        self.disp12MaxDiff.setValue(settings["bm_disp12MaxDiff"])
        self.preFilterType.setValue(settings["bm_preFilterType"])
        self.preFilterSize.setValue(settings["bm_preFilterSize"])
        self.preFilterCap.setValue(settings["bm_preFilterCap"])
        self.smallerBlockSize.setValue(settings["bm_smallerBlockSize"])
        self.calib_image_index.setValue(settings["cal_calib_image_index"])
        self.rms_limit.setValue(settings["cal_rms_limit"])
        print(settings["cal_advanced"])
        self.advanced.setChecked(bool(settings["cal_advanced"]))
        self.ignoreExistingImageData.setChecked(
            bool(settings["cal_ignoreExitingImageData"]))
        self.increment.setValue(settings["cal_rms_increment"])
        self.max_rms.setValue(settings["cal_max_rms"])

        settingsPath = Path(settingsName).with_suffix('')
        settingsPath = settingsPath.stem
        directory = f"calibImages/left/{str(settingsPath)}"
        if os.path.isdir(str(directory)):
            files = glob.glob(f"{directory}/*")
            for f in files:
                shutil.copy(f, "calibImages/left/")
        else:
            print(
                f"No left calib images to load \
(calibImages/left/{str(settingsPath)})")

        directory = f"calibImages/right/{str(settingsPath)}"
        if os.path.isdir(str(directory)):
            files = glob.glob(f"{directory}/*")
            for f in files:
                shutil.copy(f, "calibImages/right/")
        else:
            print(f"No left calib images to load \
(calibImages/left/{str(settingsPath)})")

    def loadSettings(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()", "", "npz (*.npz)",
            options=options)
        if fileName:
            try:
                self._setLoadedValues(fileName)
            except IOError:
                print("Settings file at {0} not found"
                      .format(fileName))
                self.sigint_handler()

    def _createMenu(self):
        saveAction = QAction('&Save Settings', self)
        saveAction.setShortcut('Ctrl+S')
        saveAction.setStatusTip('Save settings')
        saveAction.triggered.connect(self.saveSettings)

        loadAction = QAction('&Load Settings', self)
        loadAction.setShortcut('Ctrl+L')
        loadAction.setStatusTip('Load settings')
        loadAction.triggered.connect(self.loadSettings)

        exitAction = QAction('&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.sigint_handler)

        # Create menu bar and add action
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')
        fileMenu.addAction(saveAction)
        fileMenu.addAction(loadAction)
        fileMenu.addAction(exitAction)

    def _createCalibrationUI(self):
        layout = QGridLayout()
        helpLabel = QLabel("Quick user guide:\n\
    1. To start calibration interface, press start.\n\
    2. Once started, to capture a pair of frames, press 'n' or 'Take image'\
when your sensor's are in the desired position.\n\
    3. When enough calib images are create press process, to create the final\
calibration file.\n\n\
Use modes:\n\
    a, Simple (Default): capture as many calibration images as you want, \
when done press 'Process'.\n\
    b, Advanced: during capture the system will analyse the calibration RMS\
and will throw away if there is ahigh RMS result. See more in README.md\n\
       When finished press 'Process'")

        calibInfoLabel = QLabel("Calibration info")
        self.calibInfo = QLabel()

        RMSLabel = QLabel("RMS")
        self.RMSValue = QLabel()
        labelLayout = QVBoxLayout()
        labelLayout.addWidget(helpLabel)
        labelLayout.addWidget(calibInfoLabel)
        labelLayout.addWidget(self.calibInfo)
        labelLayout.addWidget(RMSLabel)
        labelLayout.addWidget(self.RMSValue)
        labelsLayoutWidget = QWidget()
        labelsLayoutWidget.setLayout(labelLayout)

        ignoreExistingImageDataLabel = QLabel("Ignore existing image data")
        self.ignoreExistingImageData = QCheckBox()
        self.ignoreExistingImageData.setDisabled(True)
        advancedLabel = QLabel("Advanced use mode")
        self.advanced = QCheckBox()
        self.advanced.setDisabled(True)
        optionsLayout = QHBoxLayout()
        optionsLayout.addWidget(ignoreExistingImageDataLabel)
        optionsLayout.addWidget(self.ignoreExistingImageData)
        optionsLayout.addWidget(advancedLabel)
        optionsLayout.addWidget(self.advanced)
        optionsLayoutWidget = QWidget()
        optionsLayoutWidget.setLayout(optionsLayout)

        self.calib_image_index = QSpinBox()
        self.rms_limit = QDoubleSpinBox()
        self.increment = QDoubleSpinBox()
        self.max_rms = QDoubleSpinBox()
        configLayout = QHBoxLayout()
        configLayout.addWidget(QLabel("calib_image_index"))
        configLayout.addWidget(self.calib_image_index)
        configLayout.addWidget(QLabel("rms_limit"))
        configLayout.addWidget(self.rms_limit)
        configLayout.addWidget(QLabel("increment"))
        configLayout.addWidget(self.increment)
        configLayout.addWidget(QLabel("max_rms"))
        configLayout.addWidget(self.max_rms)
        configLayoutWidget = QWidget()
        configLayoutWidget.setLayout(configLayout)
        self.calib_image_index.setRange(0, 1000)
        self.calib_image_index.setSingleStep(1)
        self.rms_limit.setRange(0, 10.0)
        self.rms_limit.setDecimals(5)
        self.rms_limit.setSingleStep(0.005)
        self.increment.setRange(0, 1.0)
        self.increment.setDecimals(5)
        self.increment.setSingleStep(0.005)
        self.max_rms.setRange(0, 10.0)
        self.max_rms.setDecimals(5)
        self.max_rms.setSingleStep(0.005)

        self.video0Calib = QLabel()
        self.video1Calib = QLabel()
        layout.addWidget(self.video0Calib, 0, 0, 1, 4)
        layout.addWidget(self.video1Calib, 0, 4, 1, 4)

        start = QPushButton("Start")
        self.process = QPushButton("Process")
        self.process.hide()
        self.takeImage = QPushButton("Take image")
        self.takeImage.hide()
        start.clicked.connect(self._startCalibration)
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(start)
        buttonLayout.addWidget(self.process)
        buttonLayout.addWidget(self.takeImage)
        buttonLayoutWidget = QWidget()
        buttonLayoutWidget.setLayout(buttonLayout)

        layout.addWidget(self.video0Calib, 0, 0, 1, 4)
        layout.addWidget(self.video1Calib, 0, 4, 1, 4)
        layout.addWidget(labelsLayoutWidget, 1, 0, 1, 8)
        layout.addWidget(optionsLayoutWidget, 2, 0, 1, 8)
        layout.addWidget(configLayoutWidget, 3, 0, 1, 8)
        layout.addWidget(buttonLayoutWidget, 4, 0, 1, 8)
        return layout

    def _createBlockMatchingConfiguratorUI(self):
        self.win_sizeUpdated = QtCore.pyqtSignal(int)
        self.min_dispUpdated = QtCore.pyqtSignal(int)
        self.num_dispUpdated = QtCore.pyqtSignal(int)
        self.blockSizeUpdated = QtCore.pyqtSignal(int)
        self.uniquenessRatioUpdated = QtCore.pyqtSignal(int)
        self.speckleWindowSizeUpdated = QtCore.pyqtSignal(int)
        self.speckleRangeUpdated = QtCore.pyqtSignal(int)
        self.disp12MaxDiffUpdated = QtCore.pyqtSignal(int)
        layout = QGridLayout()

        self.video0Bm = QLabel()
        self.video1Bm = QLabel()
        self.video_disp = QLabel()
        self.textureThreshold = QSpinBox()
        self.min_disp = QSpinBox()
        self.num_disp = QSpinBox()
        self.blockSize = QSpinBox()
        self.uniquenessRatio = QSpinBox()
        self.speckleWindowSize = QSpinBox()
        self.speckleRange = QSpinBox()
        self.disp12MaxDiff = QSpinBox()
        self.preFilterSize = QSpinBox()
        self.preFilterType = QSpinBox()
        self.preFilterCap = QSpinBox()
        self.smallerBlockSize = QSpinBox()
        self.start = QPushButton("Start")
        self.start.clicked.connect(self._startBMConfiguration)

        # Init spin boxes
        self.textureThreshold.setRange(0, 10000)
        self.min_disp.setRange(-1, 1000)
        self.min_disp.setSingleStep(1)
        self.num_disp.setRange(16, 1000)
        self.num_disp.setSingleStep(16)
        self.blockSize.setRange(5, 255)
        self.blockSize.setSingleStep(2)
        self.smallerBlockSize.setRange(-1, 1000000)
        self.smallerBlockSize.setSingleStep(1)
        self.uniquenessRatio.setRange(0, 1000)
        self.speckleWindowSize.setRange(-1, 1000)
        self.speckleRange.setRange(-1, 1000)
        self.disp12MaxDiff.setRange(-1, 1000)
        self.preFilterType.setRange(0, 1)
        self.preFilterType.setSingleStep(1)
        self.preFilterSize.setRange(5, 255)
        self.preFilterSize.setSingleStep(2)
        self.preFilterCap.setRange(1, 63)
        self.preFilterCap.setSingleStep(1)

        layout.addWidget(self.video0Bm, 0, 0, 1, 4)
        layout.addWidget(self.video1Bm, 0, 4, 1, 4)
        layout.addWidget(self.video_disp, 1, 2, 1, 4)
        spinlayout = QHBoxLayout()
        spinlayout.addWidget(QLabel("textureThreshold"))
        spinlayout.addWidget(self.textureThreshold)
        spinlayout.addWidget(QLabel("min_disp"))
        spinlayout.addWidget(self.min_disp)
        spinlayout.addWidget(QLabel("num_disp"))
        spinlayout.addWidget(self.num_disp)
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
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.start)
        buttonLayoutWidget = QWidget()
        buttonLayoutWidget.setLayout(buttonLayout)
        layout.addWidget(buttonLayoutWidget, 4, 0, 1, 8)
        return layout

    def updateVideo(self, v0, v1, v_disp):
        if self.mode == "calib":
            self.process.show()
            self.takeImage.show()
            self.ignoreExistingImageData.setDisabled(False)
            self.advanced.setDisabled(False)
            self.video0Calib.setPixmap(v0)
            self.video1Calib.setPixmap(v1)
        elif self.mode == "bm":
            self.video0Bm.setPixmap(v0)
            self.video1Bm.setPixmap(v1)
            self.video_disp.setPixmap(v_disp)

    def thread_complete(self):
        print("Worker thread stopped...")

    def updateCalibInfo(self, rms, message):
        self.RMSValue.setText(f"{rms}")
        self.calibInfo.setText(message)

    def _initUIElements(self):
        # wire bm signals/slots
        self.textureThreshold.valueChanged.connect(
            self.worker.updateTextureThreshold)
        self.min_disp.valueChanged.connect(self.worker.updateMin_disp)
        self.num_disp.valueChanged.connect(self.worker.updateNum_disp)
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

        # wire calib signals/slots
        self.calibratorLayoutWidget.takeImageTriggered.connect(
            self.worker.saveImage)
        self.process.clicked.connect(self.worker.calibrateSensor)
        self.takeImage.clicked.connect(self.worker.saveImage)
        self.ignoreExistingImageData.toggled.connect(
            self.worker.setIgnoreExistingImageData)
        self.advanced.toggled.connect(
            self.worker.enableAdvancedCalib)
        self.calib_image_index.valueChanged.connect(
            self.worker.updateCalib_image_index)
        self.rms_limit.valueChanged.connect(
            self.worker.updateRms_limit)
        self.increment.valueChanged.connect(
            self.worker.updateIncrement)
        self.max_rms.valueChanged.connect(
            self.worker.updateMax_rms)
        self.worker.signals.updateCalibInfo.connect(self.updateCalibInfo)
        self.worker.signals.rmsLimitUpdated.connect(self.rms_limit.setValue)
        self.worker.signals.calibImageIndexUpdated.connect(
            self.calib_image_index.setValue)

        # load values
        try:
            self._setLoadedValues(f"{self.SETTINGS_DIR}/lastSaved.npz")
        except IOError:
            print("Settings file at {0} not found"
                  .format(f"{self.SETTINGS_DIR}/lastSaved.npz"))
            self.sigint_handler()

    def _startCalibration(self):
        print("Starting calibration...")
        if self.mode == "calib":
            print("Calibration is already running...")
            return

        self.worker.enableAdvancedCalib(self.advanced.isChecked())
        self.worker.setIgnoreExistingImageData(
            self.ignoreExistingImageData.isChecked())

        # Execute
        self.worker.state = States.Idle
        if self.mode == "none":
            self.worker.moveToThread(self.workerThread)
            self.workerThread.finished.connect(self.worker.deleteLater)
            self.workerThread.started.connect(self.worker.run)
            self.workerThread.start()
        self.mode = "calib"

    def _startBMConfiguration(self):
        print("Starting block matching...")
        if self.mode == "bm":
            print("Block Mathcing is already running...")
            return

        # Execute
        self.worker.state = States.BlockMatching
        if self.mode == "none":
            self.worker.moveToThread(self.workerThread)
            self.workerThread.finished.connect(self.worker.deleteLater)
            self.workerThread.started.connect(self.worker.run)
            self.workerThread.start()
        self.mode = "bm"

    def sigint_handler(self):
        if self.worker is not None:
            self.worker.stop = True
            self.workerThread.quit()
            self.workerThread.wait()
        print("Exiting app through GUI")
        QApplication.quit()
