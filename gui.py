import os
import numpy as np
import glob
import shutil
from pathlib import Path

from backend import Backend, States, Modes
from pointCloudGLWidget import PointCloudGLWidget
from linePlotWidget import LinePlotWidget

from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QMainWindow, QGridLayout, QAction
from PyQt5.QtWidgets import QSpinBox, QLabel, QHBoxLayout, QFileDialog
from PyQt5.QtWidgets import QWidget, QApplication, QCheckBox
from PyQt5.QtWidgets import QPushButton, QTabWidget, QVBoxLayout
from PyQt5.QtWidgets import QDoubleSpinBox, QComboBox, QGroupBox


# Deletes all files in the content path.
def _clearContent(content):
    files = glob.glob(content)
    for f in files:
        if not os.path.isdir(f):
            os.remove(f)

    files = glob.glob(content)
    for f in files:
        if not os.path.isdir(f):
            os.remove(f)


# Calibration widget.
# the only reason it is created to catch the press 'n' event.
class CalibWidget(QWidget):
    takeImageTriggered = QtCore.pyqtSignal()

    def keyPressEvent(self, event):
        super(CalibWidget, self).keyPressEvent(event)
        if event.key() == QtCore.Qt.Key_N:
            self.takeImageTriggered.emit()


# Main Qt UI window
class MainWindow(QMainWindow):
    # Folder that contains the saved UI settings.
    SETTINGS_DIR = "settings"

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # Initialize worker thread related members.
        self.workerThread = QThread(self)
        self.worker = Backend()
        self.worker.signals.framesSent.connect(self.updateVideo)
        self.worker.signals.finished.connect(self.thread_complete)
        self.worker.signals.error.connect(self.sigint_handler)

        # init UI.
        self.setMinimumSize(200, 100)
        mainLayout = QGridLayout()
        self._createMenu()

        # Create tabs
        tabwidget = QTabWidget()
        calibratorLayout = self._createCalibrationUI()
        self.calibratorLayoutWidget = CalibWidget()
        self.calibratorLayoutWidget.setLayout(calibratorLayout)
        tabwidget.addTab(self.calibratorLayoutWidget, "Sensor calibration")

        bmConfiguratorLayout = self._createBlockMatchingConfiguratorUI()
        bmConfiguratorLayoutWidget = QWidget()
        bmConfiguratorLayoutWidget.setLayout(bmConfiguratorLayout)
        tabwidget.addTab(
            bmConfiguratorLayoutWidget, "Block Matching Configurator")

        featureDetectorLayout = self._createFeatureDetectionUI()
        featureDetectionLayoutWidget = QWidget()
        featureDetectionLayoutWidget.setLayout(featureDetectorLayout)
        tabwidget.addTab(featureDetectionLayoutWidget, "Feature detection")

        motionEstimationLayout = self._createMotionEstimationUI()
        motionEstimationLayoutWidget = QWidget()
        motionEstimationLayoutWidget.setLayout(motionEstimationLayout)
        tabwidget.addTab(motionEstimationLayoutWidget, "Motion estimation")

        self._initUIElements()

        mainLayout.addWidget(tabwidget, 0, 0)
        mainWidget = QWidget()
        mainWidget.setLayout(mainLayout)
        self.setCentralWidget(mainWidget)

        desktop = QApplication.desktop()
        screenRect = desktop.screenGeometry()
        self.resize(screenRect.width(), screenRect.height())

        if not os.path.isdir(self.SETTINGS_DIR):
            os.mkdir(self.SETTINGS_DIR)

        self.show()

    # Saves all UI values into an npz file.
    # Saves all calibration images and chessboard.npz for each sensor
    # in the folder named identical to the settings npz file
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
            bm_mode=self.blockMatching.currentIndex(),
            bm_drawEpipolar=self.drawEpipolar.isChecked(),
            bm_resolution=self.resolutionBm.currentIndex(),
            bm_leftCameraIndex=self.bmCameraIndexLeft.currentIndex(),
            bm_rightCameraIndex=self.bmCameraIndexLeft.currentIndex(),
            pc_fov=self.fov.value(),
            pc_samplingRatio=self.samplingRatio.value(),
            pc_ignoreRendererMaxDepth=self.rendererMaxDepth.value(),
            cal_calib_image_index=self.calib_image_index.value(),
            cal_rms_limit=self.rms_limit.value(),
            cal_advanced=self.advanced.isChecked(),
            cal_ignoreExitingImageData=self
                                        .ignoreExistingImageData
                                        .isChecked(),
            cal_rms_increment=self.increment.value(),
            cal_max_rms=self.max_rms.value(),
            cal_resolution=self.resolutionCal.currentIndex(),
            cal_leftCameraIndex=self.calibCameraIndexLeft.currentIndex(),
            cal_rightCameraIndex=self.calibCameraIndexRight.currentIndex(),
            feat_featureDetector=self.featureDetector.currentIndex(),
            feat_featureMatcher=self.featureMatcher.currentIndex(),
            feat_maxDistance=self.maxDistance.value(),
            motion_inliers=self.inliers.value(),
            motion_maxDepth=self.maxDepth.value(),
            motion_reprojectionError=self.reprojectionError.value())

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

    # Saves UI settings and calibration images/data
    # Also creates a lastSaved folder
    # for quick loading last saved info when the application starts.
    def saveSettings(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(
            self,
            "QFileDialog.getSaveFileName()", "", "npz (*.npz)",
            options=options)
        if fileName:
            self._saveValues(fileName)

            _clearContent("calibImages/left/lastSaved/*")
            _clearContent("calibImages/left/lastSaved/*")
            self._saveValues(f"{self.SETTINGS_DIR}/lastSaved.npz")

    # Loads and sets settings values from the npz file.
    # Also loads the appropriate calib images and data.
    def _setLoadedValues(self, settingsName):
        _clearContent("calibImages/left/*")
        _clearContent("calibImages/right/*")

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
        self.blockMatching.setCurrentIndex(settings["bm_mode"])
        self.drawEpipolar.setChecked(bool(settings["bm_drawEpipolar"]))
        self.smallerBlockSize.setValue(settings["bm_smallerBlockSize"])
        self.resolutionBm.setCurrentIndex(settings["bm_resolution"])
        self.bmCameraIndexLeft.setCurrentIndex(settings["bm_leftCameraIndex"])
        self.bmCameraIndexRight.setCurrentIndex(settings["bm_rightCameraIndex"])
        self.fov.setValue(settings["pc_fov"])
        self.samplingRatio.setValue(settings["pc_samplingRatio"])
        self.rendererMaxDepth.setValue(settings["pc_ignoreRendererMaxDepth"])
        self.calib_image_index.setValue(settings["cal_calib_image_index"])
        self.rms_limit.setValue(settings["cal_rms_limit"])
        self.advanced.setChecked(bool(settings["cal_advanced"]))
        self.ignoreExistingImageData.setChecked(
            bool(settings["cal_ignoreExitingImageData"]))
        self.increment.setValue(settings["cal_rms_increment"])
        self.max_rms.setValue(settings["cal_max_rms"])
        self.resolutionCal.setCurrentIndex(settings["cal_resolution"])
        self.calibCameraIndexLeft.setCurrentIndex(settings["cal_leftCameraIndex"])
        self.calibCameraIndexRight.setCurrentIndex(settings["cal_rightCameraIndex"])
        self.featureDetector.setCurrentIndex(settings["feat_featureDetector"])
        self.featureMatcher.setCurrentIndex(settings["feat_featureMatcher"])
        self.maxDistance.setValue(settings["feat_maxDistance"])
        self.inliers.setValue(settings["motion_inliers"])
        self.maxDepth.setValue(settings["motion_maxDepth"])
        self.reprojectionError.setValue(settings["motion_reprojectionError"])

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

    # Loads settings,calib images and data.
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

    # Creates the menu items.
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

    # Creates the calibration tab UI.
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

        self.resolutionCal = QComboBox()
        self.resolutionCal.addItems(["480p", "720p"])
        ignoreExistingImageDataLabel = QLabel("Ignore existing image data")
        self.ignoreExistingImageData = QCheckBox()
        self.ignoreExistingImageData.setDisabled(True)
        advancedLabel = QLabel("Advanced use mode")
        self.advanced = QCheckBox()
        self.advanced.setDisabled(True)
        optionsLayout = QHBoxLayout()
        optionsLayout.addWidget(QLabel("Resolution"))
        optionsLayout.addWidget(self.resolutionCal)
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

        cameraLayout = QGridLayout()
        displayGroupBox = QGroupBox("Visualisation")
        displayGroupBox.setLayout(cameraLayout)
        self.video0Calib = QLabel()
        self.calibSwapCameras = QPushButton("Swap lenses")
        self.calibCameraIndexLeft = QComboBox()
        cameraLayout.addWidget(self.video0Calib, 0, 0, 4, 3)
        cameraLayout.addWidget(self.calibSwapCameras, 5, 0, 1, 1)
        cameraLayout.addWidget(QLabel("Left camera indices"), 5, 1, 1, 1)
        cameraLayout.addWidget(self.calibCameraIndexLeft, 5, 2, 1, 1)
        self.video1Calib = QLabel()
        self.calibCameraIndexRight = QComboBox()
        cameraLayout.addWidget(self.video1Calib, 0, 3, 4, 3)
        cameraLayout.addWidget(QLabel("Right camera indices"), 5, 3, 1, 1)
        cameraLayout.addWidget(self.calibCameraIndexRight, 5, 4, 1, 1)
        layout.addWidget(displayGroupBox, 0, 0, 1, 1)

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

        layout.addWidget(labelsLayoutWidget, 1, 0, 1, 2)
        layout.addWidget(optionsLayoutWidget, 2, 0, 1, 4)
        layout.addWidget(configLayoutWidget, 3, 0, 1, 4)
        layout.addWidget(buttonLayoutWidget, 4, 0, 1, 4)
        return layout

    # Creates the block matching UI.
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

        self.textureThresholdLabel = QLabel("textureThreshold")
        self.textureThreshold = QSpinBox()
        self.min_disp = QSpinBox()
        self.num_disp = QSpinBox()
        self.blockSize = QSpinBox()
        self.uniquenessRatio = QSpinBox()
        self.speckleWindowSize = QSpinBox()
        self.speckleRange = QSpinBox()
        self.disp12MaxDiff = QSpinBox()
        self.preFilterSizeLabel = QLabel("preFilterSize")
        self.preFilterSize = QSpinBox()
        self.preFilterTypeLabel = QLabel("preFilterType")
        self.preFilterType = QSpinBox()
        self.preFilterCapLabel = QLabel("preFilterCap")
        self.preFilterCap = QSpinBox()
        self.smallerBlockSizeLabel = QLabel("smallerBlockSize")
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

        self.resolutionBm = QComboBox()
        self.resolutionBm.addItems(["480p", "720p"])
        self.blockMatching = QComboBox()
        self.blockMatching.addItems(
            ["Block Matching", "Semi-Global Block Matching"])
        self.drawEpipolar = QCheckBox()

        self.video0Bm = QLabel()
        self.bmCameraIndexLeft = QComboBox()
        self.bmSwapCameras = QPushButton("Swap lenses")
        self.video1Bm = QLabel()
        self.bmCameraIndexRight = QComboBox()
        self.video_disp = QLabel()
        self.pointCloud = PointCloudGLWidget()
        self.fov = QSpinBox()
        self.fov.setRange(1, 360)
        self.fov.valueChanged.connect(
            self.pointCloud.setFov)
        self.rendererMaxDepth = QSpinBox()
        self.rendererMaxDepth.setRange(1, 5000)
        self.rendererMaxDepth.valueChanged.connect(
            self.pointCloud.setIgnoreDepthLimit)
        self.samplingRatio = QSpinBox()
        self.samplingRatio.setRange(1, 10000)
        self.samplingRatio.setSingleStep(50)
        self.samplingRatio.valueChanged.connect(
            self.pointCloud.setSamplingRatio)

        cameraLayout = QGridLayout()
        displayGroupBox = QGroupBox("Visualisation")
        displayGroupBox.setLayout(cameraLayout)
        pointCloudLayout = QGridLayout()
        pointCloudLayout.addWidget(self.pointCloud, 0, 0, 4, 4)
        pointCloudLayout.addWidget(QLabel("Field of view"), 0, 4, 1, 1)
        pointCloudLayout.addWidget(self.fov, 0, 5, 1, 1)
        pointCloudLayout.addWidget(QLabel("Sampling ratio"), 1, 4, 1, 1)
        pointCloudLayout.addWidget(self.samplingRatio, 1, 5, 1, 1)
        pointCloudLayout.addWidget(QLabel("Ignore depth"), 2, 4, 1, 1)
        pointCloudLayout.addWidget(self.rendererMaxDepth, 2, 5, 1, 1)
        pointCloudControl = QGroupBox("Point cloud")
        pointCloudControl.setLayout(pointCloudLayout)
        cameraLayout.addWidget(self.video0Bm, 0, 0, 4, 4)
        cameraLayout.addWidget(self.bmSwapCameras, 5, 0, 1, 1)
        cameraLayout.addWidget(QLabel("Left camera indices"), 5, 1, 1, 1)
        cameraLayout.addWidget(self.bmCameraIndexLeft, 5, 2, 1, 1)
        cameraLayout.addWidget(self.video1Bm, 0, 3, 4, 4)
        cameraLayout.addWidget(QLabel("Right camera indices"), 5, 3, 1, 1)
        cameraLayout.addWidget(self.bmCameraIndexRight, 5, 4, 1, 1)
        cameraLayout.addWidget(self.video_disp, 6, 0, 4, 4)
        cameraLayout.addWidget(pointCloudControl, 6, 3, 4, 4)
        layout.addWidget(displayGroupBox, 0, 0, 1, 6)

        bmControlLayout = QGridLayout()
        bmControlLayout.addWidget(QLabel("Block matching type"), 0, 0, 1, 1)
        bmControlLayout.addWidget(self.blockMatching, 0, 1, 1, 1)
        bmControlLayout.addWidget(QLabel("Draw epipolar lines"), 0, 2, 1, 1)
        bmControlLayout.addWidget(self.drawEpipolar, 0, 3, 1, 1)
        bmControlLayout.addWidget(QLabel("Resolution"), 0, 4, 1, 1)
        bmControlLayout.addWidget(self.resolutionBm, 0, 5, 1, 1)
        bmControlLayout.addWidget(self.textureThresholdLabel, 1, 0, 1, 1)
        bmControlLayout.addWidget(self.textureThreshold, 1, 1, 1, 1)
        bmControlLayout.addWidget(QLabel("min_disp"), 1, 2, 1, 1)
        bmControlLayout.addWidget(self.min_disp, 1, 3, 1, 1)
        bmControlLayout.addWidget(QLabel("num_disp"), 1, 4, 1, 1)
        bmControlLayout.addWidget(self.num_disp, 1, 5, 1, 1)
        bmControlLayout.addWidget(QLabel("blockSize"), 1, 6, 1, 1)
        bmControlLayout.addWidget(self.blockSize, 1, 7, 1, 1)
        bmControlLayout.addWidget(QLabel("uniquenessRatio"), 1, 8, 1, 1)
        bmControlLayout.addWidget(self.uniquenessRatio, 1, 9, 1, 1)
        bmControlLayout.addWidget(QLabel("speckleWindowSize"), 1, 10, 1, 1)
        bmControlLayout.addWidget(self.speckleWindowSize, 1, 11, 1, 1)
        bmControlLayout.addWidget(QLabel("speckleRange"), 1, 12, 1, 1)
        bmControlLayout.addWidget(self.speckleRange, 1, 13, 1, 1)
        bmControlLayout.addWidget(QLabel("disp12MaxDiff"), 1, 14, 1, 1)
        bmControlLayout.addWidget(self.disp12MaxDiff, 1, 15, 1, 1)
        bmControlLayout.addWidget(self.smallerBlockSizeLabel, 2, 0, 1, 1)
        bmControlLayout.addWidget(self.smallerBlockSize, 2, 1, 1, 1)
        bmControlLayout.addWidget(self.preFilterTypeLabel, 2, 2, 1, 1)
        bmControlLayout.addWidget(self.preFilterType, 2, 3, 1, 1)
        bmControlLayout.addWidget(self.preFilterCapLabel, 2, 4, 1, 1)
        bmControlLayout.addWidget(self.preFilterCap, 2, 5, 1, 1)
        bmControlLayout.addWidget(self.preFilterSizeLabel, 2, 6, 1, 1)
        bmControlLayout.addWidget(self.preFilterSize, 2, 7, 1, 1)
        bmControlGroup = QGroupBox("Block Matching control")
        bmControlGroup.setLayout(bmControlLayout)
        layout.addWidget(bmControlGroup, 2, 0, 1, 8)
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.start)
        buttonLayoutWidget = QWidget()
        buttonLayoutWidget.setLayout(buttonLayout)
        layout.addWidget(buttonLayoutWidget, 4, 0, 1, 8)
        return layout

    ## @brief Creates the feature detection UI.
    def _createFeatureDetectionUI(self):
        layout = QGridLayout()
        self.videoFD = QLabel()

        messageTitle = QLabel("Logging:")
        messageLabel = QLabel()
        self.worker.signals.updateFeatureInfo.connect(messageLabel.setText)
        messageLayout = QVBoxLayout()
        messageLayout.addWidget(messageTitle)
        messageLayout.addWidget(messageLabel)
        messageLayoutWidget = QWidget()
        messageLayoutWidget.setLayout(messageLayout)

        featureDetectorLabel = QLabel("Feature detector")
        self.featureDetector = QComboBox()
        self.featureDetector.addItems(["sift", "orb", "surf"])
        self.featureDetector.currentTextChanged.connect(self.worker.updateFeatureDetector)
        featureMatcherLabel = QLabel("Feature matcher")
        self.featureMatcher = QComboBox()
        self.featureMatcher.addItems(["BF", "FLANN"])
        self.featureMatcher.currentTextChanged.connect(self.worker.updateFeatureMatcher)
        maxDistanceLabel = QLabel("Max allowed distance between best matches")
        self.maxDistance = QDoubleSpinBox()
        self.maxDistance.valueChanged.connect(self.worker.updateMatchDistanceThreshold)
        self.maxDistance.setRange(0.01, 100)
        self.maxDistance.setSingleStep(0.01)
        controlLayout = QGridLayout()
        controlLayout.addWidget(featureDetectorLabel, 0, 2, 1, 1)
        controlLayout.addWidget(self.featureDetector, 0, 3, 1, 1)
        controlLayout.addWidget(featureMatcherLabel, 0, 4, 1, 1)
        controlLayout.addWidget(self.featureMatcher, 0, 5, 1, 1)
        controlLayout.addWidget(maxDistanceLabel, 0, 6, 1, 1)
        controlLayout.addWidget(self.maxDistance, 0, 7, 1, 1)
        controlLayoutWidget = QWidget()
        controlLayoutWidget.setLayout(controlLayout)

        start = QPushButton("Start")
        start.clicked.connect(self._startFeatureDetection)
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(start)
        buttonLayoutWidget = QWidget()
        buttonLayoutWidget.setLayout(buttonLayout)

        layout.addWidget(self.videoFD, 0, 0, 3, 1)
        layout.addWidget(messageLayoutWidget, 3, 0, 1, 1)
        layout.addWidget(controlLayoutWidget, 4, 0, 1, 1)
        layout.addWidget(buttonLayoutWidget, 5, 0, 1, 1)
        return layout

    ## @brief Creates the motion estimation UI.
    def _createMotionEstimationUI(self):
        layout = QGridLayout()
        self.videoDepthME = QLabel()
        self.motionDisplay = PointCloudGLWidget()
        self.trajectoryPlotDepth = LinePlotWidget()
        self.trajectoryPlotDepth.setAxisLabel("time", "z")
        self.trajectoryPlotXY = LinePlotWidget()
        self.trajectoryPlotXY.setAxisLabel("x", "y")

        start = QPushButton("Start")
        start.clicked.connect(self._startMotionEstimation)
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(start)
        buttonLayoutWidget = QWidget()
        buttonLayoutWidget.setLayout(buttonLayout)

        inliersLabel = QLabel("inliers")
        self.inliers = QSpinBox()
        self.inliers.valueChanged.connect(self.worker.updateInlierLimit)
        self.inliers.setRange(1, 200)
        self.inliers.setSingleStep(1)
        maxDepthLabel = QLabel("maxDepth")
        self.maxDepth = QSpinBox()
        self.maxDepth.valueChanged.connect(self.worker.updateMaxDepth)
        self.maxDepth.setRange(1, 200)
        self.maxDepth.setSingleStep(1)
        reprojectionErrorLabel = QLabel("reprojectionError")
        self.reprojectionError = QDoubleSpinBox()
        self.reprojectionError.valueChanged.connect(self.worker.updateReprojectionError)
        self.reprojectionError.setRange(0.01, 10)
        self.reprojectionError.setSingleStep(0.01)
        controlLayout = QGridLayout()
        controlLayout.addWidget(inliersLabel, 0, 0, 1, 1)
        controlLayout.addWidget(self.inliers, 0, 1, 1, 1)
        controlLayout.addWidget(maxDepthLabel, 0, 2, 1, 1)
        controlLayout.addWidget(self.maxDepth, 0, 3, 1, 1)
        controlLayout.addWidget(reprojectionErrorLabel, 0, 4, 1, 1)
        controlLayout.addWidget(self.reprojectionError, 0, 5, 1, 1)
        controlLayoutWidget = QWidget()
        controlLayoutWidget.setLayout(controlLayout)

        layout.addWidget(self.motionDisplay, 0, 0, 3, 4)
        layout.addWidget(self.videoDepthME, 3, 0, 1, 2)
        layout.addWidget(self.trajectoryPlotDepth, 3, 2, 2, 1)
        layout.addWidget(self.trajectoryPlotXY, 3, 3, 2, 1)
        layout.addWidget(controlLayoutWidget, 5, 0, 1, 4)
        layout.addWidget(buttonLayoutWidget, 6, 0, 1, 4)
        return layout

    # Updates the video stream labels.
    def updateVideo(self, v0, v1, v_depth_color, v_depth_gray, v_feature, trajectory):
        if self.worker.mode == Modes.Calibration:
            self.process.show()
            self.takeImage.show()
            self.ignoreExistingImageData.setDisabled(False)
            self.advanced.setDisabled(False)
            self.video0Calib.setPixmap(v0)
            self.video1Calib.setPixmap(v1)
        elif self.worker.mode == Modes.BlockMatching:
            self.video0Bm.setPixmap(v0)
            self.video1Bm.setPixmap(v1)
            self.video_disp.setPixmap(v_depth_color)
            self.pointCloud.setMapVBO(v_depth_gray)
            self.pointCloud.updateGL()
        elif self.worker.mode == Modes.FeatureDetection:
            self.videoFD.setPixmap(v_feature)
        elif self.worker.mode == Modes.MotionEstimation:
            self.videoDepthME.setPixmap(v_depth_color)
            self.motionDisplay.setTrajectoryVBO(trajectory)
            self.motionDisplay.updateGL()
            if trajectory is not None:
                self.trajectoryPlotDepth.plotData(
                    np.arange(len(trajectory)), trajectory[:, 2, 3])
            self.trajectoryPlotXY.plotData(trajectory[:, 0, 3], trajectory[:, 1, 3])

    # Shows/hides the appropriate controls for SGBM and BM.
    def updateBmType(self, index):
        if index == 1:
            self.textureThresholdLabel.hide()
            self.textureThreshold.hide()
            self.preFilterSizeLabel.hide()
            self.preFilterSize.hide()
            self.preFilterCapLabel.hide()
            self.preFilterCap.hide()
            self.preFilterTypeLabel.hide()
            self.preFilterType.hide()
            self.smallerBlockSizeLabel.hide()
            self.smallerBlockSize.hide()
        else:
            self.textureThresholdLabel.show()
            self.textureThreshold.show()
            self.preFilterSizeLabel.show()
            self.preFilterSize.show()
            self.preFilterCapLabel.show()
            self.preFilterCap.show()
            self.preFilterTypeLabel.show()
            self.preFilterType.show()
            self.smallerBlockSizeLabel.show()
            self.smallerBlockSize.show()
        if self.worker is not None:
            self.worker.updateBmType(index)

    def thread_complete(self):
        print("Worker thread stopped...")

    # Updates the calib UI control from the worker thread.
    def updateCalibInfo(self, rms, message):
        self.RMSValue.setText(f"{rms}")
        self.calibInfo.setText(message)

    # Updates the camera indices in all tabs.
    def updateCameraIndices(self, leftIndex, rightIndex, indicesList):
        if self.bmCameraIndexLeft.count() == 0:
            for index in indicesList:
                self.bmCameraIndexLeft.addItem(f"{index}")
                self.bmCameraIndexRight.addItem(f"{index}")
                self.calibCameraIndexLeft.addItem(f"{index}")
                self.calibCameraIndexRight.addItem(f"{index}")
        self.bmCameraIndexLeft.setCurrentIndex(leftIndex)
        self.bmCameraIndexRight.setCurrentIndex(rightIndex)
        self.calibCameraIndexLeft.setCurrentIndex(leftIndex)
        self.calibCameraIndexRight.setCurrentIndex(rightIndex)

    # Sets the ranges, limits and signal connections for each UI element.
    def _initUIElements(self):
        info = self.worker.getSensorIndices()
        self.updateCameraIndices(*info)

        # wire bm signals/slots
        self.blockMatching.currentIndexChanged.connect(
            self.updateBmType)
        self.resolutionBm.currentIndexChanged.connect(
            self.worker.updateResolution)
        self.drawEpipolar.toggled.connect(
            self.worker.updateDrawEpipolar)
        self.textureThreshold.valueChanged.connect(
            self.worker.updateTextureThreshold)
        self.min_disp.valueChanged.connect(self.worker.updateMin_disp)
        self.num_disp.valueChanged.connect(self.worker.updateNum_disp)
        self.blockSize.valueChanged.connect(self.worker.updateBlockSize)
        self.uniquenessRatio.valueChanged.connect(
            self.worker.updateUniquenessRatio)
        self.speckleWindowSize.valueChanged.connect(
            self.worker.updateSpeckleWindowSize)
        self.speckleRange.valueChanged.connect(self.worker.updateSpeckleRange)
        self.disp12MaxDiff.valueChanged.connect(self.worker.updateDisp12MaxDiff)
        self.smallerBlockSize.valueChanged.connect(self.worker.updateSmallerBlockSize)
        self.preFilterType.valueChanged.connect(self.worker.updatePreFilterType)
        self.preFilterSize.valueChanged.connect(self.worker.updatePrefilterSize)
        self.preFilterCap.valueChanged.connect(self.worker.updatePrefilterCap)
        self.worker.signals.cameraIndicesUpdated.connect(
            self.updateCameraIndices)
        self.bmCameraIndexLeft.currentIndexChanged.connect(
            self.worker.updateLeftCameraIndex)
        self.bmCameraIndexRight.currentIndexChanged.connect(
            self.worker.updateRightCameraIndex)
        self.bmSwapCameras.clicked.connect(self.worker.swapSensorIndices)

        # wire calib signals/slots
        self.calibratorLayoutWidget.takeImageTriggered.connect(
            self.worker.saveImage)
        self.process.clicked.connect(self.worker.calibrateSensor)
        self.takeImage.clicked.connect(self.worker.saveImage)
        self.resolutionCal.currentIndexChanged.connect(
            self.worker.updateResolution)
        self.ignoreExistingImageData.toggled.connect(
            self.worker.setIgnoreExistingImageData)
        self.advanced.toggled.connect(self.worker.enableAdvancedCalib)
        self.calib_image_index.valueChanged.connect(
            self.worker.updateCalib_image_index)
        self.rms_limit.valueChanged.connect(self.worker.updateRms_limit)
        self.increment.valueChanged.connect(self.worker.updateIncrement)
        self.max_rms.valueChanged.connect(self.worker.updateMax_rms)
        self.worker.signals.updateCalibInfo.connect(self.updateCalibInfo)
        self.worker.signals.rmsLimitUpdated.connect(self.rms_limit.setValue)
        self.worker.signals.calibImageIndexUpdated.connect(
            self.calib_image_index.setValue)
        self.calibCameraIndexLeft.currentIndexChanged.connect(
            self.worker.updateLeftCameraIndex)
        self.calibCameraIndexRight.currentIndexChanged.connect(
            self.worker.updateRightCameraIndex)
        self.calibSwapCameras.clicked.connect(self.worker.swapSensorIndices)

        # load values
        try:
            self._setLoadedValues(f"{self.SETTINGS_DIR}/lastSaved.npz")
        except IOError:
            print("Settings file at {0} not found"
                  .format(f"{self.SETTINGS_DIR}/lastSaved.npz"))
            self.sigint_handler()

        self.updateBmType(self.blockMatching.currentIndex())

    # Start the thread with calibration.
    # if the thread is already running, it will not restart it
    # just set to calibration mode, if it wasn't yet.
    def _startCalibration(self):
        print("Starting calibration...")
        if self.worker.mode == Modes.Calibration:
            print("Calibration is already running...")
            return

        self.worker.enableAdvancedCalib(self.advanced.isChecked())
        self.worker.setIgnoreExistingImageData(
            self.ignoreExistingImageData.isChecked())

        # Execute
        if self.worker.mode == Modes.NoMode:
            self.worker.moveToThread(self.workerThread)
            self.workerThread.finished.connect(self.worker.deleteLater)
            self.workerThread.started.connect(self.worker.run)
            self.workerThread.start()
        self.worker.state = States.Idle
        self.worker.mode = Modes.Calibration

    # @brief  Start the thread with block matching.
    #
    # If the thread is already running, it will not restart it
    # just set to block matching mode, if it wasn't yet.
    def _startBMConfiguration(self):
        print("Starting block matching...")
        if self.worker.mode == Modes.BlockMatching:
            print("Block Matching is already running...")
            return

        # Execute
        if self.worker.mode == Modes.NoMode:
            self.worker.moveToThread(self.workerThread)
            self.workerThread.finished.connect(self.worker.deleteLater)
            self.workerThread.started.connect(self.worker.run)
            self.workerThread.start()
        self.worker.state = States.Idle
        self.worker.mode = Modes.BlockMatching

    ## @brieft Start the thread with feature detection.
    #
    # If the thread is already running, it will not restart it
    # just set to block matching mode, if it wasn't yet.
    def _startFeatureDetection(self):
        print("Starting feature detection...")
        if self.worker.mode == Modes.FeatureDetection:
            print("Feature detection is already running...")
            return

        # Execute
        if self.worker.mode == Modes.NoMode:
            self.worker.moveToThread(self.workerThread)
            self.workerThread.finished.connect(self.worker.deleteLater)
            self.workerThread.started.connect(self.worker.run)
            self.workerThread.start()
        self.worker.state = States.Idle
        self.worker.mode = Modes.FeatureDetection

    ## @brieft Start the thread with motion estimation.
    #
    # If the thread is already running, it will not restart it
    # just set to block matching mode, if it wasn't yet.
    def _startMotionEstimation(self):
        print("Starting motion estimation...")
        if self.worker.mode == Modes.MotionEstimation:
            print("Motion estimation is already running...")
            return

        # Execute
        if self.worker.mode == Modes.NoMode:
            self.worker.moveToThread(self.workerThread)
            self.workerThread.finished.connect(self.worker.deleteLater)
            self.workerThread.started.connect(self.worker.run)
            self.workerThread.start()
        self.worker.state = States.Idle
        self.worker.mode = Modes.MotionEstimation

    # Terminate UI and the threads appropriately.
    def sigint_handler(self):
        if self.worker is not None:
            self.worker.stop = True
            self.workerThread.quit()
            self.workerThread.wait()
        self.trajectoryPlotXY.clear()
        self.trajectoryPlotDepth.clear()
        print("Exiting app through GUI")
        QApplication.quit()
