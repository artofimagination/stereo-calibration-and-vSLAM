import sys
import numpy as np

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow, QHBoxLayout, QWidget
# Insert the to pointCloudWidget
sys.path.insert(0, "..")
from pointCloudGLWidget import PointCloudGLWidget


if __name__ == "__main__":
    app = QApplication([])
    window = QMainWindow()
    mainLayout = QHBoxLayout()

    # Create PointCloudWidget()
    pointCloudWidget = PointCloudGLWidget()
    pointCloudWidget.fov = 50
    # Set the viewpoint distance farther so the point cloud is visible.
    mainLayout.addWidget(pointCloudWidget)

    # Loading example depth map
    depthMapFile = np.load("testDepthMap.npz", allow_pickle=False)
    depthMap = depthMapFile["depthMap"]
    pointCloudWidget.calculatePointcloud(depthMap)

    mainWidget = QWidget()
    mainWidget.setLayout(mainLayout)

    # Resize main window to display size.
    desktop = QApplication.desktop()
    screenRect = desktop.screenGeometry()
    window.resize(screenRect.width(), screenRect.height())
    window.setCentralWidget(mainWidget)
    window.show()

    # Periodic trigger to update the widget.
    timer = QTimer()
    timer.timeout.connect(pointCloudWidget.updateGL)
    timer.start(20)

    sys.exit(app.exec_())
