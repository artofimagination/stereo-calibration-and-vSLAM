from PyQt5 import QtGui, QtCore
from GLWidget import GLWidget
from OpenGL.GL import glPushMatrix, glEnableClientState, GL_VERTEX_ARRAY
from OpenGL.GL import GL_COLOR_ARRAY, glVertexPointer, GL_FLOAT, glColorPointer
from OpenGL.GL import glDrawArrays, GL_POINTS, glDisableClientState, glPopMatrix
from OpenGL.GL import glClearColor, GL_DEPTH_TEST, glEnable, GL_UNSIGNED_BYTE
import OpenGL.arrays.vbo as glvbo

import numpy as np


BACKGROUND_BLACK = True

DELAY_FOR_25FPS = 0.04
DELAY_FOR_100FPS = 0.01


# Visualizes point cloud. Much faster than using pyqtgraphs
# The credit is for this repo
# https://github.com/SoonminHwang/py-pointcloud-viewer
class PointCloudGLWidget(GLWidget):
    samplingRatioUpdated = QtCore.pyqtSignal(int)
    fovUpdated = QtCore.pyqtSignal(int)

    def __init__(self):
        super(PointCloudGLWidget, self).__init__()

        self.disp_count = 0
        self.vbo_disp = None
        self.vbo_disp_clr = None
        self.pointCloud = None

        self.fov = 50
        # Which elements to keep and display in the array
        # For instance 100 will only take every 100th point.
        self.samplingRatio = 1
        self.setDefaultModelViewMatrix()

    def keyPressEvent(self, event):
        if type(event) == QtGui.QKeyEvent:
            if event.key() == QtCore.Qt.Key_R:
                self.setDefaultModelViewMatrix()

    def setDefaultModelViewMatrix(self):
        self.modelview_matrix_ = np.array(
            [[9.99986172e-01, 4.30997461e-03, 2.50429893e-03, 0.00000000e+00],
             [-4.28775977e-03, 9.99951363e-01, -8.90435092e-03, 0.00000000e+00],
             [-2.54254532e-03, 8.89338460e-03, 9.99955297e-01, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, -3.25000000e+01, 1.00000000e+00]])

    # Sets the sampling ratio.
    def setSamplingRatio(self, value):
        if self.samplingRatio != value:
            self.samplingRatio = value
            self.samplingRatioUpdated.emit(self.samplingRatio)

    # Sets the field of view.
    def setFov(self, value):
        if self.fov != value:
            self.fov = value
            self.fovUpdated.emit(self.fov)

    def paintGL(self):
        GLWidget.paintGL(self)
        if self.disp_count > 0:
            glPushMatrix()
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)

            vbo_disp = self.vbo_disp
            vbo_disp_clr = self.vbo_disp_clr
            disp_count = self.disp_count

            vbo_disp.bind()
            glVertexPointer(3, GL_FLOAT, 0, vbo_disp)
            vbo_disp.unbind()

            vbo_disp_clr.bind()
            glColorPointer(3,  GL_UNSIGNED_BYTE, 0, vbo_disp_clr)
            vbo_disp_clr.unbind()

            glDrawArrays(GL_POINTS, 0, disp_count)

            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)

            glPopMatrix()

    def calculatePointcloud(self, depth):
        height = depth.shape[0]
        width = depth.shape[1]
        fy = 0.5 / np.tan(self.fov * 0.5)
        aspectRatio = width / height
        fx = fy / aspectRatio
        mask = np.where(depth > 0)
        x = mask[1]
        y = mask[0]

        normalized_x = (x.astype(np.float32) - width * 0.5) / width
        normalized_y = (y.astype(np.float32) - height * 0.5) / height
        # 255 represents the camera window (not a proper solution),
        # should be incorporated with the fov calculations.
        world_x = normalized_x * 255 / fx
        world_y = normalized_y * 255 / fy
        world_z = depth[y, x]
        pointCloud = np.vstack((-world_x, world_y, world_z)).T
        self.pointCloud = pointCloud[0::self.samplingRatio, :]

        if self.pointCloud is not None:
            self.disp_count = len(self.pointCloud)
            indices = np.arange(self.disp_count)
            redIndices = np.full(self.disp_count, 0)
            greenIndices = np.full(self.disp_count, 1)
            depthIndices = np.full(self.disp_count, 2)

            colors = np.zeros((self.disp_count, 3), dtype=np.uint8)
            colors[indices, redIndices] = self.pointCloud[indices, depthIndices]
            colors[indices, greenIndices] =\
                (255 - self.pointCloud[indices, depthIndices])
            self.vbo_disp = glvbo.VBO(self.pointCloud / 10.0)
            self.vbo_disp_clr = glvbo.VBO(colors)

    def initializeGL(self):
        if BACKGROUND_BLACK:
            glClearColor(0.0, 0.0, 0.0, 0.0)
        else:
            glClearColor(1.0, 1.0, 1.0, 1.0)

        glEnable(GL_DEPTH_TEST)
        self.reset_view()
