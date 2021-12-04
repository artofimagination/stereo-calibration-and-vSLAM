from PyQt5 import QtGui, QtCore
from GLWidget import GLWidget
from OpenGL.GL import glPushMatrix, glEnableClientState, GL_VERTEX_ARRAY
from OpenGL.GL import GL_COLOR_ARRAY, glVertexPointer, GL_FLOAT, glColorPointer
from OpenGL.GL import glDrawArrays, GL_POINTS, glDisableClientState, glPopMatrix
from OpenGL.GL import glClearColor, GL_DEPTH_TEST, glEnable, GL_UNSIGNED_BYTE
import OpenGL.arrays.vbo as glvbo

import numpy as np


## @class PointCloudGLWidget
# Visualizes point cloud and trajectory as a series of points.
#
# Much faster than using pyqtgraphs
# The credit is to this repo
# https://github.com/SoonminHwang/py-pointcloud-viewer
class PointCloudGLWidget(GLWidget):

    def __init__(self):
        super(PointCloudGLWidget, self).__init__()

        # rendering related members
        self.disp_count = 0
        self.vbo_disp = None
        self.vbo_disp_clr = None
        self.trajectory_count = 0
        self.vbo_trajectory = None
        self.vbo_trajectory_clr = None

        # Which elements to keep and display in the array
        # For instance 100 will only take every 100th point.
        self.samplingRatio = 1
        # Any point that represents a depth greater this limit will be ignored.
        self.ignoreDepthLimit = 500
        # Field of view
        self.fov = 50

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

    # Sets the field of view.
    def setFov(self, value):
        if self.fov != value:
            self.fov = value

    # Sets the field of view.
    def setIgnoreDepthLimit(self, value):
        if self.ignoreDepthLimit != value:
            self.ignoreDepthLimit = value

    def paintGL(self):
        GLWidget.paintGL(self)
        glPushMatrix()
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        if self.disp_count > 0:
            vbo_disp = self.vbo_disp
            vbo_disp_clr = self.vbo_disp_clr
            disp_count = self.disp_count

            vbo_disp.bind()
            glVertexPointer(3, GL_FLOAT, 0, vbo_disp)
            vbo_disp.unbind()

            vbo_disp_clr.bind()
            glColorPointer(3, GL_UNSIGNED_BYTE, 0, vbo_disp_clr)
            vbo_disp_clr.unbind()

            glDrawArrays(GL_POINTS, 0, disp_count)

        if self.trajectory_count > 0:
            vbo_trajectory = self.vbo_trajectory
            vbo_trajectory_clr = self.vbo_trajectory_clr
            trajectory_count = self.trajectory_count

            vbo_trajectory.bind()
            glVertexPointer(3, GL_FLOAT, 0, vbo_trajectory)
            vbo_trajectory.unbind()

            vbo_trajectory_clr.bind()
            glColorPointer(3, GL_UNSIGNED_BYTE, 0, vbo_trajectory_clr)
            vbo_trajectory_clr.unbind()

            glDrawArrays(GL_POINTS, 0, trajectory_count)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glPopMatrix()

    # Calculates pointcloud from depth map.
    def calculatePointCloud(self, depth):
        if depth is None:
            return
        height = depth.shape[0]
        width = depth.shape[1]
        fy = 0.5 / np.tan(self.fov * 0.5)
        aspectRatio = width / height
        fx = fy / aspectRatio
        mask = np.where((depth > 0.1) & (depth < self.ignoreDepthLimit))
        x = mask[1]
        y = mask[0]

        normalized_x = (x.astype(np.float32) - width * 0.5) / width
        normalized_y = (y.astype(np.float32) - height * 0.5) / height
        # 255 represents the camera window (not a proper solution),
        # should be incorporated with the fov calculations.
        world_x = normalized_x * 255 / fx
        world_y = normalized_y * 255 / fy
        world_z = depth[y, x]
        pointCloud = np.vstack((-world_x, world_y, -world_z)).T
        pointCloud = pointCloud[0::self.samplingRatio, :]
        return pointCloud

    # Sets the map VBO.
    def setMapVBO(self, pointCloud):
        if pointCloud is not None:
            self.disp_count = len(pointCloud)
            indices = np.arange(self.disp_count)
            redIndices = np.full(self.disp_count, 0)
            greenIndices = np.full(self.disp_count, 1)
            depthIndices = np.full(self.disp_count, 2)

            colors = np.zeros((self.disp_count, 3), dtype=np.uint8)
            colors[indices, redIndices] = pointCloud[indices, depthIndices]
            colors[indices, greenIndices] =\
                (255 - pointCloud[indices, depthIndices])
            self.vbo_disp = glvbo.VBO(pointCloud.astype(np.float32) / 10.0)
            self.vbo_disp_clr = glvbo.VBO(colors)

    # Sets the trajectory VBO.
    def setTrajectoryVBO(self, trajectory):
        if trajectory is not None:
            self.trajectory_count = len(trajectory)
            colors = np.full([self.trajectory_count, 3], [255, 255, 255], dtype=np.uint8)
            self.vbo_trajectory = glvbo.VBO(
                (trajectory[:, :, 3][:, ]).astype(np.float32) / 10.0)
            self.vbo_trajectory_clr = glvbo.VBO(colors)

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glEnable(GL_DEPTH_TEST)
        self.reset_view()
