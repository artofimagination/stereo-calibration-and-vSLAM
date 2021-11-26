# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QWidget, QHBoxLayout
from PyQt5 import QtCore
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np


# This class is a simple point cloud visualizer,
# using pyqtgraph scatterplot feature
# Note: that it is not a high performance tool.
# I've got smooth results when drawing about 500-2000 points
# Need some optimisation.
# Initial example:
# https://gist.github.com/markjay4k/da2f55e28514be7160a7c5fbf95bd243
class PointCloudWidget(QWidget):

    def __init__(self):
        super(PointCloudWidget, self).__init__()
        self.vertices = dict()
        layout = QHBoxLayout(self)
        self.w = gl.GLViewWidget()
        layout.addWidget(self.w)
        self.w.opts['distance'] = 300

        self.fov = 60
        self.magnification = 20.0
        self.resolution = (640, 480)
        # Which elements to keep and display in the array
        # For instance 100 will only take every 100th point.
        self.samplingRatio = 500
        # + 1 to handle rounding error
        count = int(
            self.resolution[0] * self.resolution[1] / self.samplingRatio) + 1

        self._recreateScatterPlot(count)

        self.data = list()
        self.quarter = 0

    # Sets the field of view.
    def setFov(self, value):
        if self.fov != value:
            self.fov = value

    # Sets resolution. This changes the scatterplot item counts,
    # so the items have to be regenerated.
    def setResolution(self, value):
        self.resolution = value
        count = int(
                self.resolution[0] *
                self.resolution[1] / self.samplingRatio) + 1
        self._recreateScatterPlot(count)

    # Regenerates the scatterplot item list.
    # The count depends on the resolution and sampling ratio
    def _recreateScatterPlot(self, count):
        self.w.clear()

        # create the background grids
        for i in range(count):
            pts = np.array([0, 0, 0])
            self.vertices[i] = gl.GLScatterPlotItem(pos=pts, color=pg.glColor(
                0, 0, 0), size=2, pxMode=True)
            self.vertices[i].setGLOptions('opaque')
            self.w.addItem(self.vertices[i])

    # Sets the sampling ratio.
    # Changes the scatterplot ittem count, so it has to be regenerated.
    def setSamplingRatio(self, value):
        if self.samplingRatio != value:
            self.samplingRatio = value

            count = int(
                self.resolution[0] *
                self.resolution[1] / self.samplingRatio) + 1
            self._recreateScatterPlot(count)

    # Calculates the 3D point cloud from depth map frame.
    def setMapVBO(self, depth):
        # TODO: FOV doesn't seem to work that well. need improvement.
        height = depth.shape[0]
        width = depth.shape[1]
        fy = 0.5 / np.tan(self.fov * 0.5)
        aspectRatio = width / height
        fx = fy / aspectRatio
        mask = np.where(depth >= 0)
        x = mask[1]
        y = mask[0]

        normalized_x = (x.astype(np.float32) - width * 0.5) / width
        normalized_y = (y.astype(np.float32) - height * 0.5) / height
        # TODO: 255 represents the camera window (not a proper solution),
        # should be incorporated with the fov calculations.
        world_x = normalized_x * 255 / fx
        world_y = normalized_y * 255 / fy
        world_z = depth[y, x]

        pointCloud = np.vstack((world_x, world_y, world_z)).T

        pointCloud = pointCloud
        pointCloud = pointCloud[0::self.samplingRatio, :]
        self.count = len(pointCloud)

        return pointCloud

    # Sets the new point cloud data array to be visualized.
    def setData(self, depthMap):
        if depthMap is None:
            return
        self.data = self.setMapVBO(depthMap)

    # Updates the scatterplot items with new positions and colors.
    def update(self):
        if self.quarter > 16:
            self.quarter = 1
        for i in range(int(self.quarter * len(self.data) / 16)):
            color = pg.glColor(
                self.data[i][2], 255 - self.data[i][2], 0)
            if self.data[i][2] == 0:
                color = pg.glColor(0, 0, 0)
            pts = np.vstack(
                [self.data[i][0],
                 self.data[i][2],
                 self.data[i][1]]).transpose()
            if (self.vertices[i].color != color):
                self.vertices[i].setData(
                    pos=pts,
                    color=color,
                    size=2,
                    pxMode=True)
        self.quarter += 1
