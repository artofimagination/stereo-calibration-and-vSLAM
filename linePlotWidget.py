# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QWidget, QHBoxLayout
import pyqtgraph as pg


## @class LinePlotWidget
#  @brief Draws a simple line graph.
#
# Extremely basic solution. No multilines or legends, axis names.
# PlotWidget support pan/zoom by default, though.
class LinePlotWidget(QWidget):

    def __init__(self):
        super(LinePlotWidget, self).__init__()
        self.graphWidget = pg.PlotWidget()
        self.vertices = dict()
        layout = QHBoxLayout(self)
        layout.addWidget(self.graphWidget)

    # Sets the new point cloud data array to be visualized.
    def plotData(self, x, y):
        self.graphWidget.plot(x, y)

    # Clears the widget.
    def clear(self):
        self.graphWidget.clear()
