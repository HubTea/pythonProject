from OpenGL.GL import *
from OpenGL.GLU import *

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QMouseEvent, QKeyEvent, QVector3D, QCursor
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtOpenGL import QGLWidget

import numpy as np
#
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super(MainWindow, self).__init__()
#         self.glWidget = GLWidget()
#         self.setCentralWidget(self.glWidget)
#
#
# class GLWidget(QGLWidget):
#     def __init__(self):
#         super(GLWidget, self).__init__(None)
#         self.size = QSize(1000, 1000)
#         self.vertices = np.array([
#             -0.5, 0, 0,
#             0, -0.5, 0,
#             0.5, 0, 0,
#             0.1, 0.5, 0,
#             -0.1, 0.5, 0
#         ], dtype='float32')
#
#         self.elements = np.array([
#             0, 1, 4,
#             2, 1, 3
#         ], dtype='int')
#
#     def initializeGL(self):
#         glClearColor(0, 0, 0, 0)
#         glEnable(GL_DEPTH_TEST)
#         glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
#
#         self.vbo = glGenBuffers(1)
#         glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
#         glBufferData(GL_ARRAY_BUFFER, self.vertices, GL_STATIC_DRAW)
#
#         # self.ebo = glGenBuffers(1)
#         # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
#         # glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.elements, GL_STATIC_DRAW)
#
#     def sizeHint(self):
#         return QSize(1000, 1000)
#
#     def paintGL(self):
#         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#
#         glBegin(GL_TRIANGLES)
#         glVertex3f(-1, -1, 0)
#         glVertex3f(-1, 0, 0)
#         glVertex3f(-0.5, 0, 0)
#         glEnd()
#
#         glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
#         # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
#         glVertexPointer(2, GL_FLOAT, 0, None)
#         glDrawArrays(GL_TRIANGLES, 0, 3)
#         glFlush()
#
#
# def main():
#     app = QApplication([])
#     window = MainWindow()
#     window.show()
#     app.exec_()
#
#
# if __name__ == '__main__':
#     main()

import sys


class SimpleTestWidget(QGLWidget):

    def __init__(self):
        QGLWidget.__init__(self)

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)

        self.vbo1 = glGenBuffers(1)
        self.mesh1 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype='float32')
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo1)
        glBufferData(GL_ARRAY_BUFFER, self.mesh1, GL_STATIC_DRAW)
        # self._vertexBuffer = glGenBuffers(1)
        # glBindBuffer(GL_ARRAY_BUFFER, self._vertexBuffer)
        # vertices = np.array([0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5], dtype='float32')
        # glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW)
        #
        # self._colorBuffer = glGenBuffers(1)
        # glBindBuffer(GL_ARRAY_BUFFER, self._colorBuffer)
        # colors = np.array([1.0, 1.0, 1.0, 0, 0, 1.0, 0, 1, 0, 1.0, 0, 0], dtype='float32')
        # glBufferData(GL_ARRAY_BUFFER, colors, GL_STATIC_DRAW)
        #
        # elements = np.array([0, 1, 2, 3], dtype='int')

    def paintGL(self):
        glViewport(0, 0, self.width(), self.height())
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glEnableClientState(GL_VERTEX_ARRAY)

        glLoadIdentity()
        glBegin(GL_LINES)
        glColor3f(1, 1, 1)
        glVertex3f(-1, 0, 0)
        glVertex3f(1, 0, 0)
        glColor3f(0, 0, 1)
        glVertex3f(0, 1, 0)
        glVertex3f(0, -1, 0)
        glEnd()

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo1)

        glVertexPointer(3, GL_FLOAT, 0, None)
        glColor3f(1, 0, 0)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        glRotate(-200, 1, 0, 0)
        glTranslate(-0.1, -0.5, -0.1)
        glColor3f(0, 1, 0)
        glDrawArrays(GL_TRIANGLES, 0, 3)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = SimpleTestWidget()
    w.show()
    app.exec_()
