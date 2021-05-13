# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from OpenGL.GL import *
from OpenGL.GLU import *

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QMouseEvent, QKeyEvent, QVector3D
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout
from PyQt5.QtOpenGL import QGLWidget

import numpy as np
import mini3d
from enum import Enum
import time


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.glWidget = GLWidget()
        self.setCentralWidget(self.glWidget)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        k = e.key()
        if k == Qt.Key_Left:
            self.glWidget.cam.move(0, 0, 0.1)
        elif k == Qt.Key_Right:
            self.glWidget.cam.move(0, 0, -0.1)
        elif k == Qt.Key_Up:
            self.glWidget.cam.move(0, 0.1, 0)
        elif k == Qt.Key_Down:
            self.glWidget.cam.move(0, -0.1, 0)
        elif k == Qt.Key_Q:
            self.glWidget.cam.rotate_direction_axis(1)
        elif k == Qt.Key_E:
            self.glWidget.cam.rotate_direction_axis(-1)
        elif k == Qt.Key_W:
            self.glWidget.cam.rotate_left_axis(1)
        elif k == Qt.Key_S:
            self.glWidget.cam.rotate_left_axis(-1)
        elif k == Qt.Key_A:
            self.glWidget.cam.rotate_up_axis(-1)
        elif k == Qt.Key_D:
            self.glWidget.cam.rotate_up_axis(1)
        elif k == Qt.Key_R:
            self.glWidget.cam.zoom()
        elif k == Qt.Key_F:
            self.glWidget.cam.zoom(False)
        elif k == Qt.Key_T:
            self.glWidget.cam.rev_left_axis(1)
        elif k == Qt.Key_G:
            self.glWidget.cam.rev_left_axis(-1)
        elif k == Qt.Key_Y:
            self.glWidget.cam.rev_up_axis(1)
        elif k == Qt.Key_H:
            self.glWidget.cam.rev_up_axis(-1)
        elif k == Qt.Key_1:
            self.glWidget.selected_planes.clear()
        elif k == Qt.Key_2:
            self.glWidget.widget_state.select_mode = SelectionMode.DO_NOT_SELECT
        elif k == Qt.Key_3:
            self.glWidget.widget_state.select_mode = SelectionMode.PLANE
        elif k == Qt.Key_4:
            self.glWidget.widget_state.select_cascade = CascadeMode.DO_NOT_CASCADE
        elif k == Qt.Key_5:
            self.glWidget.widget_state.select_cascade = CascadeMode.UNION
        elif k == Qt.Key_6:
            self.glWidget.widget_state.select_cascade = CascadeMode.DIFFERENCE
        elif k == Qt.Key_7:
            pass
        elif k == Qt.Key_8:
            pass
        elif k == Qt.Key_9:
            pass
        elif k == Qt.Key_0:
            pass
        self.glWidget.repaint()


class MouseStamp:
    def __init__(self):
        self.x = -1
        self.y = -1
        self.time_counter = -1
        self.button = None
        self.is_pressed = False


class SelectionMode(Enum):
    DO_NOT_SELECT = 0
    VERTEX = 1
    LINE = 2
    PLANE = 3


class CascadeMode(Enum):
    DO_NOT_CASCADE = 0
    UNION = 1
    DIFFERENCE = 2


class ProgramMode:
    def __init__(self):
        self.select_mode = SelectionMode.DO_NOT_SELECT
        self.select_cascade = CascadeMode.DO_NOT_CASCADE
        pass


class GLWidget(QGLWidget):
    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)

        self.size = QSize(1000, 1000)
        self.mouseTrack = False
        self.rot_x = self.rot_y = self.rot_z = 0
        self.vertex = []
        self.mouse_prev_pos = 0

        self.widget_state = ProgramMode()
        #self.selected_vertices = dict()
        self.selected_planes = set()

        self.cam = mini3d.Camera(0, 0, 3)
        self.cam.rotate_up_axis(180)

        self.modeler = mini3d.Mesh()
        self.modeler.polygon_mode = (GL_FRONT_AND_BACK, GL_LINE)
        self.v1 = self.modeler.append_vertex(0, 0, 0)
        self.v2 = self.modeler.append_vertex(0.5, 0, 0)
        self.v3 = self.modeler.append_vertex(0.5, 0.5, 0)
        self.v4 = self.modeler.append_vertex(0, 0.5, 0)
        new_p = self.modeler.make_plane(self.v1, self.v2, self.v3)
        new_p2 = self.modeler.make_plane(self.v1, self.v3, self.v4)

        self.selected_planes.add(new_p)
        self.selected_planes.add(new_p2)
        copy_p = self.modeler.copy_planes(self.selected_planes)
        self.selected_planes = copy_p
        for p in self.selected_planes:
            for v in p:
                v.setZ(-1)
        for p in self.modeler.planes:
            p.correct_normal()
        self.selected_planes.clear()

        self.grid = mini3d.Mesh()
        self.grid.polygon_mode = (GL_FRONT_AND_BACK, GL_LINE)
        self.grid.append_vertex(1000, 0, 0)
        self.grid.append_vertex(-1000, 0, 0)
        self.grid.make_line_with_latest()

        self.grid.append_vertex(0, 1000, 0)
        self.grid.append_vertex(0, -1000, 0)
        self.grid.make_line_with_latest()

        self.grid.append_vertex(0, 0, 1000)
        self.grid.append_vertex(0, 0, -1000)
        self.grid.make_line_with_latest()

    def sizeHint(self):
        return QSize(1000, 1000)

    def initializeGL(self):
        glClearColor(0, 0, 0, 0)
        glEnable(GL_DEPTH_TEST)

        #glEnable(GL_CULL_FACE)
        #glFrontFace(GL_CCW)

        glShadeModel(GL_FLAT)


        # self._createVertexBuffer()

    def paintGL(self):
        self._draw()

    def resizeGL(self, width, height):
        print("resize : ", width, height)
        glViewport(0, 0, width, height)
        # side = min(width, height)
        # if side < 0:
        #    return
        # glViewport((width - side) // 2, (height - side) // 2, side, side)
        return

    def _draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.cam.update_view()

        glMatrixMode(GL_MODELVIEW)
        #glMatrixMode(GL_PROJECTION)
        #glLoadIdentity()
        #glOrtho(-1, 1, -1, 1, -100, 100)
        # glFrustum(-1.0, 1.0, -1.0, 1.0, 0.1, 1)

        #glMatrixMode(GL_MODELVIEW)
        #glLoadIdentity()
        # gluLookAt(1, 1, 10, 0, 0, 0, 1, 0, 0)
        #glRotate(self.rot_x, 1, 0, 0)
        #glRotate(self.rot_y, 0, 1, 0)
        #glRotate(self.rot_z, 0, 0, 1)

        self.modeler.draw()


        pos = self.modeler.pos
        up = pos + 0.1 * self.modeler.up
        left = pos + 0.1 * self.modeler.left
        direction = pos + 0.1 * self.modeler.direction
        glBegin(GL_LINES)
        glColor3f(1, 0, 0)
        glVertex3f(pos.x(), pos.y(), pos.z())
        glVertex3f(up.x(), up.y(), up.z())
        glColor3f(0, 1, 0)
        glVertex3f(pos.x(), pos.y(), pos.z())
        glVertex3f(left.x(), left.y(), left.z())
        glColor3f(0, 0, 1)
        glVertex3f(pos.x(), pos.y(), pos.z())
        glVertex3f(direction.x(), direction.y(), direction.z())
        glEnd()

        glPointSize(20)
        glBegin(GL_POINTS)
        p = QVector3D(self.cam.pos)
        p += self.cam.dist_from_target * self.cam.direction

        glColor3f(0, 1, 1)
        glVertex3f(p.x(), p.y(), p.z())



        for p in self.selected_planes:
            avg = QVector3D(0, 0, 0)
            for v in p:
                avg = avg + v
                glColor3f(1, 1, 0)
                glVertex3f(v.x(), v.y(), v.z())
            avg = avg / 3
            glColor3f(0, 0, 1)
            glVertex3f(avg.x(), avg.y(), avg.z())
        glEnd()


        #self.grid.draw()
        glBegin(GL_LINES)
        glColor3f(1, 0, 0)
        glVertex3f(-1000, 0, 0)
        glVertex3f(1000, 0, 0)
        glColor3f(0, 1, 0)
        glVertex3f(0, -10000, 0)
        glVertex3f(0, 1000, 0)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, -1000)
        glVertex3f(0, 0, 1000)
        glEnd()




        glPointSize(10)
        glBegin(GL_POINTS)
        glColor3f(1, 1, 1)
        for coeff in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            for x in range(0, 10):
                x = x / 10
                glVertex3f(coeff[0] * x, coeff[1] * x, coeff[2] * x)

        glVertex3f(0, 0, 0)
        for v in self.vertex:
            glVertex3fv(v)
        glEnd()
        glFlush()

    def mousePressEvent(self, e: QMouseEvent) -> None:
        self.mouse_prev_pos = (e.x(), e.y())
        print("m press : ", e.x(), e.y())

        if e.button() == Qt.RightButton:
            start = QVector3D(*gluUnProject(e.x(), self.height() - e.y(), 0))
            end = QVector3D(*gluUnProject(e.x(), self.height() - e.y(), 1))
            collision = mini3d.ray_to_plane(
                start,
                end,
                self.cam.direction,
                self.cam.pos + self.cam.direction * self.cam.dist_from_target)
            if collision is not None:
                vertex = self.modeler.append_vertex(collision.x(), collision.y(), collision.z())
                self.modeler.make_plane_with_latest(-self.cam.direction)



        self.repaint()
        # for v in self.vertex:
        #    print(v)
        return

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        x = e.x()
        y = e.y()
        if e.button() == Qt.LeftButton:
            if self.widget_state.select_mode is SelectionMode.DO_NOT_SELECT:
                return
            start = gluUnProject(e.x(), self.height() - e.y(), 0)
            end = gluUnProject(e.x(), self.height() - e.y(), 1)
            generator = self.modeler.collision_with_ray(mini3d.MeshVertex(*start), mini3d.MeshVertex(*end))

            plane_to_camera = 999999999
            #selected_list = set()
            selected_plane = None
            for plane, collision_point in generator:
                depth = self.cam.pos.distanceToPoint(collision_point)
                if depth < plane_to_camera:
                    plane_to_camera = depth
                    selected_plane = plane
            if selected_plane is not None:
                #if self.widget_state.select_mode is SelectionMode.VERTEX:
                #    selected_list.add(selected_plane.nearest_vertex(collision_point))
                #elif self.widget_state.select_mode is SelectionMode.LINE:
                #    selected_list = set(selected_plane.nearest_line(collision_point))
                #elif self.widget_state.select_mode is SelectionMode.PLANE:
                #    selected_list = set(selected_plane.group[0:])

                if self.widget_state.select_cascade is CascadeMode.DO_NOT_CASCADE:
                    self.selected_planes.clear()
                    self.selected_planes.add(selected_plane)
                elif self.widget_state.select_cascade is CascadeMode.UNION:
                    self.selected_planes.add(selected_plane)
                elif self.widget_state.select_cascade is CascadeMode.DIFFERENCE:
                    if selected_plane in self.selected_planes:
                        self.selected_planes.remove(selected_plane)

            self.repaint()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        x = e.x()
        y = e.y()

        if self.mouse_prev_pos == 0:
            self.mouse_prev_pos = (x, y)
            return

        dx = x - self.mouse_prev_pos[0]
        dy = y - self.mouse_prev_pos[1]
        buttons = e.buttons()
        if buttons & Qt.LeftButton:
            self.rot_y = self.rot_y + dx
            self.rot_x = self.rot_x + dy
        elif buttons & Qt.MidButton:
            print('z')
            self.rot_z = self.rot_z + abs(dx) + abs(dy)

        print("m move : ", x, y, dx, dy)
        self.mouse_prev_pos = (x, y)
        self.repaint()

        return




def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
    print("finish")


if __name__ == "__main__":
    main()
