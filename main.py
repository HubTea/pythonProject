# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from OpenGL.GL import *
from OpenGL.GLU import *

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QMouseEvent, QKeyEvent, QVector3D, QCursor
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout
from PyQt5.QtOpenGL import QGLWidget

import numpy as np
import mini3d
from enum import Enum, auto
import time
import threading


lock = threading.Lock()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.glWidget = GLWidget()
        self.setCentralWidget(self.glWidget)
        self.stack = []

    def set_control_mode(self, mode: 'ControlMode'):
        m = self.glWidget.widget_state.control_mode
        if m is not mode:
            self.glWidget.widget_state.control_mode = mode
            p = self.glWidget.mapFromGlobal(QCursor.pos())
            self.glWidget.mouse_snapshot = (p.x(), p.y())
        else:
            self.glWidget.widget_state.control_mode = ControlMode.DO_NOT_ANYTHING

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
            self.glWidget.cam.rotate_direction_axis(3)
        elif k == Qt.Key_E:
            self.glWidget.cam.rotate_direction_axis(-3)
        elif k == Qt.Key_W:
            self.glWidget.cam.rotate_left_axis(3)
        elif k == Qt.Key_S:
            self.glWidget.cam.rotate_left_axis(-3)
        elif k == Qt.Key_A:
            self.glWidget.cam.rotate_up_axis(-3)
        elif k == Qt.Key_D:
            self.glWidget.cam.rotate_up_axis(3)
        elif k == Qt.Key_R:
            self.glWidget.cam.zoom()
        elif k == Qt.Key_F:
            self.glWidget.cam.zoom(False)
        elif k == Qt.Key_T:
            self.glWidget.cam.rev_left_axis(3)
        elif k == Qt.Key_G:
            self.glWidget.cam.rev_left_axis(-3)
        elif k == Qt.Key_Y:
            self.glWidget.cam.rev_up_axis(3)
        elif k == Qt.Key_H:
            self.glWidget.cam.rev_up_axis(-3)
        elif k == Qt.Key_I:
            for p in self.glWidget.selected_planes:
                p.inverse_direction()
                p.inverse_normal()
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
            if self.glWidget.widget_state.axis[0]:
                self.glWidget.widget_state.axis[0] = 0
            else:
                self.glWidget.widget_state.axis[0] = 1
        elif k == Qt.Key_8:
            if self.glWidget.widget_state.axis[1]:
                self.glWidget.widget_state.axis[1] = 0
            else:
                self.glWidget.widget_state.axis[1] = 1
        elif k == Qt.Key_9:
            if self.glWidget.widget_state.axis[2]:
                self.glWidget.widget_state.axis[2] = 0
            else:
                self.glWidget.widget_state.axis[2] = 1
        elif k == Qt.Key_0:
            s = set(self.glWidget.selected_planes)
            cv, sd = self.glWidget.modeler.copy_planes(s)
            self.glWidget.selected_planes = cv
        elif k == Qt.Key_Minus:
            s = set(self.glWidget.selected_planes)
            cv, sd = self.glWidget.modeler.copy_planes(s)
            for p in self.glWidget.selected_planes:
                self.glWidget.modeler.delete_plane(p)
            self.glWidget.selected_planes = cv
        elif k == Qt.Key_Z:
            self.set_control_mode(ControlMode.TRANSLATION)
        elif k == Qt.Key_X:
            self.set_control_mode(ControlMode.ROTATION)
        elif k == Qt.Key_C:
            self.set_control_mode(ControlMode.SCALING)
        elif k == Qt.Key_M:
            self.stack.append(self.glWidget.modeler)
            self.glWidget.modeler = mini3d.catmull_clark(self.glWidget.modeler)
            if self.stack[-1] is self.glWidget.modeler:
                self.stack.pop(-1)
            self.glWidget.selected_planes.clear()
        elif k == Qt.Key_N:
            if len(self.stack) > 0:
                self.glWidget.modeler = self.stack[-1]
                self.stack.pop(-1)
        elif k == Qt.Key_Delete:
            for p in self.glWidget.selected_planes:
                self.glWidget.modeler.delete_plane(p)
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


class ControlMode(Enum):
    DO_NOT_ANYTHING = auto()
    TRANSLATION = auto()
    ROTATION = auto()
    SCALING = auto()


class ProgramMode:
    def __init__(self):
        self.select_mode = SelectionMode.DO_NOT_SELECT
        self.select_cascade = CascadeMode.DO_NOT_CASCADE

        self.control_mode = ControlMode.DO_NOT_ANYTHING
        self.axis = [1, 0, 0]

        self.palette = (255, 255, 255)
        pass


class GLWidget(QGLWidget):
    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)

        self.size = QSize(1000, 1000)
        self.setMouseTracking(True)

        self.vertex = []

        self.widget_state = ProgramMode()
        self.selected_planes = set()
        self.mouse_snapshot = None

        self.cam = mini3d.Camera(0, 0, 3)
        self.cam.rotate_up_axis(180)

        self.modeler = mini3d.Mesh()
        #self.modeler.polygon_mode = (GL_FRONT_AND_BACK, GL_LINE)
        m = self.modeler

        v1 = m.append_vertex(0, 0, 0)
        v2 = m.append_vertex(0.5, 0, 0)
        v3 = m.append_vertex(0.5, 0.5, 0)
        v4 = m.append_vertex(0, 0.5, 0)
        v5 = m.append_vertex(0.25, 0.25, 0)
        m.make_plane(v1, v2, v5, QVector3D(0, 0, 1))
        m.make_plane(v2, v3, v5, QVector3D(0, 0, 1))
        m.make_plane(v3, v4, v5, QVector3D(0, 0, 1))
        m.make_plane(v4, v1, v5, QVector3D(0, 0, 1))
        '''
        v6 = m.append_vertex(0, 0, -1)
        v7 = m.append_vertex(0.5, 0, -1)
        v8 = m.append_vertex(0.5, 0.5, -1)
        v9 = m.append_vertex(0, 0.5, -1)
        v10 = m.append_vertex(0.25, 0.25, -1)

        

        m.make_plane(v6, v7, v10)
        m.make_plane(v7, v8, v10)
        m.make_plane(v8, v9, v10)
        m.make_plane(v9, v6, v10)

        m.make_plane(v1, v2, v6)
        m.make_plane(v6, v7, v2)
        m.make_plane(v2, v3, v7)
        m.make_plane(v7, v8, v3)
        m.make_plane(v3, v4, v8)
        m.make_plane(v8, v9, v4)
        m.make_plane(v4, v1, v9)
        m.make_plane(v9, v6, v1)
        '''

        cv, sd = self.modeler.copy_planes(set(self.modeler.planes))
        for p in cv:
            for v in p:
                v.setZ(-1)
            p.inverse_direction()
        for p in self.modeler.planes:
            p.correct_normal()

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

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT)

        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.5, 0.5, 0.5, 1])
        # self._createVertexBuffer()

    def paintGL(self):
        s = time.perf_counter()
        self._draw()
        print('drawing time : ', time.perf_counter() - s)

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

        self.modeler.draw()
        #self.modeler2.draw()

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

        glLightfv(GL_LIGHT0, GL_POSITION, [10, 10, 10, 1])

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


        for i, t in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
            if self.widget_state.axis[i] == 0:
                glLineWidth(1)
            else:
                glLineWidth(5)
            glBegin(GL_LINES)
            glColor3f(t[0], t[1], t[2])
            glVertex3f(1000 * t[0], 1000 * t[1], 1000 * t[2])
            glVertex3f(-1000 * t[0], -1000 * t[1], -1000 * t[2])
            glEnd()
        '''
        if self.widget_state.axis[0] == 0:
            glLineWidth(1)
        else:
            glLineWidth(3)
        glColor3f(0, 1, 0)
        glVertex3f(0, -10000, 0)
        glVertex3f(0, 1000, 0)

        glColor3f(0, 0, 1)
        glVertex3f(0, 0, -1000)
        glVertex3f(0, 0, 1000)
        '''


        glPointSize(10)
        glBegin(GL_POINTS)
        glColor3f(1, 1, 1)
        for coeff in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            for x in range(0, 10):
                x = x / 10
                glVertex3f(coeff[0] * x, coeff[1] * x, coeff[2] * x)
        glEnd()
        glFlush()

    def get_plane(self, mx, my):
        if self.widget_state.select_mode is SelectionMode.DO_NOT_SELECT:
            return
        start = gluUnProject(mx, self.height() - my, 0)
        end = gluUnProject(mx, self.height() - my, 1)

        generator = self.modeler.collision_with_ray(mini3d.MeshVertex(*start), mini3d.MeshVertex(*end))

        plane_to_camera = 999999999
        selected_plane = None
        for plane, collision_point in generator:
            depth = self.cam.pos.distanceToPoint(collision_point)
            if depth < plane_to_camera:
                plane_to_camera = depth
                selected_plane = plane
        return selected_plane

    def select_plane(self, mx, my):
        selected_plane = self.get_plane(mx, my)
        if selected_plane is not None:
            if self.widget_state.select_cascade is CascadeMode.DO_NOT_CASCADE:
                self.selected_planes.clear()
                self.selected_planes.add(selected_plane)
            elif self.widget_state.select_cascade is CascadeMode.UNION:
                self.selected_planes.add(selected_plane)
            elif self.widget_state.select_cascade is CascadeMode.DIFFERENCE:
                if selected_plane in self.selected_planes:
                    self.selected_planes.remove(selected_plane)

    def transform_mesh(self, arg: 'float'):
        m = self.widget_state.control_mode
        if m is ControlMode.DO_NOT_ANYTHING:
            return
        v_set = set()
        for p in self.selected_planes:
            for v in p:
                v_set.add(v)

        if m is ControlMode.TRANSLATION:
            delta = arg * QVector3D(*self.widget_state.axis)
            for v in v_set:
                v += delta
            for p in self.modeler.planes:   # copy직후에는 normal벡터 설정이 어려우므로 이 단계에서 VertexGroup.direction기반으로 normal벡터 설정
                p.correct_normal()
        elif m is ControlMode.SCALING:
            if len(v_set) != 0:
                s = QVector3D(0, 0, 0)
                for v in v_set:
                    s += v
                s /= len(v_set)

                ratio = arg * QVector3D(*self.widget_state.axis) + QVector3D(1, 1, 1)
                mat = np.array(
                    [[ratio[0], 0, 0],
                     [0, ratio[1], 0],
                     [0, 0, ratio[2]]])

                for v in v_set:
                    t = v - s
                    b = np.array([[t.x()], [t.y()], [t.z()]])
                    pos = mat.dot(b)
                    t = QVector3D(pos[0], pos[1], pos[2]) + s
                    v.setX(t.x())
                    v.setY(t.y())
                    v.setZ(t.z())
                for p in self.modeler.planes:
                    p.correct_direction()
        elif m is ControlMode.ROTATION:
            if len(v_set) != 0:
                s = QVector3D(0, 0, 0)
                for v in v_set:
                    s += v
                s /= len(v_set)

                if self.widget_state.axis[0] != 0:
                    mat = np.array(
                        [[1, 0, 0],
                         [0, np.cos(arg), -np.sin(arg)],
                         [0, np.sin(arg), np.cos(arg)]])
                elif self.widget_state.axis[1] != 0:
                    mat = np.array(
                        [[np.cos(arg), 0, np.sin(arg)],
                         [0, 1, 0],
                         [-np.sin(arg), 0, np.cos(arg)]])
                else:
                    mat = np.array(
                        [[np.cos(arg), -np.sin(arg), 0],
                         [np.sin(arg), np.cos(arg), 0],
                         [0, 0, 1]])

                for v in v_set:
                    t = v - s
                    b = np.array([[t.x()], [t.y()], [t.z()]])
                    pos = mat.dot(b)
                    t = QVector3D(pos[0], pos[1], pos[2]) + s
                    v.setX(t.x())
                    v.setY(t.y())
                    v.setZ(t.z())
                for p in self.modeler.planes:
                    p.correct_direction()
        self.repaint()
        return

    def mousePressEvent(self, e: QMouseEvent) -> None:
        #print("m press : ", e.x(), e.y())
        mx = e.x()
        my = e.y()
        if self.widget_state.select_mode is SelectionMode.DO_NOT_SELECT:
            if e.button() == Qt.RightButton:
                start = QVector3D(*gluUnProject(mx, self.height() - my, 0))
                end = QVector3D(*gluUnProject(mx, self.height() - my, 1))
                collision = mini3d.ray_to_plane(
                    start,
                    end,
                    self.cam.direction,
                    self.cam.pos + self.cam.direction * self.cam.dist_from_target)
                if collision is not None:
                    vertex = self.modeler.append_vertex(collision.x(), collision.y(), collision.z())
                    self.modeler.make_plane_with_latest(-self.cam.direction)
        else:
            if e.button() == Qt.LeftButton:
                self.select_plane(mx, my)
            elif e.button() == Qt.RightButton:
                p = self.get_plane(mx, my)
                if p is not None:
                    lock.acquire()
                    p.color = self.widget_state.palette
                    lock.release()
        self.repaint()
        return

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        mx = e.x()
        my = e.y()

        cx = self.width() / 2
        cy = self.height() / 2
        if self.mouse_snapshot is not None:
            px = self.mouse_snapshot[0]
            py = self.mouse_snapshot[1]

            dp = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            dm = np.sqrt((mx - cx) ** 2 + (my - cy) ** 2)
            #print('m', mx, my)
            #print('p', px, py)
            #print('c', cx, cy)
            self.transform_mesh(0.1 * (dm - dp))
            self.mouse_snapshot = (mx, my)
        return


def cmd_line(widget):
    while True:
        cmdline = input('input "end" to finish cmd line>>')
        tokens = cmdline.split(' ')
        if len(tokens) == 0:
            continue

        cmd = tokens[0]
        try:
            if cmd == 'rgb':
                lock.acquire()
                widget.widget_state.palette = (int(tokens[1]), int(tokens[2]), int(tokens[3]))
                lock.release()
            elif cmd == 'end':
                return
        except (ValueError, IndexError):
            print('wrong command. input again')


def main():
    app = QApplication([])
    window = MainWindow()
    t = threading.Thread(target=cmd_line, args=(window.glWidget,), daemon=True)
    t.start()
    window.show()
    app.exec_()
    print("finish")


if __name__ == "__main__":
    main()
