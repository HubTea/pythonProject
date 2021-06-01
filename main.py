# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from OpenGL.GL import *
from OpenGL.GLU import *

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QMouseEvent, QKeyEvent, QVector3D, QCursor
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtOpenGL import QGLWidget

import numpy as np
import mini3d
from enum import Enum, auto
import threading


lock = threading.Lock()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.glWidget = GLWidget()
        self.setCentralWidget(self.glWidget)
        self.stack = []

    def set_control_mode(self, mode: 'ControlMode'):
        """선형 변환의 종류 변경 및 변경 순간의 마우스 위치 저장"""
        m = self.glWidget.widget_state.control_mode
        if m is not mode:
            self.glWidget.widget_state.control_mode = mode
            p = self.glWidget.mapFromGlobal(QCursor.pos())
            self.glWidget.mouse_snapshot = (p.x(), p.y())
        else:
            self.glWidget.widget_state.control_mode = ControlMode.DO_NOT_ANYTHING

    def keyPressEvent(self, e: QKeyEvent) -> None:
        k = e.key()
        if k == Qt.Key_Left:         # 카메라를 왼쪽으로 이동
            self.glWidget.cam.move(0, 0, 0.1)
        elif k == Qt.Key_Right:      # 카메라를 오른쪽으로 이동
            self.glWidget.cam.move(0, 0, -0.1)
        elif k == Qt.Key_Up:         # 카메라를 위쪽으로 이동
            self.glWidget.cam.move(0, 0.1, 0)
        elif k == Qt.Key_Down:       # 카메라를 아래쪽으로 이동
            self.glWidget.cam.move(0, -0.1, 0)
        elif k == Qt.Key_Q:          # 카메라를 반시계방향으로 회전
            self.glWidget.cam.rotate_direction_axis(3)
        elif k == Qt.Key_E:          # 카메라를 시계방향으로 회전
            self.glWidget.cam.rotate_direction_axis(-3)
        elif k == Qt.Key_W:          # 카메라가 위쪽을 바라보도록 회전
            self.glWidget.cam.rotate_left_axis(3)
        elif k == Qt.Key_S:          # 카메라가 아래쪽을 바라보도록 회전
            self.glWidget.cam.rotate_left_axis(-3)
        elif k == Qt.Key_A:          # 카메라가 왼쪽을 바라보도록 회전
            self.glWidget.cam.rotate_up_axis(-3)
        elif k == Qt.Key_D:          # 카메라가 오른쪽을 바라보도록 회전
            self.glWidget.cam.rotate_up_axis(3)
        elif k == Qt.Key_R:          # 카메라를 앞으로 이동시킴
            self.glWidget.cam.zoom()
        elif k == Qt.Key_F:          # 카메라를 뒤로 이동시킴
            self.glWidget.cam.zoom(False)
        elif k == Qt.Key_T:          # 카메라가 물체의 위를 보도록 회전
            self.glWidget.cam.rev_left_axis(3)
        elif k == Qt.Key_G:          # 카메라가 물체의 아래를 보도록 회전
            self.glWidget.cam.rev_left_axis(-3)
        elif k == Qt.Key_Y:          # 카메라가 물체의 왼쪽을 보도록 회전
            self.glWidget.cam.rev_up_axis(3)
        elif k == Qt.Key_H:          # 카메라가 물체의 오른쪽을 보도록 회전
            self.glWidget.cam.rev_up_axis(-3)
        elif k == Qt.Key_I:          # 선택된 면의 노멀 벡터의 방향을 뒤집음
            for p in self.glWidget.selected_planes:
                p.inverse_direction()
                p.inverse_normal()
        elif k == Qt.Key_1:          # 선택된 면의 목록을 비움
            self.glWidget.selected_planes.clear()
        elif k == Qt.Key_2:          # 아무것도 선택 안 함
            self.glWidget.widget_state.select_mode = SelectionMode.DO_NOT_SELECT
        elif k == Qt.Key_3:          # 면 선택
            self.glWidget.widget_state.select_mode = SelectionMode.PLANE
        elif k == Qt.Key_4:          # 오직 하나의 면만 선택
            self.glWidget.widget_state.select_cascade = CascadeMode.DO_NOT_CASCADE
        elif k == Qt.Key_5:          # 계속 이어서 선택함
            self.glWidget.widget_state.select_cascade = CascadeMode.UNION
        elif k == Qt.Key_6:          # 선택된 면의 목록에서 제외
            self.glWidget.widget_state.select_cascade = CascadeMode.DIFFERENCE
        elif k == Qt.Key_7:          # x축 선택
            if self.glWidget.widget_state.axis[0]:
                self.glWidget.widget_state.axis[0] = 0
            else:
                self.glWidget.widget_state.axis[0] = 1
        elif k == Qt.Key_8:          # y축 선택
            if self.glWidget.widget_state.axis[1]:
                self.glWidget.widget_state.axis[1] = 0
            else:
                self.glWidget.widget_state.axis[1] = 1
        elif k == Qt.Key_9:          # z축 선택
            if self.glWidget.widget_state.axis[2]:
                self.glWidget.widget_state.axis[2] = 0
            else:
                self.glWidget.widget_state.axis[2] = 1
        elif k == Qt.Key_0:
            pass
            #self.glWidget.widget_state.select_mode = SelectionMode.ALL
        elif k == Qt.Key_Minus:      # 선택된 면 복제 후 기존 면 삭제
            s = set(self.glWidget.selected_planes)
            cv, sd = self.glWidget.modeler.copy_planes(s)
            for p in self.glWidget.selected_planes:
                self.glWidget.modeler.delete_plane(p)
            self.glWidget.selected_planes = cv
        elif k == Qt.Key_Equal:      # 선택된 면 복제, 기존 면은 그대로 둠
            s = set(self.glWidget.selected_planes)
            cv, sd = self.glWidget.modeler.copy_planes(s, False)
            self.glWidget.selected_planes = cv
        elif k == Qt.Key_Z:          # 선택된 면 평행이동
            self.set_control_mode(ControlMode.TRANSLATION)
        elif k == Qt.Key_X:          # 선택된 면 회전
            self.set_control_mode(ControlMode.ROTATION)
        elif k == Qt.Key_C:          # 선택된 면 크기 변경
            self.set_control_mode(ControlMode.SCALING)
        elif k == Qt.Key_M:          # subdivision 단계 하나 증가. 증가 이전 3D 물체는 스택에 push
            self.stack.append(self.glWidget.modeler)
            self.glWidget.modeler = mini3d.catmull_clark(self.glWidget.modeler)
            if self.stack[-1] is self.glWidget.modeler:
                self.stack.pop(-1)
            self.glWidget.selected_planes.clear()
        elif k == Qt.Key_N:          # subdivisㅑon 단계 하나 감소. 스택에서 pop
            if len(self.stack) > 0:
                self.glWidget.modeler = self.stack[-1]
                self.stack.pop(-1)
        elif k == Qt.Key_Delete:     # 선택된 면 삭제
            for p in self.glWidget.selected_planes:
                self.glWidget.modeler.delete_plane(p)
        elif k == Qt.Key_F1:         # mesh.3d 파일에서 3D 물체를 불러 옴
            self.glWidget.modeler.load('./mesh.3d')
        elif k == Qt.Key_F2:         # mesh.3d 파일에 3D 물체를 저장함
            self.glWidget.modeler.save('./mesh.3d')
        self.glWidget.repaint()


class SelectionMode(Enum):
    """
    면 선택 시 모드
    DO_NOT_SELECT : 면을 선택하지 않음
    ALL : 선택된 면이 포함된 물체의 모든 면을 선택함
    PLANE : 선택된 면만 선택함
    """
    DO_NOT_SELECT = 0
    ALL = auto()
    PLANE = 3


class CascadeMode(Enum):
    """
    면 선택 시 모드
    DO_NOT_CASCADE : 오직 하나의 면만 선택 가능
    UNION : 기존에 선택된 면의 집합에 합함
    DEFFERENCE : 기존에 선택된 면의 집합에서 뺌
    """
    DO_NOT_CASCADE = 0
    UNION = 1
    DIFFERENCE = 2


class ControlMode(Enum):
    """선형 변환의 종류"""
    DO_NOT_ANYTHING = auto()
    TRANSLATION = auto()
    ROTATION = auto()
    SCALING = auto()


class ProgramMode:
    """프로그램의 상태를 저장"""
    def __init__(self):
        self.select_mode = SelectionMode.DO_NOT_SELECT
        self.select_cascade = CascadeMode.DO_NOT_CASCADE

        self.control_mode = ControlMode.DO_NOT_ANYTHING
        self.axis = [1, 0, 0]

        self.palette = (255, 255, 255)
        pass


class GLWidget(QGLWidget):
    """모델링 작업이 이루어지는 Widget"""
    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)

        self.size = QSize(1000, 1000)
        self.setMouseTracking(True)

        self.widget_state = ProgramMode()
        self.selected_planes = set()
        self.mouse_snapshot = None

        # 카메라 생성
        self.cam = mini3d.Camera(0, 0, 3)
        self.cam.rotate_up_axis(180)

        # 직육면체 생성
        self.modeler = mini3d.Mesh()
        self.modeler.polygon_mode = (GL_FRONT_AND_BACK, GL_LINE)
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

        cv, sd = self.modeler.copy_planes(set(self.modeler.planes))
        for p in cv:
            for v in p:
                v.setZ(-1)
            p.inverse_direction()
        for p in self.modeler.planes:
            p.correct_normal()

    def sizeHint(self):
        return QSize(1000, 1000)

    def initializeGL(self):
        glClearColor(0, 0, 0, 0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glFrontFace(GL_CCW)
        glShadeModel(GL_FLAT)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.5, 0.5, 0.5, 1])

    def paintGL(self):
        self._draw()

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)

    def _draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.cam.update_view()

        glMatrixMode(GL_MODELVIEW)

        self.modeler.draw()

        glPointSize(20)
        glBegin(GL_POINTS)

        # 카메라가 회전할 때 중심이 되는 지점 표시
        p = QVector3D(self.cam.pos)
        p += self.cam.dist_from_target * self.cam.direction
        glColor3f(0, 1, 1)
        glVertex3f(p.x(), p.y(), p.z())

        # 선택된 면의 무게중심과 포함된 정점을 큰 점으로 표시
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

        # x축은 빨간색, y축은 초록색, z축은 파란색으로 그림
        # 축이 선택된 상태면 굵게 그림
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

        # 각 축마다 +방향으로 0.1간격으로 10개의 점 표시
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
        """화면 상의 2차원 좌표를 통해 ray casting 수행. 충돌한 VertexGroup 객체 반환"""
        if self.widget_state.select_mode is SelectionMode.DO_NOT_SELECT:
            return
        start = gluUnProject(mx, self.height() - my, 0)
        end = gluUnProject(mx, self.height() - my, 1)

        generator = self.modeler.collision_with_ray(mini3d.MeshVertex(*start), mini3d.MeshVertex(*end))

        plane_to_camera = 999999999
        selected_plane = None

        # 모든 면에 대해 충돌 검사. 충돌한 면 중에서 사용자의 시점에서 제일 가까운 면 선택
        for plane, collision_point in generator:
            depth = self.cam.pos.distanceToPoint(collision_point)
            if depth < plane_to_camera:
                plane_to_camera = depth
                selected_plane = plane
        return selected_plane

    def select_plane(self, mx, my):
        """get_plane 으로 VertexGroup 객체를 얻고 이를 self.selected_planes 에 추가 혹은 제거함"""
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
        """3D 물체에서 선택된 면의 정점의 위치를 바꿈"""
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
                # 무게중심 계산
                s = QVector3D(0, 0, 0)
                for v in v_set:
                    s += v
                s /= len(v_set)

                # 변환 행렬 계산
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
                # 무게중심 계산
                s = QVector3D(0, 0, 0)
                for v in v_set:
                    s += v
                s /= len(v_set)

                # 변환 행렬 계산
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
        mx = e.x()
        my = e.y()
        if self.widget_state.select_mode is not SelectionMode.DO_NOT_SELECT:
            if e.button() == Qt.LeftButton:     # 면 선택
                self.select_plane(mx, my)
            elif e.button() == Qt.RightButton:  # 해당 면에 색 지정
                p = self.get_plane(mx, my)
                if p is not None:
                    lock.acquire()
                    p.color = self.widget_state.palette
                    lock.release()
            elif e.button() == Qt.MiddleButton: # 모든 면 선택
                p = self.get_plane(mx, my)
                self.selected_planes = set(p.owner.planes)
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
            self.transform_mesh(0.1 * (dm - dp))
            self.mouse_snapshot = (mx, my)
        return


def cmd_line(widget):
    """콘솔 창에서 명령어를 입력받고 해당 작업 수행"""
    while True:
        cmdline = input('input "end" to finish cmd line>>')
        tokens = cmdline.split(' ')
        if len(tokens) == 0:
            continue

        cmd = tokens[0]
        try:
            if cmd == 'rgb':        # 색 설정. rgb r g b
                lock.acquire()
                widget.widget_state.palette = (int(tokens[1]), int(tokens[2]), int(tokens[3]))
                lock.release()
            elif cmd == 'end':
                return
        except (ValueError, IndexError):
            print('wrong command. input again')
        widget.repaint()


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
