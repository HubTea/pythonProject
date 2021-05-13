'''
QVector를 numpy array로 교체
Mesh에 선분 데이터 추가로 저장
VertexGroup 노멀벡터가 최신인지 표시하는 플래그 추가 및 정점 좌표 변경시 업데이트 되도록 수정
추가적인 충돌체크 함수 제작, Mesh.collision_with_ray의 충돌체크 코드 함수로 따로 제작
'''





from PyQt5.QtGui import QMatrix4x4
from PyQt5.QtGui import QVector3D
from PyQt5.QtGui import QVector4D

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np

pi_div_180 = np.pi / 180


def inner_product(a, b):
    product = a.toVector4D() * b.toVector4D()
    return product[0] + product[1] + product[2] + product[3]


def cross_product(a, b):
    x = a[1] * b[2] - b[1] * a[2]
    y = -(a[0] * b[2] - b[0] * a[2])
    z = a[0] * b[1] - b[0] * a[1]
    return QVector3D(x, y, z)


def triple_product(a, b, c):
    return inner_product(cross_product(a, b), c)


def normal_vector(org, v1, v2):
    return cross_product(v1 - org, v2 - org)


def rotate(vector_as_x, vector_as_y, degree):
    c = np.cos(pi_div_180 * degree)
    s = np.sin(pi_div_180 * degree)
    temp_y_axis = c * vector_as_y - s * vector_as_x
    temp_x_axis = c * vector_as_x + s * vector_as_y
    return temp_x_axis, temp_y_axis


def ray_to_plane(start, end, normal, point):
    t = inner_product(normal, point - start) / inner_product(normal, end - start)
    if 0 <= t <= 1:
        return start + t * (end - start)


def is_inner_line(v1: 'MeshVertex', v2: 'MeshVertex', p_set: 'set[VertexGroup]') -> bool:
    count = 0
    for p in v1.adjacent_plane:
        if p in p_set and v2 in p:
            count += 1
    return count > 1


def get_inner_line(p1: 'VertexGroup', p2: 'VertexGroup') -> 'tuple[MeshVertex, MeshVertex]':
    count = 0
    intersection = []
    for v1 in p1:
        if v1 in p2:
            count += 1
            intersection.append(v1)
    if count == 2:
        return tuple(intersection)


class WorldObject:
    def __init__(self, object_name="object", x=0, y=0, z=0):
        self.pos = QVector3D(x, y, z)
        self.direction = QVector3D(0, 0, 1)
        self.up = QVector3D(0, 1, 0)
        self.left = QVector3D(1, 0, 0)

        self.visible = True
        self.name = object_name
        return

    def set_pos(self, x, y, z):
        self.pos.x = x
        self.pos.y = y
        self.pos.z = z

    def move(self, forward, up, left):
        self.pos += left * self.left + up * self.up + forward * self.direction

    def rotate_direction_axis(self, degree):
        self.up, self.left = rotate(self.up, self.left, degree)

    def rotate_up_axis(self, degree):
        self.left, self.direction = rotate(self.left, self.direction, degree)

    def rotate_left_axis(self, degree):
        self.direction, self.up = rotate(self.direction, self.up, degree)

    def draw(self):
        pass


class VertexGroup:
    LINE = 2
    TRIANGLE = 3

    def __init__(self, v1, v2, v3=None):
        if v3 is None:
            self.group = (v1, v2)
            self.type = VertexGroup.LINE
        else:
            self.group = (v1, v2, v3)
            self.type = VertexGroup.TRIANGLE
            self.direction = QVector3D(1, 0, 0)

    def is_line(self):
        return self.type == VertexGroup.LINE

    def is_triangle(self):
        return self.type == VertexGroup.TRIANGLE

    def nearest_vertex(self, org):
        li = []
        for v in self.group:
            li.append((v.distanceToPoint(org), v))
        li.sort()
        return li[0][1]

    def nearest_line(self, org):
        li = []
        if self.type == VertexGroup.LINE:
            return self.group
        else:
            for start, end in ((0, 1), (1, 2), (2, 0)):
                v1 = self.group[start]
                v2 = self.group[end]
                direction = (v1 - v2)
                direction.normalize()
                distance = org.distanceToLine(v2, direction)
                li.append((distance, v1, v2))
            li.sort()
            return li[0][1], li[0][2]

    def opposite(self, org: 'MeshVertex') -> 'tuple[MeshVertex, MeshVertex]':
        op = []
        for v in self.group:
            if v is not org:
                op.append(v)
        return tuple(op)

    def inverse_normal(self):
        if self.type == VertexGroup.TRIANGLE:
            self.group = (self.group[0], self.group[2], self.group[1])

    def get_normal(self):
        return cross_product(self.group[1] - self.group[0], self.group[2] - self.group[0])

    def set_direction(self, direction):
        self.direction = direction

    def correct_normal(self):
        normal = self.get_normal()
        if inner_product(normal, self.direction) < 0:
            self.inverse_normal()

    def __iter__(self):
        return self.group.__iter__()

    def __len__(self):
        return len(self.group)

    def __getitem__(self, item):
        return self.group[item]

    def __contains__(self, item):
        for x in self.group:
            if item is x:
                return True
        return False

    def __hash__(self):
        return id(self)


class MeshVertex(QVector3D):
    def __init__(self, x, y, z):
        QVector3D.__init__(self, x, y, z)
        self.adjacent_plane = []

    def push_plane(self, plane):
        self.adjacent_plane.append(plane)

    def pop_plane(self, plane):
        for (i, p) in self.adjacent_plane:
            if p is plane:
                self.adjacent_plane[i], self.adjacent_plane[-1] = self.adjacent_plane[-1], self.adjacent_plane[i]
                self.adjacent_plane.pop()
                break

    def __hash__(self):
        return id(self)


class Mesh(WorldObject):
    def __init__(self):
        WorldObject.__init__(self)
        self.vertices = []
        self.planes = []
        self.collision_check = True
        self.catmull_clark = 1
        self.polygon_mode = (GL_FRONT_AND_BACK, GL_FILL)
        return

    def append_vertex(self, x, y, z):
        self.vertices.append(MeshVertex(x, y, z))
        return self.vertices[-1]

    def delete_vertex(self, vertex):
        for (i, v) in enumerate(self.vertices):
            if v is vertex:
                self.vertices[i], self.vertices[-1] = self.vertices[-1], self.vertices[i]
                self.vertices.pop()

                for plane in v.adjacent_plane:
                    for adj_vertex in plane:
                        if adj_vertex is not v:
                            adj_vertex.pop_plane(plane)
                    v.pop_plane(plane)
                    idx = self.planes.index(plane)
                    self.planes[idx], self.planes[-1] = self.planes[-1], self.planes[idx]
                    self.planes.pop()
                return

    def make_plane(self, v1, v2, v3=None, direction=None) -> VertexGroup:
        plane = VertexGroup(v1, v2, v3)
        self.planes.append(plane)
        v1.push_plane(plane)
        v2.push_plane(plane)
        if v3 is not None:
            v3.push_plane(plane)
        if direction is not None:
            plane.set_direction(direction)
            plane.correct_normal()
        return plane

    def make_plane_with_latest(self, direction=None) -> VertexGroup:
        if len(self.vertices) < 3:
            return None
        return self.make_plane(self.vertices[-3], self.vertices[-2], self.vertices[-1], direction)

    def make_line_with_latest(self) -> VertexGroup:
        if len(self.vertices) < 2:
            return None
        return self.make_plane(self.vertices[-2], self.vertices[-1])

    # self.planes : VertexGroup 리스트.
    # VertexGroup : 물체의 면. 3개의 정점의 벡터를 튜플로 가짐.
    # return value : (plane: VertexGroup, collision point: QVector3D)
    def collision_with_ray(self, start, end):
        if not self.collision_check:
            return
        for plane in self.planes:
            if not plane.is_triangle():
                continue
            edge1 = plane[1] - plane[0]
            edge2 = plane[2] - plane[0]
            ray = end - start
            p0_to_start = start - plane[0]

            matrix_a = np.array([
                    [edge1[0], edge2[0], -ray[0]],
                    [edge1[1], edge2[1], -ray[1]],
                    [edge1[2], edge2[2], -ray[2]]])
            matrix_b = np.array([[p0_to_start[0]], [p0_to_start[1]], [p0_to_start[2]]])
            try:
                (u, v, t) = np.linalg.solve(matrix_a, matrix_b)
            except np.linalg.LinAlgError:
                continue

            if t[0] >= 0 and u[0] >= 0 and v[0] >= 0 and u[0] + v[0] <= 1:
                yield plane, start + t[0] * ray
            else:
                continue

    def copy_planes(self, p_set: 'set[VertexGroup]') -> 'set[VertexGroup]':
        new_planes = set()
        pillars = []
        for p in p_set:
            pairs = []
            for v in p:
                new_v = self.append_vertex(v.x(), v.y(), v.z())
                pairs.append((v, new_v))
            pillars.append(pairs)
            new_p = self.make_plane_with_latest(p.get_normal())
            new_planes.add(new_p)

        for pairs in pillars:
            for idx, pair in enumerate(pairs):
                n = idx + 1
                if n >= len(pairs):
                    n = 0
                neighbour1 = pairs[n]

                n = idx - 1
                neighbour2 = pairs[n]
                if not is_inner_line(pair[0], neighbour1[0], p_set):
                    self.make_plane(pair[0], pair[1], neighbour1[0], pair[0] - neighbour2[0])

                if not is_inner_line(pair[0], neighbour2[0], p_set):
                    self.make_plane(pair[0], pair[1], neighbour2[1], pair[0] - neighbour1[0])

        return new_planes

    def draw(self):
        glPolygonMode(*self.polygon_mode)

        glPolygonMode(*self.polygon_mode)

        face_points = dict()
        for p in self.planes:
            f = QVector3D(0, 0, 0)
            for v in p:
                f += v
            face_points[p] = f / len(p)

        '''
        코드 수정 필요
        '''
        new_points = dict()
        edge_points = dict()

        for v in self.vertices:
            edge_avg = QVector3D(0, 0, 0)
            face_avg = QVector3D(0, 0, 0)
            n = len(v.adjacent_plane)
            checklist = set()
            for p in v.adjacent_plane:
                face_avg += face_points[p]
                ops = p.opposite(v)
                for v_op in ops:
                    if (v, v_op) in edge_points:
                        if v_op not in checklist:
                            checklist.add(v_op)
                            edge_avg += edge_points[(v, v_op)]
                    for p2 in v.adjacent_plane:
                        if v in p2 and v_op in p2 and p2 is not p:
                            a = face_points[p]
                            b = face_points[p2]
                            ep = (v + v_op + a + b) / 4
                            edge_points[(v, v_op)] = ep
                            edge_points[(v_op, v)] = ep
                            if v_op not in checklist:
                                checklist.add(v_op)
                                edge_avg += ep
            new_points[v] = (face_avg / 3 + 2 * edge_avg / len(checklist) + (n - 3) * v) / n
        #for i, p1 in enumerate(self.planes):
        #    for p2 in self.planes[i + 1:]:
        #        for x in range(-1, 3):
        #            if is_inner_line(p1[x], p1[x + 1], set([p1, p2])):
        #                a = face_points[p1]
        #                b = face_points[p2]
        #                edge_points[(p1, p2)] = (a + b + p1[x] + p1[x + 1]) / 4
        glBegin(GL_POINTS)
        glColor3f(0, 0, 1)
        for p in face_points:
            fp = face_points[p]
            glVertex3f(fp.x(), fp.y(), fp.z())
        glColor3f(1, 0, 0)
        for e in edge_points:
            ep = edge_points[e]
            glVertex3f(ep.x(), ep.y(), ep.z())
        glColor3f(0, 1, 0)
        for v in new_points:
            nv = new_points[v]
            glVertex3f(nv.x(), nv.y(), nv.z())
        glEnd()

        glBegin(GL_TRIANGLES)
        for v in self.vertices:
            for p1 in v.adjacent_plane:
                for p2 in v.adjacent_plane:
                    inner = get_inner_line(p1, p2)
                    if inner is None:
                        continue
                    nv = new_points[v]
                    mid = edge_points[inner]
                    fp1 = face_points[p1]
                    fp2 = face_points[p2]
                    for point in [nv, mid, fp1, nv, mid, fp2]:
                        glVertex3f(point.x(), point.y(), point.z())
        glEnd()


        lines = []
        glBegin(GL_TRIANGLES)
        glColor3f(1, 1, 1)
        for plane in self.planes:
            if not plane.is_triangle():
                lines.append(plane)
                continue
            for vertex in plane:
                glVertex3f(vertex.x(), vertex.y(), vertex.z())
        glEnd()

        '''
        glBegin(GL_LINES)
        glColor3f(0, 0, 1)
        for plane in self.planes:
            s = QVector3D(0, 0, 0)
            for v in plane:
                s += v
            s = s / len(plane)
            glVertex3f(s.x(), s.y(), s.z())
            n = plane.get_normal()
            n.normalize()
            s += n
            glVertex3f(s.x(), s.y(), s.z())
        glEnd()
        '''


        #glBegin(GL_LINES)
        #for line in lines:
        #    for vertex in line:
        #        glVertex3f(vertex.x(), vertex.y(), vertex.z())
        #glEnd()


class Camera(WorldObject):
    def __init__(self, x=0, y=0, z=1):
        WorldObject.__init__(self, "camera", x, y, z)
        self.dist_from_target = z
        self.zoom_limit = 0.2
        self.zoom_step = 0.1

        self.perspective_mode = "p"
        self.perspective()

        self.angle = 45
        self.ratio = 1
        self.near = 0.1
        self.far = 1000

        self.height = 1

    def update_view(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        target = self.pos + self.direction
        gluLookAt(
            self.pos.x(), self.pos.y(), self.pos.z(),
            target.x(), target.y(), target.z(),
            self.up.x(), self.up.y(), self.up.z())

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if self.perspective_mode == "p":
            gluPerspective(self.angle, self.ratio, self.near, self.far)
        else:
            width = self.ratio * self.height
            glOrtho(-width / 2, width / 2, -self.height, self.height, -self.near, -self.far)

    def set_screen_ratio(self, x, y):
        self.ratio = x / y

    def perspective(self):
        self.perspective_mode = "p"

    def orthogonal(self):
        self.perspective_mode = "o"

    def zoom(self, forward=True):
        if forward:
            new_distance = self.dist_from_target - self.zoom_step
            if new_distance < self.zoom_limit:
                return
            self.dist_from_target = new_distance
            self.move(self.zoom_step, 0, 0)
        else:
            self.dist_from_target += self.zoom_step
            self.move(-self.zoom_step, 0, 0)

    def forward_back(self, callback, degree):
        self.move(self.dist_from_target, 0, 0)
        callback(degree)
        self.move(-self.dist_from_target, 0, 0)

    def rev_up_axis(self, degree):
        self.forward_back(self.rotate_up_axis, degree)

    def rev_left_axis(self, degree):
        self.forward_back(self.rotate_left_axis, degree)



