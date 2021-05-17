'''
QVector를 numpy array로 교체
Mesh에 선분 데이터 추가로 저장
VertexGroup 노멀벡터가 최신인지 표시하는 플래그 추가 및 정점 좌표 변경시 업데이트 되도록 수정
추가적인 충돌체크 함수 제작, Mesh.collision_with_ray의 충돌체크 코드 함수로 따로 제작

vertexgroup copy method 제작
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


def catmull_clark(mesh: 'Mesh') -> 'Mesh':
    if mesh.catmull_clark_level >= 3:
        return mesh
    new_mesh = Mesh()
    new_mesh.catmull_clark_level = mesh.catmull_clark_level + 1

    face_points = dict()
    for mesh_plane in mesh.planes:
        f = QVector3D(0, 0, 0)
        for mesh_vertex in mesh_plane:
            f += mesh_vertex
        f /= len(mesh_plane)
        face_points[mesh_plane] = new_mesh.append_vertex(f.x(), f.y(), f.z())

    new_points = dict()
    edge_points = dict()
    for mesh_vertex in mesh.vertices:
        edge_avg = QVector3D(0, 0, 0)  # average point of adjacent edges with mesh_vertex
        face_avg = QVector3D(0, 0, 0)  # average point of adjacent planes with mesh_vertex
        n = len(mesh_vertex.adjacent_plane)
        opposites = dict()  # key : opposite vertex, value : set of adjacent planes with the edge
        for mesh_plane in mesh_vertex.adjacent_plane:
            face_avg += face_points[mesh_plane]
            op = mesh_plane.opposite(mesh_vertex)
            for op_v in op:
                if op_v in opposites:
                    opposites[op_v].add(mesh_plane)
                else:
                    opposites[op_v] = set([mesh_plane])
        face_avg /= len(mesh_vertex.adjacent_plane)

        for op_v in opposites:
            if (mesh_vertex, op_v) in edge_points:
                s = edge_points[(mesh_vertex, op_v)]
                edge_avg += s
                continue
            else:
                s = mesh_vertex + op_v
                for mesh_plane in opposites[op_v]:
                    s += face_points[mesh_plane]
                s /= 2 + len(opposites[op_v])
            edge_avg += s

            nv = new_mesh.append_vertex(s.x(), s.y(), s.z())
            edge_points[(mesh_vertex, op_v)] = nv
            edge_points[(op_v, mesh_vertex)] = nv

        edge_avg /= len(opposites)

        if n >= 3:
            nv = (face_avg + 2 * edge_avg + (n - 3) * mesh_vertex) / n
        elif n > 0:
            nv = (face_avg + edge_avg + mesh_vertex) / 3
        else:
            nv = mesh_vertex
        new_points[mesh_vertex] = new_mesh.append_vertex(nv.x(), nv.y(), nv.z())

    for mesh_plane in mesh.planes:
        f = face_points[mesh_plane]
        for mesh_vertex in mesh_plane:
            op = mesh_plane.opposite(mesh_vertex)

            v1 = new_points[mesh_vertex]
            e1 = edge_points[(mesh_vertex, op[0])]
            e2 = edge_points[(mesh_vertex, op[1])]

            for v in (v1, f):
                new_plane = new_mesh.make_plane(v, e1, e2, mesh_plane.get_normal())
                new_plane.copy_attr_of(mesh_plane)
    return new_mesh


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
        self.color = (255, 255, 255)
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

    def inverse_direction(self):
        if self.type == VertexGroup.TRIANGLE:
            self.direction = -self.direction

    def get_normal(self):
        return cross_product(self.group[1] - self.group[0], self.group[2] - self.group[0])

    def set_direction(self, direction):
        self.direction = direction

    def correct_normal(self):
        normal = self.get_normal()
        if inner_product(normal, self.direction) < 0:
            self.inverse_normal()

    def correct_direction(self):
        self.set_direction(self.get_normal())

    def copy_attr_of(self, plane: 'VertexGroup'):
        """
        copy some attributes of plane like color. not group
        :param plane: original
        :return: None
        """
        self.color = plane.color
        pass

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
        for (i, p) in enumerate(self.adjacent_plane):
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
        self.catmull_clark_level = 0
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

    def delete_plane(self, plane):
        for (i, p) in enumerate(self.planes):
            if p is plane:
                self.planes[i], self.planes[-1] = self.planes[-1], self.planes[i]
                self.planes.pop()

                for v in plane:
                    v.pop_plane(plane)

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
        side_planes = set()
        cover_planes = set()
        pillars = dict()
        v_set = set()

        for p in p_set:
            for v in p:
                pillars[v] = None

        for v in pillars:
            pillars[v] = self.append_vertex(v.x(), v.y(), v.z())

        for p in p_set:
            nvs = [pillars[v] for v in p]
            cap = self.make_plane(*nvs, p.get_normal())
            cap.copy_attr_of(p)
            cover_planes.add(cap)
            for x in range(-1, 2):
                ov = p[x]
                op = pillars[ov]
                n_ov = p[x + 1]
                n_op = pillars[n_ov]
                if is_inner_line(ov, n_ov, p_set):
                    continue
                for vertices in [[ov, op, n_ov], [n_ov, n_op, op]]:
                    new_plane = self.make_plane(vertices[0], vertices[1], vertices[2], vertices[0] - p[x - 1])
                    new_plane.copy_attr_of(p)
                    side_planes.add(new_plane)
        return cover_planes, side_planes

    def draw(self):
        glPolygonMode(*self.polygon_mode)

        glLineWidth(1)
        lines = []
        glBegin(GL_TRIANGLES)
        glColor3f(1, 1, 1)
        for plane in self.planes:
            if not plane.is_triangle():
                lines.append(plane)
                continue
            glColor3f(plane.color[0] / 255, plane.color[1] / 255, plane.color[2] / 255)
            for vertex in plane:
                n = plane.get_normal()
                n.normalize()
                glNormal3f(n.x(), n.y(), n.z())
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
            glColor3f(0, 0, 1)
            glVertex3f(s.x(), s.y(), s.z())
        glEnd()
        '''



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
            glOrtho(-width, width, -self.height, self.height, -self.near, self.far)

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
