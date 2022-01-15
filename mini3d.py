
from PyQt5.QtGui import QVector3D
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

pi_div_180 = np.pi / 180


#TODO
#Review, test, modify global functions

def inner_product(a: QVector3D, b: QVector3D) -> float:
    product = a.toVector4D() * b.toVector4D()
    return product[0] + product[1] + product[2] + product[3]


def cross_product(a: QVector3D, b: QVector3D) -> QVector3D:
    x = a[1] * b[2] - b[1] * a[2]
    y = -(a[0] * b[2] - b[0] * a[2])
    z = a[0] * b[1] - b[0] * a[1]
    return QVector3D(x, y, z)


def triple_product(a: QVector3D, b: QVector3D, c: QVector3D) -> float:
    return inner_product(cross_product(a, b), c)


def normal_vector(org: QVector3D, v1: QVector3D, v2: QVector3D) -> QVector3D:
    return cross_product(v1 - org, v2 - org)


def rotate(vector_as_x: QVector3D, vector_as_y: QVector3D, degree) -> 'tuple[QVector3D, QVector3D]':
    """
    vector_as_x와 vector_as_y를 포함하는 평면 상에서
    두 벡터를 degree 만큼 회전시킨 벡터의 튜플을 반환
    degree 가 0보다 클 때, 회전 후의 vector_as_x와 회전 이전의 vector_as_y 사이의 각도가 줄어들도록
    vector_as_x 와 vector_as_y 를 설정해야 함
    """

    c = np.cos(pi_div_180 * degree)
    s = np.sin(pi_div_180 * degree)
    temp_y_axis = c * vector_as_y - s * vector_as_x
    temp_x_axis = c * vector_as_x + s * vector_as_y
    return temp_x_axis, temp_y_axis


def ray_to_plane(start: QVector3D, end: QVector3D, normal: QVector3D, point: QVector3D):
    """
    평면과 선분의 교차 여부 검사. 충돌했으면 충돌 좌표의 QVector3D객체 반환. 아니면 None 반환
    
    start: 선분의 시작 점
    end: 선분의 끝 점
    normal: 평면의 노멀 벡터
    point: 평면 위의 한 점
    """

    t = inner_product(normal, point - start) / inner_product(normal, end - start)
    if 0 <= t <= 1:
        return start + t * (end - start)


def is_inner_line(v1: 'MeshVertex', v2: 'MeshVertex', p_set: 'set[VertexGroup]') -> bool:
    """두 정점 v1과 v2로 이루어진 선분이 면의 집합인 p_set 에서 내부에 위치했으면 True 반환. 아니며 False"""
    count = 0
    for p in v1.adjacent_plane:
        if p in p_set and v2 in p:
            count += 1
    return count > 1


def get_inner_line(p1: 'VertexGroup', p2: 'VertexGroup') -> 'tuple[MeshVertex, MeshVertex]':
    """두 면에 공통으로 존재하는 선분을 MeshVertex 의 튜플로 반환"""
    count = 0
    intersection = []
    for v1 in p1:
        if v1 in p2:
            count += 1
            intersection.append(v1)
    if count == 2:
        return tuple(intersection)


def catmull_clark(mesh: 'Mesh') -> 'Mesh':
    """mesh 에서 catmull clark subdivision 을 한 단계 적용한 새 Mesh 객체 반환"""
    if mesh.catmull_clark_level >= 3:  # 최대 단계 제한
        return mesh
    new_mesh = Mesh()
    new_mesh.catmull_clark_level = mesh.catmull_clark_level + 1
    
    # subdivision 으로 인해 생성되는 정점들의 좌표 계산
    #
    # face_points: dictionary. Key 는 VertexGroup, Value 는 VertexGroup 에 포함된 정점의 좌표의 평균 F
    # edge_points: dictionary. Key 는 MeshVertex 두 개로 이루어진 선분, Value 는 선분의 양 끝 점과 인접한 면의 F 점들의 평균 E
    # new_points: dictionary. Key 는 MeshVertex, Value 는 새로 옮겨질 정점의 좌표
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
        edge_avg = QVector3D(0, 0, 0)  # mesh_vertex 에 인접한 선분의 R 점들의 평균
        face_avg = QVector3D(0, 0, 0)  # mesh_vertex 에 인접한 면의 F 점들의 평균
        n = len(mesh_vertex.adjacent_plane)
        opposites = dict()  # key : 반대편 정점, value : mesh_vertex 와 key 를 양 끝으로 하는 선분에 인접한 면의 집합
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
    
    # 위에서 만들어진 정점들을 연결하여 면 구성 및 Mesh 생성
    for mesh_plane in mesh.planes:
        f = face_points[mesh_plane]
        for mesh_vertex in mesh_plane:
            op = mesh_plane.opposite(mesh_vertex)

            v1 = new_points[mesh_vertex]
            e1 = edge_points[(mesh_vertex, op[0])]
            e2 = edge_points[(mesh_vertex, op[1])]

            for v in (v1, f):
                new_plane = new_mesh.make_plane(v, e1, e2, mesh_plane.calc_normal())
                new_plane.copy_attr_of(mesh_plane)
    return new_mesh


class WorldObject:
    """
    3차원 공간 상에 존재하는 오브젝트를 정의함

    pos: 오브젝트의 좌표
    direction: 오브젝트 기준으로 정면이 어느 방향인지 나타내는 벡터
    up: 오브젝트 기준으로 위쪽이 어느 방향인지 나타내는 벡터
    left: 오브젝트 기준으로 왼쪽이 어느 방향인지 나타내는 벡터
    """
    
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
        """오브젝트의 위치를 forward 방향, up 방향, left 방향으로 각각 주어진 수치만큼 옮김"""
        self.pos += left * self.left + up * self.up + forward * self.direction

    def rotate_direction_axis(self, degree):
        """오브젝트를 direction 벡터를 회전축으로 삼아서 회전시킴"""
        self.up, self.left = rotate(self.up, self.left, degree)

    def rotate_up_axis(self, degree):
        """오브젝트를 up 벡터를 회전축으로 삼아서 회전시킴"""
        self.left, self.direction = rotate(self.left, self.direction, degree)

    def rotate_left_axis(self, degree):
        """오브젝트를 left 벡터를 회전축으로 삼아서 회전시킴"""
        self.direction, self.up = rotate(self.direction, self.up, degree)

    def draw(self):
        """렌더링"""
        pass


#TODO
#Review and test VertexGroup
class VertexGroup:
    """
    3D 물체에서 면을 정의함
    VertexGroup 객체는 생성 후 생성한 쪽에서 correct_normal()을 호출해야 함

    color: 면의 색상. RGB
    owner: 이 면을 소유한 Mesh 객체
    group: 이 면이 포함하고 있는 정점들의 튜플
    direction:
        이 면의 노멀 벡터의 방향을 지시하는 벡터
        노멀벡터와 같지는 않으나 내적을 했을 때 양수가 나옴
    """

    def __init__(self, v1: 'MeshVertex', v2: 'MeshVertex', v3: 'MeshVertex'):
        self.color = (255, 255, 255)
        self.owner = None
        self.group = (v1, v2, v3)
        self.direction = np.array([1, 0, 0])

    def nearest_vertex(self, org: 'np.array') -> 'MeshVertex':
        """면에 포함된 정점 중 org 와의 거리가 가장 짧은 정점 반환"""
        li = []
        for v in self.group:
            dist_quad = (v.get_coord() - org) ** 2
            li.append((dist_quad.sum(), v))
        li.sort(key=lambda x: x[0])
        return li[0][1]

    def nearest_line(self, org: 'np.array') -> 'tuple[MeshVertex, MeshVertex]':
        """ 면에 포함된 정점들로 구성되는 선분들 중에서 org 와의 거리가 가장 짧은 선분의 양 끝 정점을 튜플로 반환"""
        li = []
        for start, end in ((0, 1), (1, 2), (2, 0)):
            v1 = self.group[start]
            v2 = self.group[end]
            direction = (v1 - v2)
            line = org - v2.get_coord()
            n = np.inner(direction, line) / np.inner(direction, direction) * direction - line
            dist_quad = n ** 2
            li.append((dist_quad.sum(), v1, v2))
        li.sort(key=lambda x: x[0])
        return li[0][1], li[0][2]

    def opposite(self, org: 'MeshVertex') -> 'tuple[MeshVertex, MeshVertex]':
        """면에 포함된 정점 중에서 org 를 제외한 나머지 정점들의 튜플 반환"""
        op = []
        for v in self.group:
            if v is not org:
                op.append(v)
        return tuple(op)

    def inverse_normal(self):
        """self.group 의 정점들의 순서를 바꿈으로써 면의 노멀 벡터의 방향을 뒤집음"""
        self.group = (self.group[0], self.group[2], self.group[1])

    def inverse_direction(self):
        """self.direction 의 방향을 뒤집음"""
        self.direction = -self.direction

    def calc_normal(self) -> 'np.array':
        """이 객체의 실제 normal vector를 반환함"""
        return np.cross(self.group[1] - self.group[0], self.group[2] - self.group[0])

    def set_direction(self, direction):
        self.direction = direction

    def correct_normal(self):
        """self.direction 과 면의 노멀 벡터의 내적이 양수가 되도록 노멀 벡터의 방향을 조정함"""
        normal = self.calc_normal()
        if np.inner(normal, self.direction) < 0:
            self.inverse_normal()

    def correct_direction(self):
        """self.direction 과 면의 노멀 벡터의 내적이 양수가 되도록 self.direction 의 방향을 조정함"""
        self.set_direction(self.calc_normal())

    def copy_attr_of(self, plane: 'VertexGroup'):
        """
        self 의 일부 attribute 를 plane 에 복사함
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


class MeshVertex:
    """
    3D 물체에서 정점을 정의함

    adjacent_plane: 정점에 인접한 VertexGroup 의 리스트
    """

    def __init__(self, owner, vertex_id):
        self.owner = owner
        self.vertex_id = vertex_id
        self.adjacent_plane = []

    def set_id(self, vertex_id):
        self.vertex_id = vertex_id

    def push_plane(self, plane):
        self.adjacent_plane.append(plane)

    def pop_plane(self, plane):
        """adjacent_plane 에서 plane 제거"""
        for (i, p) in enumerate(self.adjacent_plane):
            if p is plane:
                self.adjacent_plane[i], self.adjacent_plane[-1] = self.adjacent_plane[-1], self.adjacent_plane[i]
                self.adjacent_plane.pop()
                break

    def get_coord(self) -> 'np.array[x, y, z]':
        return self.owner.get_coord(self.vertex_id)

    def __hash__(self):
        return id(self)

    def __sub__(self, other: 'MeshVertex') -> np.ndarray:
        return self.get_coord() - other.get_coord()


#TODO
#Test Mesh.copy_planes() after modify VertexGroup class
class Mesh(WorldObject):
    """
    3D 물체를 정의함

    vertices: 물체를 구성하는 MeshVertex 의 리스트
    vertex_coordinates: 4행 n열 행렬. 정점들의 좌표 저장
    planes: 물체를 구성하는 VertexGroup 의 리스트
    collision_check: 충돌 검사를 실행할 지 결정
    catmull_clark_level: subdivision 의 현재 단계를 나타냄
    """
    
    def __init__(self):
        WorldObject.__init__(self)
        self.vertices = []
        self.vertex_coordinates = np.ndarray(shape=(4, 0), dtype=np.float)
        self.planes = []

        self.collision_check = True
        self.catmull_clark_level = 0
        self.polygon_mode = (GL_FRONT_AND_BACK, GL_LINE)
        return

    def is_valid(self):
        if len(self.vertices) != self.vertex_coordinates.shape[1]:
            return False
        for (i, v) in enumerate(self.vertices):
            for c in range(3):
                if v.get_coord()[c] != self.vertex_coordinates[:, i][c]:
                    return False
        return True

    def vertices_count(self):
        return self.vertex_coordinates.shape[1]

    def planes_count(self):
        return len(self.planes)

    def planes_count(self):
        return len(self.planes)

    def get_coord(self, index: int) -> 'np.array[x, y, z]':
        return self.vertex_coordinates[:3, index]

    def append_vertex(self, coordinates: 'list[list, ...]') -> 'list[MeshVertex, ...]':
        """
        주어진 좌표의 정점 추가

        coordinates: 길이 3인 시퀀스의 시퀀스

        return: 새로 생성된 MeshVertex 객체의 리스트
        """
        current_len = self.vertices_count()

        expanded_coord = [self.vertex_coordinates]
        for coord in coordinates:
            expanded_coord.append((coord[0], coord[1], coord[2], 1))

        self.vertex_coordinates = np.column_stack(expanded_coord)

        for vertex_id in range(current_len, self.vertices_count()):
            self.vertices.append(MeshVertex(self, vertex_id))
        return self.vertices[current_len:self.vertices_count()]

    def delete_vertex(self, vertex: MeshVertex):
        """정점 제거. 정점에 인접한 면의 제거도 같이 실행"""
        for (i, v) in enumerate(self.vertices):
            if v is vertex:
                self.vertices[i], self.vertices[-1] = self.vertices[-1], self.vertices[i]
                self.vertices[i].set_id(i)
                self.vertices.pop()

                last = self.vertices_count() - 1
                self.vertex_coordinates[:, i] = self.vertex_coordinates[:, last]
                self.vertex_coordinates = self.vertex_coordinates[:, :last]

                for plane in v.adjacent_plane:
                    for adj_vertex in plane:
                        if adj_vertex is not v:
                            adj_vertex.pop_plane(plane)

                adj_plane_seq = v.adjacent_plane[:]
                for plane in adj_plane_seq:
                    v.pop_plane(plane)
                    idx = self.planes.index(plane)
                    self.planes[idx], self.planes[-1] = self.planes[-1], self.planes[idx]
                    self.planes.pop()

    def delete_plane(self, plane: VertexGroup):
        """
        Mesh.planes에서 plane에 해당하는 원소 제거.
        plane에 포함된 MeshVertex의 pop_plane 실행
        """
        for (i, p) in enumerate(self.planes):
            if p is plane:
                self.planes[i], self.planes[-1] = self.planes[-1], self.planes[i]
                self.planes.pop()

                for v in plane:
                    v.pop_plane(plane)

    def make_plane(self, v1: MeshVertex, v2: MeshVertex, v3: MeshVertex, direction=None) -> VertexGroup:
        """v1, v2, v3 로 구성되는 면 생성."""
        plane = VertexGroup(v1, v2, v3)
        plane.owner = self

        self.planes.append(plane)
        v1.push_plane(plane)
        v2.push_plane(plane)
        v3.push_plane(plane)

        if direction is not None:
            plane.set_direction(direction)
            plane.correct_normal()
        return plane

    def make_plane_with_latest(self, direction=None) -> VertexGroup:
        """
        self.vertiece 의 마지막 세 정점으로 면 생성
        정점이 부족할 경우 None 반환
        """
        if len(self.vertices) < 3:
            return None
        return self.make_plane(self.vertices[-3], self.vertices[-2], self.vertices[-1], direction)

    def collision_with_ray(self, start: 'list[x, y, z]', end: 'list[x, y, z]') -> 'tuple[VertexGroup, np.array]':
        """
        Mesh에 포함된 면들과 start와 end를 시작과 끝으로 하는 선분과의 충돌을 검사하는 generator
        충돌한 면과 좌표로 구성된 튜플 반환
        반환되는 순서는 충돌점과 start 사이의 거리와는 관련이 없음
        """
        if not self.collision_check:
            return

        start_vector = np.array([start[0], start[1], start[2]])
        end_vector = np.array([end[0], end[1], end[2]])
        ray = end_vector - start_vector
        for plane in self.planes:
            edge1 = plane[1] - plane[0]
            edge2 = plane[2] - plane[0]

            p0_to_start = start - plane[0].get_coord()

            matrix_a = np.array([edge1, edge2, -ray]).transpose()
            matrix_b = p0_to_start.transpose()
            try:
                (u, v, t) = np.linalg.solve(matrix_a, matrix_b)
            except np.linalg.LinAlgError:
                continue

            if t >= 0 and u >= 0 and v >= 0 and u + v <= 1:
                yield plane, start_vector + t * ray
            else:
                continue

    def copy_planes(self, p_set: 'set[VertexGroup]', connect=True) -> 'set[VertexGroup]':
        """
        p_set 으로 주어진 면들을 복사함

        connect: True 이면 원본과 복제본의 사이에 면을 생성하여 연결하여 기둥 형태로 만듬. False 이면 사이에 면을 생성하지 않음.
        """

        side_planes = set()
        cover_planes = set()
        pillars = dict()
        v_set = set()

        for p in p_set:
            for v in p:
                pillars[v] = None

        #위의 반복문과 합치면 안 됨.
        #MeshVertex가 중복될 수 있기 때문.
        for v in pillars:
            coord = v.get_coord()
            pillars[v] = self.append_vertex([
                [coord[0], coord[1], coord[2]]
            ])[0]

        for p in p_set:
            new_vertices = [pillars[v] for v in p]
            cap = self.make_plane(*new_vertices, p.calc_normal())
            cap.copy_attr_of(p)
            cover_planes.add(cap)

            if not connect:
                continue

            for vertex_index in range(-1, 2):
                original_vertex = p[vertex_index]
                next_original_vertex = p[vertex_index + 1]

                if is_inner_line(original_vertex, next_original_vertex, p_set):
                    continue

                opposite = pillars[original_vertex]
                next_opposite = pillars[next_original_vertex]

                for vertices in [
                    [original_vertex, opposite, next_original_vertex],
                    [next_original_vertex, next_opposite, opposite]
                ]:
                    new_plane = self.make_plane(
                        vertices[0], vertices[1], vertices[2], vertices[0] - p[vertex_index - 1]
                    )
                    new_plane.copy_attr_of(p)
                    side_planes.add(new_plane)
        return cover_planes, side_planes

    def set_polygon_mode(self, mode):
        self.polygon_mode = (GL_FRONT_AND_BACK, mode)

    def draw(self):
        """화면에 3D 물체를 출력함"""
        glPolygonMode(*self.polygon_mode)
        glLineWidth(1)

        glBegin(GL_TRIANGLES)
        glColor3f(1, 1, 1)
        for plane in self.planes:
            if plane.is_triangle():
                glColor3f(plane.color[0] / 255, plane.color[1] / 255, plane.color[2] / 255)
                for vertex in plane:
                    n = plane.calc_normal()
                    n.normalize()
                    glNormal3f(n.x(), n.y(), n.z())
                    glVertex3f(vertex.x(), vertex.y(), vertex.z())
        glEnd()

    def save(self, path):
        """Mesh 를 파일로 저장"""
        vertex_label = dict()
        with open(path, 'w') as file:
            file.write('v\n')
            for i, v in enumerate(self.vertices):
                vertex_label[v] = i
                file.write('{0} {1} {2} {3}\n'.format(i, v.x(), v.y(), v.z()))
            file.write('p\n')
            for p in self.planes:
                for v in p:
                    file.write(str(vertex_label[v]) + ' ')
                for c in p.color:
                    file.write(str(c) + ' ')
                file.write('\n')

    def load(self, path):
        """파일로부터 Mesh 를 생성함"""
        try:
            with open(path, 'r') as file:
                vertex_label = dict()
                self.vertices.clear()
                self.planes.clear()

                content = file.readline()
                if 'v' in content:
                    content = file.readline()
                    while 'p' not in content:
                        s = content.split(' ')
                        v = self.append_vertex(*[float(c) for c in s[1:4]])
                        vertex_label[s[0]] = v
                        content = file.readline()

                if 'p' in content:
                    content = file.readline()
                    while content != '':
                        s = content.split(' ')
                        p = self.make_plane(vertex_label[s[0]], vertex_label[s[1]], vertex_label[s[2]])
                        p.correct_direction()

                        p.color = tuple([int(c) for c in s[3:6]])
                        content = file.readline()
        except FileNotFoundError:
            pass


class Camera(WorldObject):
    """
    사용자가 물체를 바라보는 위치와 각도를 관리함.
    투영 변환에 관여함.
    """

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
        """gluLootAt 으로 modelview 행렬 설정. projection 행렬 설정"""
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
        """화면 가로 세로 비율 설정"""
        self.ratio = x / y

    def perspective(self):
        """원근 투영 사용"""
        self.perspective_mode = "p"

    def orthogonal(self):
        """직교 투영 사용"""
        self.perspective_mode = "o"

    def zoom(self, forward=True):
        """
        카메라의 위치에서 self.direction 방향으로
        self.dist_from_target 만큼 떨어진 점에서
        가까워지거나 멀어짐

        forward: True 이면 가까워짐, False 이면 멀어짐
        """

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
        """
        카마라가 self.direction 뱡향으로 self.dist_from_target만큼 떨어진 점을 중심으로 공전함.
        카메라를 self.direction 방향으로 self.dist_from_target 만큼 이동한 뒤
        callback 함수 호출.
        이후 -self.direction 방향으로 self.dist_from_target 만큼 이동
        
        callback: 호출할 함수
        degree: 공전 각도
        """
        
        self.move(self.dist_from_target, 0, 0)
        callback(degree)
        self.move(-self.dist_from_target, 0, 0)

    def rev_up_axis(self, degree):
        """self.up 을 회전축으로 삼아 공전"""
        self.forward_back(self.rotate_up_axis, degree)

    def rev_left_axis(self, degree):
        """self.left 를 회전축으로 삼아 공전"""
        self.forward_back(self.rotate_left_axis, degree)
