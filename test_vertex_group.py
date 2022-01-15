import mini3d
import numpy as np


def test_vertex_group_normal_vector_calculation():
    mesh = mini3d.Mesh()

    coord_seq = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    vertex_seq = mesh.append_vertex(coord_seq)
    plane = mesh.make_plane(*vertex_seq)

    vector_seq = [np.array(coord) for coord in coord_seq]
    edge1 = vector_seq[1] - vector_seq[0]
    edge2 = vector_seq[2] - vector_seq[0]
    assert np.abs(plane.calc_normal() - np.cross(edge1, edge2)).sum() < 0.000001


def test_vertex_group_returns_opposite_vertices():
    mesh = mini3d.Mesh()

    coord_seq = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    vertex_seq = mesh.append_vertex(coord_seq)
    plane = mesh.make_plane(*vertex_seq)

    vertex_pair = plane.opposite(vertex_seq[0])
    assert vertex_pair[0] in vertex_seq[1:3]
    assert vertex_pair[1] in vertex_seq[1:3]


def test_vertex_group_returns_nearest_vertex_from_a_point():
    mesh = mini3d.Mesh()

    coord_seq = [
        [0, 0, 0],
        [10, 0, 0],
        [0, 10, 0]
    ]

    vertex_seq = mesh.append_vertex(coord_seq)
    plane = mesh.make_plane(*vertex_seq)

    #[coord, answer]
    case_seq = [
        [[0, 0, 0], vertex_seq[0]],
        [[10, 0, 0], vertex_seq[1]],
        [[0, 10, 0], vertex_seq[2]],
        [[4, 6, 0], vertex_seq[2]],
        [[6, 4, 0], vertex_seq[1]]
    ]

    for case in case_seq:
        assert plane.nearest_vertex(np.array(case[0])) is case[1]


def test_vertex_group_returns_nearest_edge_from_a_point():
    mesh = mini3d.Mesh()

    coord_seq = [
        [0, 0, 0],
        [10, 0, 0],
        [0, 10, 0]
    ]

    vertex_seq = mesh.append_vertex(coord_seq)
    plane = mesh.make_plane(*vertex_seq)

    # [coord, answer]
    case_seq = [
        [[1, 3, 0], [vertex_seq[0], vertex_seq[2]]],
        [[3, 1, 0], [vertex_seq[0], vertex_seq[1]]],
        [[4.5, 4.5, 0], [vertex_seq[1], vertex_seq[2]]],
    ]

    for case in case_seq:
        edge = plane.nearest_line(case[0])
        assert edge[0] in case[1] and edge[1] in case[1]
