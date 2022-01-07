import mini3d


def test_mesh_make_plane_no_direction():
    coord_seq = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    mesh = mini3d.Mesh()

    vertex_seq = mesh.append_vertex(coord_seq)
    plane = mesh.make_plane(vertex_seq[0], vertex_seq[1], vertex_seq[2])
    assert mesh.planes_count() == 1
    is_valid_plane(plane, vertex_seq[0], vertex_seq[1], vertex_seq[2])

    assert mesh.is_valid()


def test_mesh_make_plane_with_latest_no_direction():
    #vertex가 부족할 경우 Mesh.make_plane_with_latest가 실패하는 것 테스트
    coord_seq = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    mesh = mini3d.Mesh()

    vertex_seq = mesh.append_vertex(coord_seq)
    plane = mesh.make_plane_with_latest()
    assert mesh.planes_count() == 0
    assert plane is None
    assert mesh.is_valid()

    #Mesh.make_plane_with_latest가 정상작동하는지 테스트
    coord_seq = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15]
    ]
    mesh = mini3d.Mesh()

    vertex_seq = mesh.append_vertex(coord_seq)
    plane = mesh.make_plane_with_latest()
    assert mesh.planes_count() == 1
    is_valid_plane(plane, vertex_seq[2], vertex_seq[3], vertex_seq[4])
    assert mesh.is_valid()


def test_mesh_delete_plane():
    coord_seq = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15]
    ]
    mesh = mini3d.Mesh()

    vertex_seq = mesh.append_vertex(coord_seq)
    plane1 = mesh.make_plane(vertex_seq[0], vertex_seq[1], vertex_seq[2])
    plane2 = mesh.make_plane_with_latest()
    mesh.delete_plane(plane1)

    assert mesh.planes_count() == 1
    assert mesh.is_valid()


def is_valid_plane(plane: mini3d.VertexGroup, v1: mini3d.MeshVertex, v2: mini3d.MeshVertex, v3: mini3d.MeshVertex):
    opposite_seq = plane.opposite(v1)
    assert plane.is_triangle()
    assert opposite_seq[0] is v2 or opposite_seq[0] is v3
    assert opposite_seq[1] is v2 or opposite_seq[1] is v3
