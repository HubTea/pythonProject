import mini3d


def mesh_with_eight_planes():
    coord_seq = [
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [2, 1, 0],
        [0, 2, 0],
        [1, 2, 0],
        [2, 2, 0]
    ]

    mesh = mini3d.Mesh()

    vertex_seq = mesh.append_vertex(coord_seq)
    mesh.make_plane(vertex_seq[0], vertex_seq[1], vertex_seq[3])
    mesh.make_plane(vertex_seq[1], vertex_seq[4], vertex_seq[3])
    mesh.make_plane(vertex_seq[1], vertex_seq[4], vertex_seq[5])
    mesh.make_plane(vertex_seq[1], vertex_seq[2], vertex_seq[5])
    mesh.make_plane(vertex_seq[5], vertex_seq[8], vertex_seq[7])
    mesh.make_plane(vertex_seq[4], vertex_seq[5], vertex_seq[7])
    mesh.make_plane(vertex_seq[3], vertex_seq[4], vertex_seq[7])
    mesh.make_plane(vertex_seq[3], vertex_seq[7], vertex_seq[6])

    # 위 코드에 오류 없는지 확인
    assert mesh.planes_count() == 8 and mesh.is_valid()
    return mesh



