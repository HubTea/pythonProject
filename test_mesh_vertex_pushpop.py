import mini3d


def test_vertices_insertion():
    vertex_insertion([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])


def test_vertex_deletion():
    pass


def is_equal_coord(coord1: 'list[x, y, z]', coord2: 'list[x, y, z]') -> bool:
    for index in range(0, 3):
        if coord1[index] != coord2[index]:
            return False
    return True


def vertex_insertion(coord_seq: 'list[list[x, y, z], ...]'):
    mesh = mini3d.Mesh()

    #insert vertices into empty Mesh object
    new_mesh_vertex_seq = mesh.append_vertex(coord_seq)

    assert len(new_mesh_vertex_seq) == len(coord_seq)
    assert mesh.vertices_count() == len(coord_seq)

    for index in range(0, len(coord_seq)):
        mesh_vertex_coord = new_mesh_vertex_seq[index].get_coord()
        assert is_equal_coord(mesh_vertex_coord, coord_seq[index])

        vertex_coord = mesh.get_coord(index)
        assert is_equal_coord(vertex_coord, coord_seq[index])

    #insert vertices into non-empty Mesh object
    prev_len = mesh.vertices_count()
    new_mesh_vertex_seq = mesh.append_vertex(coord_seq)
    assert len(new_mesh_vertex_seq) == len(coord_seq)
    assert mesh.vertices_count() == prev_len + len(coord_seq)

    for index in range(0, len(coord_seq)):
        mesh_vertex_coord = new_mesh_vertex_seq[index].get_coord()
        assert is_equal_coord(mesh_vertex_coord, coord_seq[index])

        vertex_coord = mesh.get_coord(prev_len + index)
        assert is_equal_coord(vertex_coord, coord_seq[index])
