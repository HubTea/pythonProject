import mini3d


#TODO

#Mesh객체의 planes에 원소가 있는 경우 Mesh.delete_vertex() 테스트

#Mesh.delete_plane() 테스트

#Mesh.make_plane() 테스트

#Mesh.make_plane_with_latest() 테스트


def test_mesh_insert_vertices_in_empty_mesh_return_corresponding_mesh_vertex_list():
    insert_vertices_in_mesh([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], mini3d.Mesh())
    insert_vertices_in_mesh([], mini3d.Mesh())
    insert_vertices_in_mesh([[1, 2, 3]], mini3d.Mesh())


def test_mesh_insert_vertices_in_nonempty_mesh_return_corresponding_mesh_vertex_list():
    mesh = mini3d.Mesh()
    mesh.append_vertex([(1, 2, 3)])

    insert_vertices_in_mesh([
        [10, 11, 12],
        [20, 21, 22]
    ], mesh)
    insert_vertices_in_mesh([], mesh)
    insert_vertices_in_mesh([[1, 2, 3]], mesh)


def test_vertex_deletion_on_mesh_without_plane():
    coord_seq = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]

    mesh = mini3d.Mesh()
    mesh_vertex_list = mesh.append_vertex(coord_seq)

    mesh.delete_vertex(mesh_vertex_list[1])
    assert is_equal_coord(mesh.get_coord(0), coord_seq[0])
    assert is_equal_coord(mesh.get_coord(1), coord_seq[2])

    assert mesh.vertices_count() == 2
    assert mesh.is_valid()


def is_equal_coord(coord1: 'list[x, y, z]', coord2: 'list[x, y, z]') -> bool:
    for index in range(0, 3):
        if coord1[index] != coord2[index]:
            return False
    return True


def insert_vertices_in_mesh(coord_seq: 'list[list[x, y, z], ...]', mesh: 'Mesh'):
    new_mesh_vertex_seq = mesh.append_vertex(coord_seq)

    assert len(new_mesh_vertex_seq) == len(coord_seq)

    for index in range(0, len(coord_seq)):
        mesh_vertex_coord = new_mesh_vertex_seq[index].get_coord()
        assert is_equal_coord(mesh_vertex_coord, coord_seq[index])

    assert mesh.is_valid()