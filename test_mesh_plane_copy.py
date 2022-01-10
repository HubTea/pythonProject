import mini3d
import prepared_mesh


def test_mesh_copy_plane_connection():
    mesh = prepared_mesh.mesh_with_eight_planes()
    plane_set = set(mesh.planes)
    mesh.copy_planes(plane_set)
    assert mesh.planes_count() == 8 + 8 + 8 * 2

    mesh = prepared_mesh.mesh_with_eight_planes()
    plane_set = set(mesh.planes)
    mesh.copy_planes(plane_set, False)
    assert mesh.planes_count() == 8 + 8
