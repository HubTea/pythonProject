import mini3d
import numpy as np


def test_mesh_ray_collision():
    coord_seq = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    mesh = mini3d.Mesh()

    vertex_seq = mesh.append_vertex(coord_seq)
    mesh.make_plane(vertex_seq[0], vertex_seq[1], vertex_seq[2])
    mesh.make_plane(vertex_seq[0], vertex_seq[3], vertex_seq[2])

    #[[ray_start_coord, ray_end_coord, hit, max_hit_count, approximate_hit_point], ]
    #approximate_hit_point가 None인 경우는 hit point가 정확한지 테스트하지 않음
    ray_seq = [
        [[-1, 0, -10], [-1, 0, 10], False, 0, None],
        [[-0.01, 0, -10], [-0.01, 0, 10], False, 0, None],
        [[0.01, 0.01, -10], [0.01, 0.01, 10], True, 1, [0.01, 0.01, 0]],
        [[0.45, 0, -10], [0.45, 0, 10], True, 1, [0.45, 0.45, 0]],
        [[-10, 0.45, 0.45], [10, 0.45, 0.45], True, 1, [0, 0.45, 0.45]],
        [[-0.5, 0.5, 1], [1, 0.5, -0.5], True, 2, None]
    ]

    for ray in ray_seq:
        hit_count = 0
        for (plane, hit_point) in mesh.collision_with_ray(ray[0], ray[1]):
            hit_count += 1
            assert ray[2]
            if ray[4] is not None:
                assert (hit_point - np.array(ray[4])).sum() < 0.000001
        assert hit_count == ray[3]