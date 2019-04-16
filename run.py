import time

import torch

import src.MeshMask as mm
from src.util import load_mesh_data


def main():
    mesh_path = "Sphere.mesh.xml"
    image_height = 100
    image_width = 100
    mesh_vertices, normals, bone_weights, triangles = load_mesh_data(mesh_path)
    mesh = torch.tensor(mesh_vertices, dtype=torch.float32).repeat(32, 1, 1)
    print("Input size: {}".format(mesh.shape))

    # Mesh Mask
    mesh_mask = mm.MeshMask([image_height, image_width],
                            triangles)

    print("**CPU**")
    masks = torch.ByteTensor(mesh.shape[0], mesh.shape[1])
    s = time.time()
    for i in range(mesh.shape[0]):
        masks[i] = mesh_mask(mesh[i])
    print("Compute time: {}s".format(time.time() - s))
    s = time.time()
    for i in range(mesh.shape[0]):
        temp = mesh[i, masks[i]]
    print("Access time: {}s".format(time.time() - s))

    mesh_mask.triangles = triangles.cuda()
    mesh = mesh.cuda()

    print("**CUDA**")
    s = time.time()
    mask = mesh_mask(mesh)
    print("Compute time: {}s".format(time.time() - s))
    s = time.time()
    for i in range(mesh.shape[0]):
        temp = mesh[i, mask[i]]
    print("Access time: {}s".format(time.time() - s))


if __name__ == "__main__":
    main()

