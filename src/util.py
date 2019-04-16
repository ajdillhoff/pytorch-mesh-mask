import xml.etree.ElementTree as ET
import numpy as np
import torch


def load_mesh_data(mesh_path):
    """Loads mesh vertices, bone assignments, and triangle IDs.

    Args:
        mesh_path - string: Path to the OGRE XML mesh data.

    Returns:
        mesh_vertices - array (N_v x 3): Mesh vertices, where N_v is the
            number of vertices.
        bone_weights - array (N_b x N_v): Bone weights, where N_b is the bone
            count and N_v is the number of vertices.
        triangles - array (N_f x 3): Triangle IDs, where N_f is the number of
            triangle faces in the mesh.
    """

    tree = ET.parse(mesh_path)
    root = tree.getroot()

    # Store all bone assignments
    bone_assignment_dict = {}
    bone_weight_dict = {}
    num_bones = 0
    for child in root[4]:
        key = 'vertex_' + str(child.attrib['vertexindex'])
        bone_index = int(child.attrib['boneindex'])
        if bone_index > num_bones:
            num_bones = bone_index

        if key in bone_assignment_dict:
            bone_weight_dict[key] = np.append(bone_weight_dict[key], np.array([float(child.attrib['weight'])]))
            bone_assignment_dict[key] = np.append(bone_assignment_dict[key], np.array([bone_index]))
        else:
            bone_weight_dict[key] = np.array([float(child.attrib['weight'])])
            bone_assignment_dict[key] = np.array([bone_index])

    num_bones += 1 # because num_bones is only as large as the biggest index.

    # Store the vertices
    mesh_vertices = np.zeros((int(root[0].attrib['vertexcount']), 3))
    normals = np.zeros((int(root[0].attrib['vertexcount']), 3))
    i = 0
    for child in root[0][0]:
        mesh_vertices[i, 0] = child[0].attrib['x']
        mesh_vertices[i, 1] = child[0].attrib['y']
        mesh_vertices[i, 2] = child[0].attrib['z']
        normals[i, 0] = child[1].attrib['x']
        normals[i, 1] = child[1].attrib['y']
        normals[i, 2] = child[1].attrib['z']
        i += 1

    # Build the bone_weights matrix
    bone_weights = np.zeros((num_bones, len(mesh_vertices)))
    i = 0
    for key, value in bone_assignment_dict.items():
        bone_assignments = value
        bone_weight = bone_weight_dict[key]
        bone_weights[bone_assignments, i] = bone_weight
        i += 1

    triangles_idxs = None
    vertex_map = [1, 2, 0]
    i = 0

    for submesh in root[1]:
        for faces in submesh:
            num_faces = int(faces.attrib['count'])
            if triangles_idxs is None:
                triangles_idxs = np.zeros((num_faces, 3), dtype=int)
            else:
                triangles_idxs = np.append(triangles_idxs, np.zeros((num_faces, 3), dtype=int), axis=0)

            for face in faces:
                j = 0
                for _, value in face.attrib.items():
                    triangles_idxs[i, vertex_map[j]] = int(value)
                    j += 1
                i += 1

    triangles = torch.from_numpy(triangles_idxs.astype(np.int32))

    return mesh_vertices, normals, bone_weights, triangles

