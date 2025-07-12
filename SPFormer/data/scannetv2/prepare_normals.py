import os

import numpy as np
import open3d as o3d
import segmentator
import torch


def get_normals(mesh_file):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    normals = vertex_normal(vertices.numpy()[:, :3], faces.numpy())
    return normals


if __name__ == "__main__":
    os.makedirs("dataset/scannetv2/normals", exist_ok=True)
    scans_trainval = os.listdir("dataset/scannetv2/scans/*")
    for scan in scans_trainval:
        ply_file = os.path.join("dataset/scannetv2/scans", scan, f"{scan}_vh_clean_2.ply")
        normals = get_normals(ply_file)
        normals = normals.numpy()

        torch.save(spp, os.path.join("dataset/scannetv2/normals", f"{scan}.pth"))

    scans_test = os.listdir("dataset/scannetv2/scans_test/*")
    for scan in scans_test:
        ply_file = os.path.join("dataset/scannetv2/scans_test", scan, f"{scan}_vh_clean_2.ply")
        normals = get_normals(ply_file)
        normals = normals.numpy()

        torch.save(spp, os.path.join("dataset/scannetv2/normals", f"{scan}.pth"))
