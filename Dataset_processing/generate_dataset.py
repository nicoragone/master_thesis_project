import random

import numpy as np
import open3d as o3d
import os

# Read point clouds
OUT_BASE = "C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\Dataset\\4D_Match_dataset_heavy\\split\\split\\"
models = ['14', '20', '21', '25', '27']
scales = [769, 763, 750, 642, 680]
splits = ['train', 'train', 'train', 'val', '4DMatch']
labels = ['low', 'medium', 'high']

for serie, scale, split in zip(models, scales, splits):
    src_folder = ("C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\Dataset\\Clouds_for_lepard\\Model" +
                  serie + "\\")
    src_path = src_folder + "base.ply"
    src_pc = o3d.io.read_point_cloud(src_path)
    folder_out = OUT_BASE  + split + "\\Model" + serie
    os.makedirs(folder_out, exist_ok=True)
    for label in labels:
        for num_config in range(200):
            tgt_path = src_folder + label + str(num_config) + ".ply"
            tgt_pc = o3d.io.read_point_cloud(tgt_path)

            # Extract numpy array points
            src_pts = np.asarray(src_pc.points).copy()
            tgt_pts = np.asarray(tgt_pc.points).copy()
            centroid = tgt_pts.mean(axis=0)
            # Centering clouds with respect with the target centroid
            src_pts -= centroid
            tgt_pts -= centroid

            # Extract source to target flow
            flow = tgt_pts - src_pts
            vec_filename = src_folder + label + str(num_config) + "_info.txt"
            vector_field = np.loadtxt(vec_filename, delimiter="\t", skiprows=5)

            # Defining global rotation matrix and translation vector
            R = (np.eye(3)).astype(np.float64)
            t = np.asarray([[0.0], [0.0], [0.0]], dtype=np.float64)

            # Defining point correspondences
            correspondances = np.zeros((src_pts.shape[0], 2))
            for i in range(src_pts.shape[0]):
                correspondances[i, :] = [i, i]

            # Overlap rate
            overlap_rate = 1.0

            # Metric index
            metric_index = np.array(random.sample(range(src_pts.shape[0]), int(0.1 * src_pts.shape[0]))).reshape(-1, 1)

            # Saving data dictionary
            filename_out = folder_out + "\\" + label + str(num_config)
            np.savez_compressed(
                filename_out,
                s_pc = src_pts.astype(np.float32),
                t_pc = tgt_pts.astype(np.float32),
                s2t_flow = vector_field.astype(np.float32),
                correspondences = correspondances.astype(np.int64),
                s_overlap_rate = np.float64(overlap_rate),
                rot = R,
                trans = t,
                metric_index = metric_index.astype(np.int64)
            )