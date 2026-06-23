import numpy as np
import random
from scipy.spatial import cKDTree

# Data files
models = ['14', '20', '21', '25', '27']
splits = ['train', 'train', 'train', 'val', '4DMatch']
labels = ['low', 'medium', 'high']


def compute_volume_bounds(models, splits, labels):
    # Compute global min and global max
    minX, maxX = np.inf, -np.inf
    minY, maxY = np.inf, -np.inf
    minZ, maxZ = np.inf, -np.inf
    for model, split in zip(models, splits):
        if split == "4DMatch":
            num_configs = 30
        else:
            num_configs = 200
        for label in labels:
            for num_config in range(num_configs):
                filename = (
                            "C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\Dataset\\4D_Match_dataset_heavy\\split\\split\\" +
                            split + "\\Model" + model + "\\" + label + str(num_config) + ".npz")
                data = np.load(filename)
                src_pts = data['s_pc']
                tgt_pts = data['t_pc']
                pc_matrix = np.vstack((src_pts, tgt_pts))
                Xmin, Xmax = np.min(pc_matrix[:, 0]), np.max(pc_matrix[:, 0])
                Ymin, Ymax = np.min(pc_matrix[:, 1]), np.max(pc_matrix[:, 1])
                Zmin, Zmax = np.min(pc_matrix[:, 2]), np.max(pc_matrix[:, 2])
                if Xmin < minX:
                    minX = Xmin
                if Xmax > maxX:
                    maxX = Xmax
                if Ymin < minY:
                    minY = Ymin
                if Ymax > maxY:
                    maxY = Ymax
                if Zmin < minZ:
                    minZ = Zmin
                if Zmax > maxZ:
                    maxZ = Zmax

    # Compute volume bounds
    rangeX = maxX - minX
    rangeY = maxY - minY
    rangeZ = maxZ - minZ
    vol_bounds = [[minX - 0.1 * rangeX, minY - 0.1 * rangeY, minZ - 0.1 * rangeZ],
                  [maxX + 0.1 * rangeX, maxY + 0.1 * rangeY, maxZ + 0.1 * rangeZ]]
    vol_bounds = [[float(x) for x in row] for row in vol_bounds]
    return vol_bounds


vol_bounds = compute_volume_bounds(models, splits, labels)
print("Parameters to insert into YAML file:")
print("\nVolume bounds:", vol_bounds)