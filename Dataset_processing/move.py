import shutil
import open3d as o3d

# # Percorsi dei file originali
series = ['00', '14', '20', '21', '25', '27']
# labels = ['low', 'medium', 'high']
#
# for serie in series:
#     for label in labels:
#         for num_config in range(25):
#             if not (serie == '00' and label == 'low'):
#                 file_ply = ("C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\Dataset\\Meshes\\" + label + "DefConfigs\\" +
#                             serie + label + "\\" + str(num_config) + "\\PointCloud.ply")
#                 file_txt = ("C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\Dataset\\Meshes\\" + label + "DefConfigs\\" +
#                             serie + label + "\\" + str(num_config) + "\\Info.txt")
#
#                 # Nuovi nomi dei file
#                 dest_ply = ("C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\Dataset\\Meshes\\Models\\" + serie +
#                                   "\\" + label + str(num_config) + ".ply")
#                 dest_txt = ("C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\Dataset\\Meshes\\Models\\" + serie +
#                                   "\\" + label + str(num_config) + "_info.txt")
#
#                 # Sposta e rinomina i file
#                 shutil.move(file_ply, dest_ply)
#                 shutil.move(file_txt, dest_txt)
#
# print("File spostati e rinominati con successo!")
for serie in series:
    filename_mesh = ("C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\Dataset\\Meshes\\Models\\" + serie +
                     "\\BasePose\\Merged.ply")
    mesh = o3d.io.read_triangle_mesh(filename_mesh)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals
    filename_pc = ("C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\Dataset\\Meshes\\Models\\" + serie +
                                       "\\base.ply")
    o3d.io.write_point_cloud(filename_pc, pcd)