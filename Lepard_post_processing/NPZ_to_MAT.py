import numpy as np
from scipy.io import savemat

filename_in = "C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\LepardTests\\2711\\correspondences.npz"
data = np.load(filename_in)
corr = data['correspondences']
print(corr[0,0])
print(corr[0,1])
new_data = []
for i in range(corr.shape[0]):
    new_data.append([corr[i, 1], corr[i, 0]])
data_out = np.array(new_data, dtype=object)
filename_out = "correspondences.mat"
savemat(filename_out, {"correspondences": data_out})