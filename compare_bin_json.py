from pycolmap import SceneManager
import numpy as np
import os
import json
import torch
colmap_dir = '/home/szxie/DL3DV/colmap_all/1K/0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3/colmap/sparse/0'
tf_path = '/home/szxie/DL3DV/colmap_all/1K/0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3/transforms.json'

def get_bin_cams():
    manager = SceneManager(colmap_dir)
    manager.load_cameras()
    manager.load_images()
    manager.load_points3D()

    imdata = manager.images
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

    w2c_mats = []
    for k in imdata:
        im = imdata[k]
        rot = im.R()
        trans = im.tvec.reshape(3, 1)
        w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
        w2c_mats.append(w2c)

    w2c_mats = np.stack(w2c_mats, axis=0)
    # Convert extrinsics to camera-to-world.
    camtoworlds = np.linalg.inv(w2c_mats)
    return camtoworlds


def get_tf_cams():
    c2w_mats = []
    with open(os.path.join(tf_path)) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        
        for idx, frame in enumerate(frames):
            # already ranked, no need
            matrix = np.array(frame["transform_matrix"]) # opencv model?
            # w2c_mats.append(matrix)
            # print(matrix) w2c or c2w?
            # w2c_mats.append(matrix)
            c2w_mats.append(matrix)
        camtoworlds = np.stack(c2w_mats, axis=0)
        return camtoworlds

bin_cams = get_bin_cams()
tf_cams = get_tf_cams()

print(bin_cams.shape)
print(bin_cams[0])
# print(bin_cams[1])
print(tf_cams.shape)
print(tf_cams[0])

tf_cams = np.concatenate([tf_cams[:, 1:2, :], tf_cams[:, 0:1, :], tf_cams[:, 2:]], axis=1)
tf_cams[:, 0, :3] = -tf_cams[:, 0, :3]
tf_cams[:, 1, :3] = -tf_cams[:, 1, :3]
tf_cams[:, :3, 0] = -tf_cams[:, :3, 0]
tf_cams[:, 2, 3] = -tf_cams[:,2, 3]

for i in range(100):
    print(np.sum(tf_cams[i] - bin_cams[i]))

# tf0 = torch.tensor(tf_cams[0])
# # print(tf_cams[1])
# # print(tf0[1:2, :].shape)
# # tf0 = torch.concat([tf0[1:2, :], tf0[0:1, :], tf0[2:]], dim=0)
# # tf0[0, :3] = -tf0[0, :3]
# # tf0[1, :3] = -tf0[1, :3]
# # tf0[:3, 0] = -tf0[:3, 0]
# # tf0[2, 3] = -tf0[2, 3]
# print(tf0 - bin_cams[0])
