import os
import json 
from typing import Any, Dict, List, Optional
from typing_extensions import assert_never

from PIL import Image
import numpy as np
from .common import Parser
import imageio.v2 as imageio
import torch
import cv2

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)

# check: https://docs.nerf.studio/quickstart/data_conventions.html

class DL3DVParser(Parser):
    def __init__(
        self,
        data_dir: str,
        factor: int = 4,
        normalize: bool = False,
        test_every: int = 8,
        use_undistortion: bool = True
    ):
        super().__init__()

        self.data_dir = data_dir
        self.factor = factor 
        self.image_paths = []
        # self.camera_ids = []
        self.image_ids = []
        w2c_mats = []
        c2w_mats = []
        self.normalize = normalize
        self.test_every = test_every
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        
        transformsfile = os.path.join(data_dir, 'transforms.json')
        with open(os.path.join(transformsfile)) as json_file:
            contents = json.load(json_file)
            
            fx, fy, cx, cy = contents["fl_x"], contents["fl_y"], contents["cx"], contents["cy"]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            
            self.K = K
            
            # Get distortion parameters. OPENCV model for DL3DV
            type_ = contents["camera_model"]
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([contents["k1"], 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([contents["k1"], contents["k2"], 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([contents["k1"], contents["k2"], contents["p1"], contents["p2"]], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([contents["k1"], contents["k2"], contents["k3"], contents["k4"]], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"
            
            self.params = params
            self.camtype = camtype
            
            self.imsize = (contents["w"] // factor, contents["h"] // factor)
            width, height = self.imsize
            
            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)
            
            self.mapx = mapx
            self.mapy = mapy
            self.K = K_undist
            self.roi_undist = roi_undist
            self.imsize = (roi_undist[2], roi_undist[3])
            
            frames = contents["frames"]
            for idx, frame in enumerate(frames):
                # already ranked, no need
                matrix = np.array(frame["transform_matrix"]) # opencv model?
                # w2c_mats.append(matrix)
                # print(matrix) w2c or c2w?
                # w2c_mats.append(matrix)
                c2w_mats.append(matrix)
                
                image_dir = os.path.join(data_dir, f'images_{factor}')
                image_name = frame["file_path"].split('/')[-1]
                image_path = os.path.join(image_dir, image_name)
                self.image_paths.append(image_path)
                self.image_ids.append(frame["colmap_im_id"])
            
            # w2c_mats = np.stack(w2c_mats, axis=0)
            # self.camtoworlds = np.linalg.inv(w2c_mats)
            camtoworlds = np.stack(c2w_mats, axis=0)
        camtoworlds = np.concatenate([camtoworlds[:, 1:2, :], camtoworlds[:, 0:1, :], camtoworlds[:, 2:]], axis=1)
        camtoworlds[:, 0, :3] = -camtoworlds[:, 0, :3]
        camtoworlds[:, 1, :3] = -camtoworlds[:, 1, :3]
        camtoworlds[:, :3, 0] = -camtoworlds[:, :3, 0]
        camtoworlds[:, 2, 3] = -camtoworlds[:,2, 3]
        self.camtoworlds = camtoworlds
                
        
        if normalize:
            T1 = similarity_from_cameras(self.camtoworlds)
            self.camtoworlds = transform_cameras(T1, self.camtoworlds)
            transform = T1
        else:
            transform = np.eye(4)    
        self.transform = transform
        
        camera_locations = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)
        # Convert extrinsics to camera-to-world.
        # self.camtoworlds = np.linalg.inv(w2c_mats)
            

    
class DL3DVDataset:
    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        use_undistortion: bool = True
    ):
        self.parser = parser
        self.split = split 
        indices = np.arange(len(self.parser.image_paths))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]
        self.use_undistort = use_undistortion

    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        image_id = self.parser.image_ids[index]
        K = self.parser.K.copy()
        params = self.parser.params
        camtoworld = self.parser.camtoworlds[index]
        
        if len(params) > 0 and self.use_undistort:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx,
                self.parser.mapy,
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist
            image = image[y : y + h, x : x + w]
        
        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworld).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }
        
        return data