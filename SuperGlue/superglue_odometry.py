import torch
import numpy as np
from pathlib import Path

from torchvision.transforms import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.matching import Matching
from models.utils import *
from tools import *
from tools import plot_path_with_matrix_and_angle


class SuperGlueOdometry():
   def __init__(self, sequence_name: str):
      self.sequence_name = sequence_name
      self.dataset_handler = DatasetHandler(self.sequence_name)
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

      config = {
         'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
         },
         'superglue': {
            # 'indoor', 'outdoor'
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
         }
      }

      self.matching = Matching(config).eval().to(self.device)

      self.output_directory = Path("results") / self.sequence_name
      self.output_directory.mkdir(parents=True, exist_ok=True)

   def visual_odometry(self, subset: int = None):
      if subset is None:
         num_frames = self.dataset_handler.num_frames

      else:
         num_frames = subset

      self.dataset_handler.reset_frames()
      image_next = next(self.dataset_handler.images_left)

      transformation_matrix = np.eye(4, dtype=np.float64)

      trajectory = np.zeros((num_frames, 3, 4))
      trajectory[0] = transformation_matrix[:3, :]

      error_angle = [(0, 0)]

      for i in tqdm(range(num_frames - 1)):
         image_current = image_next
         image_next = next(self.dataset_handler.images_left)

         image0, resized_first_image, scales0 = transform_image(image_current, 'cuda', (640, 480))
         image1, resized_second_image, scales1 = transform_image(image_next, 'cuda', (640, 480))
         # image0, resized_first_image, scales0 = transform_image(image_current, 'cuda')
         # image1, resized_second_image, scales1 = transform_image(image_next, 'cuda')

         result = vo.matching({'image0': resized_first_image, 'image1': resized_second_image})
         result = {key: value[0].cpu().detach().numpy() for key, value in result.items()}

         kpts0, kpts1 = result['keypoints0'], result['keypoints1']
         matches, conf = result['matches0'], result['matching_scores0']

         valid = matches > -1
         mkpts0 = kpts0[valid]
         mkpts1 = kpts1[matches[valid]]
         mconf = conf[valid]

         K = scale_intrinsics(vo.dataset_handler.intrinsic_matrix, scales0)

         thresh = 1.  # In pixels relative to resized image size.
         R, t, inliers = estimate_pose(mkpts0, mkpts1, K, K, thresh)

         motion = form_transf(R, t)

         error_t, error_R = compute_pose_error(np.linalg.inv(make_matrix_homogenous(vo.dataset_handler.ground_truth[i, :, :])) @ make_matrix_homogenous(vo.dataset_handler.ground_truth[i+1, :, :]), R, t)
         error_angle.append((error_R, error_t))

         transformation_matrix = transformation_matrix @ np.linalg.inv(motion)
         trajectory[i+1, :, :] = transformation_matrix[:3, :]

      plot_path_with_matrix_and_angle(self.sequence_name, self.dataset_handler.ground_truth[:subset, :, :], trajectory, error_angle)

if __name__ == "__main__":
   vo = SuperGlueOdometry("02")
   vo.visual_odometry(subset=1000)
   # first_image = vo.dataset_handler.first_image_left
   # second_image = vo.dataset_handler.second_image_left
   # # plt.imshow(first_image, 'gray')
   # # plt.show()
   #
   # image0, resized_first_image, scales0 = transform_image(first_image, 'cuda', (640, 480))
   # image1, resized_second_image, scales1 = transform_image(second_image, 'cuda', (640, 480))
   # # image0, resized_first_image, scales0 = transform_image(first_image, 'cuda')
   # # image1, resized_second_image, scales1 = transform_image(second_image, 'cuda')
   #
   # result = vo.matching({'image0': resized_first_image, 'image1': resized_second_image})
   # result = {key: value[0].cpu().detach().numpy() for key, value in result.items()}
   #
   # kpts0, kpts1 = result['keypoints0'], result['keypoints1']
   # matches, conf = result['matches0'], result['matching_scores0']
   #
   # valid = matches > -1
   # mkpts0 = kpts0[valid]
   # mkpts1 = kpts1[matches[valid]]
   # mconf = conf[valid]
   #
   # print(scales0)
   # print(scales1)
   #
   # K = scale_intrinsics(vo.dataset_handler.intrinsic_matrix, scales0)
   #
   # thresh = 1.  # In pixels relative to resized image size.
   # R, t, inliers = estimate_pose(mkpts0, mkpts1, K, K, thresh)
   #
   # matrix_result = form_transf(R, t)
   # print(np.linalg.inv(matrix_result))
   # print(vo.dataset_handler.ground_truth[1])

   # print(type(result))
   # print(result.keys())