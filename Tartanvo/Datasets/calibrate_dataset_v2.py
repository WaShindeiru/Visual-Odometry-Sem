import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def calibrate(image, camera_matrix, distortion_coefficients):
   h, w = image.shape

   newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 1, (w, h))

   dst = cv2.undistort(image, camera_matrix, distortion_coefficients, None, newcameramatrix)

   # x, y, w, h = roi
   # dst = dst[y:y + h, x:x + w]

   return dst


if __name__ == "__main__":
   path_to_calibrate = "/home/washindeiru/studia/7_semestr/vo/visual_odometry/dataset/m2dgr3/black"
   path_to_write = "/home/washindeiru/studia/7_semestr/vo/visual_odometry/dataset/m2dgr3/black_calibrate"

   num_of_images = 1000

   image_path_list = sorted(os.listdir(path_to_calibrate), key=lambda x: int(x.split(".")[0]))
   image_path_list = image_path_list[:num_of_images]

   camera_matrix = np.array([[617.971050917033, 0., 327.710279392468],
                             [0., 616.445131524790, 253.976983707814],
                             [0., 0., 1.]])

   distortion_coefficients = np.array([0.148000794688248, -0.217835187249065, 0, 0])


   for image_path in image_path_list:
      image = cv2.imread(path_to_calibrate + "/" + image_path, cv2.IMREAD_GRAYSCALE)
      image = calibrate(image, camera_matrix, distortion_coefficients)
      cv2.imwrite(path_to_write + "/" + image_path, image)