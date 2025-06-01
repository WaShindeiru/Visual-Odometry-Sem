import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":
   path_to_calibrate = "/media/washindeiru/840265A302659B46/odom_files/kitti/02/image_0"
   path_to_write = "/media/washindeiru/840265A302659B46/odom_files/kitti/02/image_0_resized"

   frame_size = (960, 540)

   # num_of_images = 10

   image_path_list = [x for x in os.listdir(path_to_calibrate) if x.endswith(".png")]
   image_path_list = sorted(image_path_list, key=lambda x: int(x.split('.')[0]))
   # image_path_list = image_path_list[:num_of_images]

   for image_path in image_path_list:
      image = cv2.imread(path_to_calibrate + "/" + image_path, 0)
      image = cv2.resize(image, frame_size)
      cv2.imwrite(path_to_write + "/" + image_path, image)
