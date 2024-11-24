import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def calibrate(image, camera_matrix, distortion_coefficients):
   w, h = image.shape

   newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 0, (w, h))

   mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, None, newcameramatrix, (w, h), 5)
   dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

   x, y, w, h = roi
   dst = dst[y:y + h, x:x + w]

   return dst


if __name__ == "__main__":
   path_to_calibrate = "./../../dataset/m2dgr3"
   path_to_write = "./../../dataset/m2dgr3_calib/image_0"

   num_of_images = 10

   image_path_list = sorted(os.listdir(path_to_calibrate + "/image_0"), key=lambda x: int(x.split(".")[0]))
   image_path_list = image_path_list[:num_of_images]

   camera_matrix = np.array([[540.645056202188, 0., 626.4125666883942],
                             [0., 539.8545023658869, 523.947634226782],
                             [0., 0., 1.]])

   distortion_coefficients = np.array(
      [-0.07015146608431883, 0.008586142263125124, -0.021968993685891842, 0.007442211946112636])

   # distortion_coefficients = np.array([-0.057963907006683066, -0.026465594265953234, 0.011980216320790046, -0.003041081642470451])

   for image_path in image_path_list:
      image = cv2.imread(path_to_calibrate + "/image_0/" + image_path, 0)
      image = calibrate(image, camera_matrix, distortion_coefficients)
      cv2.imwrite(path_to_write + "/" + image_path, image)