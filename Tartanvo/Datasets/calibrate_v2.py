import cv2
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
   path = "/home/washindeiru/studia/7_semestr/vo/visual_odometry/dataset/m2dgr3/black"
   image = "1628058876173331499.png"

   camera_matrix = np.array([[617.971050917033, 0., 327.710279392468],
                             [0., 616.445131524790, 253.976983707814],
                             [0., 0., 1.]])

   distortion_coefficients = np.array([0.148000794688248, -0.217835187249065, 0, 0])

   img = cv2.imread(path + "/" + image, cv2.IMREAD_GRAYSCALE)
   plt.imshow(img, cmap='gray')
   plt.show()

   h, w = img.shape

   newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 1, (w, h))

   dst = cv2.undistort(img, camera_matrix, distortion_coefficients, None, newcameramatrix)

   plt.imshow(dst, cmap='gray')
   plt.show()

   # crop the image
   x, y, w, h = roi
   dst = dst[y:y + h, x:x + w]

   plt.imshow(dst, cmap='gray')
   plt.show()
