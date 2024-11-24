import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
   path = "/media/washindeiru/Hard Disc/odometry_files/odometry_files/m2dgr/tum/gate_03/cam_aha/data"
   image_name = "1628058878897810936.png"
   img = cv2.imread(path + "/" + image_name)
   print(img.shape)
   plt.imshow(img)
   plt.show()