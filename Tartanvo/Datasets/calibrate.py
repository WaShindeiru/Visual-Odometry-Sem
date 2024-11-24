import cv2
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
   path = "/media/washindeiru/Hard Disc/odometry_files/odometry_files/m2dgr/data"
   image = "1628059159326742649.png"

   img = cv2.imread(path + "/" + image, cv2.IMREAD_GRAYSCALE)
   plt.imshow(img, cmap='gray')
   plt.show()