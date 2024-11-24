import cv2
import matplotlib.pyplot as plt


if __name__ == "__main__":
   path = "/home/washindeiru/studia/7_semestr/vo/visual_odometry/dataset/m2dgr3/rgb"
   image = "1628058876241695166.png"

   img = cv2.imread(path + "/" + image, cv2.COLOR_BGR2RGB)

   gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   cv2.imshow("image", gray_image)
   cv2.waitKey(0)