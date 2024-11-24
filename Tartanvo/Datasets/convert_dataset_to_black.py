import cv2
import os


if __name__ == "__main__":
   path_to_calibrate = "/home/washindeiru/studia/7_semestr/vo/visual_odometry/dataset/m2dgr3/rgb"
   path_to_write = "/home/washindeiru/studia/7_semestr/vo/visual_odometry/dataset/m2dgr3/black"

   # num_of_images = 10

   image_path_list = sorted(os.listdir(path_to_calibrate), key=lambda x: int(x.split(".")[0]))
   # image_path_list = image_path_list[:num_of_images]

   for image_path in image_path_list:
      image = cv2.imread(path_to_calibrate + "/" + image_path)
      gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      cv2.imwrite(path_to_write + "/" + image_path, gray_image)