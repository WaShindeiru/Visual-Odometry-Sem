import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2

from Tartanvo.Datasets.transformation import quat2SO, quat2so, pos_quats2SEs

if __name__ == "__main__":
   df = pd.read_csv("../data/KITTI_10/pose_left.txt", sep=" ", header=None)
   df = df.to_numpy()
   # aha = quat2SO(df)
   # ehe = quat2so(df)
   #
   # print(aha.shape)
   # print(ehe.shape)

   # aha = pos_quats2SEs(df)
   # print(aha.shape)

   # plt.figure()
   # plt.plot(aha[:, 0], aha[:, 1])
   # plt.show()

   real = pd.read_csv("../data/KITTI_10/results.txt", sep=" ", header=None)
   real = real.to_numpy()

   kurwa = pd.read_csv("../results/KITTI_10/results.txt", sep=" ", header=None)
   kurwa = kurwa.to_numpy()

   fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

   ax1.plot(df[:, 0], df[:, 1])
   ax2.plot(real[:, 3], real[:, 11])
   ax3.plot(kurwa[:, 3], kurwa[:, 11])
   fig.show()

   # plt.figure()
   # plt.plot(df[:, 0], df[:, 1])
   # plt.show()
   #
   # plt.figure()
   # plt.plot(real[:, 3], real[:, 11])
   # plt.show()



   # calib = pd.read_csv("../data/KITTI_10/calib.txt", sep=" ", header=None, index_col=0)
   # projectionMatrix_left = np.array(calib.loc["P0:"]).reshape((3, 4))
   # k1, r1, t1, _, _, _, _ = cv2.decomposeProjectionMatrix(projectionMatrix_left)
   # t1 = t1 / t1[3]
   #
   # transform = np.eye(4, dtype=np.float64)
   # transform[:3, :3] = r1
   # transform[:3, 3] = t1[:3, 0]
   #
   # print(k1)
   # print(transform)