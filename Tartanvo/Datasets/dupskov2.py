import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
   path = "/media/washindeiru/Hard Disc/odometry_files/odometry_files/m2dgr/"

   df = pd.read_csv(path + "gate_03.txt", sep=" ", header=None)
   df = df.to_numpy()

   plt.figure()
   plt.plot(df[:, 1], df[:, 3])
   plt.show()

   fig = plt.figure(figsize=(7, 6))
   ax = fig.add_subplot(111, projection='3d')
   ax.plot(df[:, 1], df[:, 2], df[:, 3])
   ax.set_xlabel('x (m)')
   ax.set_ylabel('y (m)')
   ax.set_zlabel('z (m)')
   plt.show()