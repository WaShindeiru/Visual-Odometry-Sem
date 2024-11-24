import matplotlib.pyplot as plt
import numpy as np
import os

from Tartanvo.Datasets.transformation import SE2pos_quat


def form_transf(R, t):
   """
   Makes a transformation matrix from the given rotation matrix and translation vector

   Parameters
   ----------
   R (ndarray): The rotation matrix
   t (list): The translation vector

   Returns
   -------
   T (ndarray): The transformation matrix
   """
   T = np.eye(4, dtype=np.float64)
   T[:3, :3] = R
   T[:3, 3] = t
   return T


def plot_path(sequence_name, ground_truth_path, predicted_path):
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

   # gt_numpy = np.array(ground_truth_path)
   # pt_numpy = np.array(predicted_path)

   ax1.plot(ground_truth_path[:, 0], ground_truth_path[:, 1], label="ground truth")
   ax1.plot(predicted_path[:, 0], predicted_path[:, 1], label="prediction")
   ## to delete
   # ax1.set_xlim((-150, 50))
   ax1.set_xlabel("X")
   ax1.set_ylabel("Y")
   ax1.set_title("Trajectory")

   error = np.linalg.norm(ground_truth_path - predicted_path, axis=1)
   ax2.plot(np.arange(0, error.shape[0], 1), error, label="error")
   ax2.set_title("Error")
   ax2.set_xlabel("Frame")
   ax2.set_ylabel("Error")

   ax1.legend()
   # ax2.legend()

   try:
      os.mkdir(f"./results/{sequence_name}")
   except FileExistsError:
      pass

   fig.savefig(f"./results/{sequence_name}/results.png")

   fig.show()


def make_matrix_homogenous(matrix):
   temp = np.eye(4, dtype=np.float64)
   temp[:3, :3] = matrix[:3, :3]
   temp[:3, 3] = matrix[:3, 3]
   return temp



def plot_path_with_matrix(sequence_name, ground_truth_path_matrix, predicted_path_matrix):
   ground_truth_path = ground_truth_path_matrix[:, [0, 1], [3, 3]]
   predicted_path = predicted_path_matrix[:, [0, 1], [3, 3]]
   np.savetxt(f"./results/{sequence_name}/results.txt", predicted_path_matrix.reshape(-1, 12), delimiter=' ')
   plot_path(sequence_name, ground_truth_path, predicted_path)


def plot_path_with_matrix_and_angle(sequence_name, ground_truth_path_matrix, predicted_path_matrix, error_angle_list):
   ground_truth_path = ground_truth_path_matrix[:, [0, 1], [3, 3]]
   predicted_path = predicted_path_matrix[:, [0, 1], [3, 3]]

   fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
   ax1.plot(ground_truth_path[:, 0], ground_truth_path[:, 1], label="ground truth")
   ax1.plot(predicted_path[:, 0], predicted_path[:, 1], label="prediction")
   ax1.set_xlabel("X")
   ax1.set_ylabel("Y")
   ax1.set_title("Trajectory")

   error_angle_matrix = np.array(error_angle_list, dtype=np.float64)
   ax2.plot(np.arange(0, error_angle_matrix.shape[0], 1), error_angle_matrix[:, 0])
   ax2.set_title("Rotation error angle")
   ax2.set_xlabel("Frame")
   ax2.set_ylabel("Angle error (degrees)")

   ax3.plot(np.arange(0, error_angle_matrix.shape[0], 1), error_angle_matrix[:, 1])
   ax3.set_title("Translation error angle")
   ax3.set_xlabel("Frame")
   ax3.set_ylabel("Angle error (degrees)")

   ax1.legend()

   fig.savefig(f"./results/{sequence_name}/results_angle.png")
   fig.show()


def save_as_quat(sequence_name, predicted_path_matrix):
   quat_matrix = np.zeros((predicted_path_matrix.shape[0], 7), dtype=np.float64)
   for i in range(0, predicted_path_matrix.shape[0]):
      quat_matrix[i, :] = SE2pos_quat(predicted_path_matrix[i, :, :])

   np.savetxt(f"./results/{sequence_name}/results_quat.txt", quat_matrix, delimiter=' ')


def save_as_s3(sequence_name, predicted_path_matrix):
   np.savetxt(f"./results/{sequence_name}/results_s3.txt", predicted_path_matrix.reshape(-1, 12), delimiter=' ')


def save_3d_plot(sequence_name, predicted_path_matrix):
   predicted_path_matrix = predicted_path_matrix.reshape(-1, 12)
   fig = plt.figure(figsize = (7, 6))
   ax = fig.add_subplot(111, projection='3d')
   ax.plot(predicted_path_matrix[:, 3], predicted_path_matrix[:, 7], predicted_path_matrix[:, 11])
   ax.set_xlabel('x (m)')
   ax.set_ylabel('y (m)')
   ax.set_zlabel('z (m)')

   fig.savefig(f"./results/{sequence_name}/results_3d.png")