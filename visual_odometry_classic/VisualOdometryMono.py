import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm

from tools import *


class VisualOdometryMono():
	def __init__(self, sequence_name: str) -> None:
		self.sequence_name = sequence_name
		self.dataset_handler = DatasetHandler(sequence_name)
		self.detector = cv2.ORB_create()

		FLANN_INDEX_LSH = 6
		index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
		search_params = dict(checks=50)

		self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)


	def get_matches(self, previous_image, current_image):
		keypoints1, descriptors1 = self.detector.detectAndCompute(previous_image, None)
		keypoints2, descriptors2 = self.detector.detectAndCompute(current_image, None)

		matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)

		good = []
		for m, n in matches:
			if m.distance < 0.5 * n.distance:
				good.append(m)

		q1 = np.float32([keypoints1[m.queryIdx].pt for m in good])
		q2 = np.float32([keypoints2[m.trainIdx].pt for m in good])

		return q1, q2


	def decompose_essential_mat(self, E, intrinsic_matrix, q1, q2):
		"""
		Decompose the Essential matrix

		Parameters
		----------
		E (ndarray): Essential matrix
		q1 (ndarray): The good keypoints matches position in i-1'th image
		q2 (ndarray): The good keypoints matches position in i'th image

		Returns
		-------
		right_pair (list): Contains the rotation matrix and translation vector
		"""

		R1, R2, t = cv2.decomposeEssentialMat(E)
		T1 = form_transf(R1, np.ndarray.flatten(t))
		T2 = form_transf(R2, np.ndarray.flatten(t))
		T3 = form_transf(R1, np.ndarray.flatten(-t))
		T4 = form_transf(R2, np.ndarray.flatten(-t))
		transformations = [T1, T2, T3, T4]

		# Homogenize K
		K = np.concatenate((intrinsic_matrix, np.zeros((3, 1))), axis=1)
		# print(f"Before homogenization: {self.K}")
		# print(f"After homogenization: {K}")

		# List of projections
		projections = [K @ T1, K @ T2, K @ T3, K @ T4]

		np.set_printoptions(suppress=True)

		# print ("\nTransform 1\n" +  str(T1))
		# print ("\nTransform 2\n" +  str(T2))
		# print ("\nTransform 3\n" +  str(T3))
		# print ("\nTransform 4\n" +  str(T4))

		positives = []
		for P, T in zip(projections, transformations):
			hom_Q1 = cv2.triangulatePoints(self.dataset_handler.projectionMatrix_left, P, q1.T, q2.T)
			hom_Q2 = T @ hom_Q1
			# Un-homogenize
			Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
			Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

			total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
			relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1) /
			                         np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
			positives.append(total_sum + relative_scale)

		# Decompose the Essential matrix using built in OpenCV function
		# Form the 4 possible transformation matrix T from R1, R2, and t
		# Create projection matrix using each T, and triangulate points hom_Q1
		# Transform hom_Q1 to second camera using T to create hom_Q2
		# Count how many points in hom_Q1 and hom_Q2 with positive z value
		# Return R and t pair which resulted in the most points with positive z

		max = np.argmax(positives)
		if (max == 2):
			# print(-t)
			return R1, np.ndarray.flatten(-t)
		elif (max == 3):
			# print(-t)
			return R2, np.ndarray.flatten(-t)
		elif (max == 0):
			# print(t)
			return R1, np.ndarray.flatten(t)
		elif (max == 1):
			# print(t)
			return R2, np.ndarray.flatten(t)


	def get_pose(self, intrinsic_matrix, q1, q2):
		"""
		Calculates the transformation matrix

		Parameters
		----------
		q1 (ndarray): The good keypoints matches position in i-1'th image
		q2 (ndarray): The good keypoints matches position in i'th image

		Returns
		-------
		transformation_matrix (ndarray): The transformation matrix
		"""

		essential, mask = cv2.findEssentialMat(q1, q2, intrinsic_matrix)
		# print("\nEssential matrix:\n" + str(Essential))

		R, t = self.decompose_essential_mat(essential, intrinsic_matrix, q1, q2)

		return form_transf(R, t)


	def visual_odometry(self, subset: int = None):
		if subset is None:
			num_frames = self.dataset_handler.num_frames

		else:
			num_frames = subset

		self.dataset_handler.reset_frames()
		image_next = next(self.dataset_handler.images_left)

		transformation_matrix = np.eye(4, dtype=np.float64)
		trajectory = np.zeros((num_frames, 3, 4))
		trajectory[0] = transformation_matrix[:3, :]

		# true_path = []
		# estimated_path = []
		for i in tqdm(range(num_frames - 1)):
			image_current = image_next
			image_next = next(self.dataset_handler.images_left)

			q1, q2 = self.get_matches(image_current, image_next)
			motion = self.get_pose(self.dataset_handler.intrinsic_matrix, q1, q2)

			# gt_pose = self.dataset_handler.ground_truth[0, :, :]
			# current_pose = gt_pose
			# previous_image = self.dataset_handler.first_image_left

			transformation_matrix = transformation_matrix @ np.linalg.inv(motion)
			trajectory[i+1, :, :] = transformation_matrix[:3, :]

		plot_path_with_matrix(self.sequence_name, self.dataset_handler.ground_truth[:subset, :, :], trajectory)



if __name__ == '__main__':
	vo = VisualOdometryMono("02")
	# q1, q2 = vo.get_matches(vo.dataset_handler.first_image_left, vo.dataset_handler.second_image_left)
	# temp = vo.get_pose(vo.dataset_handler.intrinsic_matrix, q1, q2)
	# print(np.linalg.inv(temp))
	#
	# print(vo.dataset_handler.ground_truth[1])

	vo.visual_odometry(subset=30)

