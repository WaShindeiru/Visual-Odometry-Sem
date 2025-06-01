import numpy as np
import cv2
import pickle
import os
from tqdm import tqdm

from SuperGlue.models.utils import make_matching_plot


def make_movie(pkl_path, photo_path):
    with open(pkl_path + '/keypoint.pkl', 'rb') as file:
        loaded_data = pickle.load(file)

        output_file = pkl_path + "/keypoints.mp4"
        frame_rate = 10
        # frame_size = (2492, 376)
        # frame_size = (640, 480)
        original = (1241, 376)
        single_photo_size = (960, 540)
        frame_size = (1920, 540)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
        video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size)

        rgbfiles = [ff for ff in os.listdir(photo_path) if (ff.endswith('.png') or ff.endswith('.jpg'))]
        for i in tqdm(range(1, len(rgbfiles) - 1, 1)):
        # for i in tqdm(range(1, 100, 1)):

            img0 = cv2.imread(photo_path + "/" + str(i-1).zfill(6) + ".png", cv2.IMREAD_GRAYSCALE)
            img1 = cv2.imread(photo_path + "/" + str(i).zfill(6) + ".png", cv2.IMREAD_GRAYSCALE)

            kpts0 = loaded_data['kpt0'][i]
            kpts0 = (kpts0 * [single_photo_size[0] / original[0], single_photo_size[1] / original[1]]).astype(int)
            kpts1 = loaded_data['kpt1'][i]
            kpts1 = (kpts1 * [single_photo_size[0] / original[0], single_photo_size[1] / original[1]]).astype(int)
            mkpts0 = loaded_data['mkpt0'][i]
            mkpts0 = (mkpts0 * [single_photo_size[0] / original[0], single_photo_size[1] / original[1]]).astype(int)
            mkpts1 = loaded_data['mkpt1'][i]
            mkpts1 = (mkpts1 * [single_photo_size[0] / original[0], single_photo_size[1] / original[1]]).astype(int)

            color = np.concatenate(
                [np.zeros((mkpts0.shape[0], 1)) * 0.5, np.ones((mkpts0.shape[0], 1)), np.zeros((mkpts0.shape[0], 1)),
                 np.ones((mkpts0.shape[0], 1))], axis=1)
            text = []

            frame = make_matching_plot(img0, img1, kpts0, kpts1, mkpts0, mkpts1, color, text,
                               None,
                               show_keypoints=True,
                               fast_viz=True)

            resized_frame = cv2.resize(frame, frame_size)

            video_writer.write(resized_frame)

        video_writer.release()




if __name__ == "__main__":
    pkl_path = '/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/kitti_02+2025-01-24_16:48:36'
    photo_path = "/media/washindeiru/840265A302659B46/odom_files/kitti/02/image_0_resized"
    make_movie(pkl_path, photo_path)
