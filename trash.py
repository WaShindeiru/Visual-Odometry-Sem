import pandas as pd
import numpy as np


def trash():
    # Read the CSV file separated by spaces
    df = pd.read_csv(
        '/home/washindeiru/studia/7_semestr/vo/visual_odometry/Tartanvo/results/m2dgr3_rgb_calib+2025-01-14_18:51:41/results_s3_with_timestamp.txt',
        delim_whitespace=True, header=None)

    # Convert the first column from 1e18 to 1e9 by dividing by 1e9
    df[0] = df[0] / 1e9

    # Save the updated DataFrame back to a CSV file
    df.to_csv(
        '/home/washindeiru/studia/7_semestr/vo/visual_odometry/Tartanvo/results/m2dgr3_rgb_calib+2025-01-14_18:51:41/results_se3.txt',
        sep=' ', index=False, header=False)

def add_column(path, output_path):
    df = pd.read_csv(path, sep=" ")

    # Convert the DataFrame to a numpy array
    data = df.to_numpy()

    # Create a new column with numbers from 1 to the length of the data
    new_column = np.arange(1, data.shape[0] + 1).reshape(-1, 1)

    # Concatenate the new column with the existing data
    updated_data = np.concatenate([new_column, data], axis=1)

    np.savetxt(output_path + '/poses_se3.txt', updated_data, delimiter=' ', fmt='%f')


if __name__ == "__main__":
    add_column("/media/washindeiru/EE366BA9366B718F/odom/dataset/poses/02.txt",
               "/home/washindeiru/studia/7_semestr/vo/visual_odometry/LightGlue/results/kitti_02+2025-06-01_00:36:04")
