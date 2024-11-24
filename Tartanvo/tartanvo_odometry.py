from os import mkdir
from os.path import isdir
from tqdm import tqdm
import pandas as pd

from TartanVO import TartanVO
from Tartanvo.Datasets.utils import dataset_intrinsics
from Datasets.tartanTrajFlowDataset import *
from Datasets.transformation import *
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow
from tools import *

from evaluator.tartanair_evaluator import TartanAirEvaluator

class TantanOdometry():
   def __init__(self, image_height=448, image_width=640, model_name="tartanvo_1914.pkl", datastr="kitti",
                test_dir="data/KITTI_10/image_left", batch_size=1, worker_num=5, ground_truth_path="./data/KITTI_10/results.txt"):

      self.model_name = model_name
      self.datastr = datastr

      self.tartan = TartanVO(model_name)
      self.focalx, self.focaly, self.centerx, self.centery = dataset_intrinsics(datastr)
      # if args.kitti_intrinsics_file.endswith('.txt') and datastr == 'kitti':
      #    focalx, focaly, centerx, centery = load_kiiti_intrinsics(args.kitti_intrinsics_file)

      transform = Compose([CropCenter((image_height, image_width)), DownscaleFlow(), ToTensor()])

      self.testDataset = TrajFolderDataset(test_dir, transform=transform, focalx=self.focalx, focaly=self.focaly, centerx=self.centerx, centery=self.centery)
      self.testDataloader = DataLoader(self.testDataset, batch_size=batch_size,
                                  shuffle=False, num_workers=worker_num)
      self.testDataiter = iter(self.testDataloader)

      ## delete this
      poses = pd.read_csv(ground_truth_path, sep=" ", header=None)

      self.ground_truth = np.zeros((len(poses), 3, 4))
      for i in range(len(poses)):
         self.ground_truth[i] = np.array(poses.iloc[i]).reshape((3, 4))


   def visual_odometry(self, subset: int = None):
      if subset is None:
         num_frames = len(self.testDataset) + 1
      else:
         num_frames = subset

      save_flow = True

      transformation_matrix = np.eye(4, dtype=np.float64)

      trajectory = np.zeros((num_frames, 3, 4))
      trajectory[0] = transformation_matrix[:3, :]

      motionlist = []

      testname = self.datastr + '_' + self.model_name.split('.')[0]
      # if save_flow:
      #    flowdir = 'results/' + testname + '_flow'
      #    if not isdir(flowdir):
      #       mkdir(flowdir)
      #    flowcount = 0

      for i in tqdm(range(num_frames - 1)):
         sample = next(self.testDataiter)

      # while True:
      #    try:
      #       sample = next(self.testDataiter)
      #    except StopIteration:
      #       break

         motions, flow = self.tartan.test_batch(sample)
         SE3motion = se2SE_better(motions)

         motionlist.extend(motions)

         transformation_matrix = transformation_matrix @ np.linalg.inv(SE3motion)
         trajectory[i+1, :, :] = transformation_matrix[:3, :]


      poselist = ses2poses_quat(np.array(motionlist))


      # calculate ATE, RPE, KITTI-RPE
      # if pose_file.endswith('.txt'):
      # evaluator = TartanAirEvaluator()
      # results = evaluator.evaluate_one_trajectory(poses, poselist, scale=True,
      #                                             kittitype=(datastr == 'kitti'))
      # if datastr == 'euroc':
      #    print("==> ATE: %.4f" % (results['ate_score']))
      # else:
      #    print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" % (
      #    results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))
      #
      # # save results and visualization
      # plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname='results/' + testname + '.png',
      #           title='ATE %.4f' % (results['ate_score']))
      # np.savetxt('results/' + testname + '.txt', results['est_aligned'])
      # else:


      # np.savetxt('results/' + testname + '.txt', poselist)

      plot_path_with_matrix("KITTI_10", self.ground_truth[:subset, :, :], trajectory)


if __name__ == "__main__":
   tartan = TantanOdometry()
   tartan.visual_odometry(subset=None)
