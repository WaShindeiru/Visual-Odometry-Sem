from os import mkdir
from os.path import isdir

from TartanVO import TartanVO
from Tartanvo.Datasets.utils import dataset_intrinsics
from Datasets.tartanTrajFlowDataset import *
from Datasets.transformation import *


class TantanOdometry():
   def __init__(self, model_name="tartanvo_1914.pkl", datastr="kitti", test_dir="data/KITTI_10/image_left", batch_size=1, worker_num=5):

      self.model_name = model_name
      self.datastr = datastr

      self.tartan = TartanVO(model_name)
      self.focalx, self.focaly, self.centerx, self.centery = dataset_intrinsics(datastr)
      # if args.kitti_intrinsics_file.endswith('.txt') and datastr == 'kitti':
      #    focalx, focaly, centerx, centery = load_kiiti_intrinsics(args.kitti_intrinsics_file)

      # transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])

      self.testDataset = TrajFolderDataset(test_dir, focalx=self.focalx, focaly=self.focaly, centerx=self.centerx, centery=self.centery)
      self.testDataloader = DataLoader(self.testDataset, batch_size=batch_size,
                                  shuffle=False, num_workers=worker_num)
      self.testDataiter = iter(self.testDataloader)

   def visual_odometry(self, subset: int = None):

      save_flow = True

      transformation_matrix = np.eye(4, dtype=np.float64)

      # trajectory = np.zeros((len(self.testDataset), 3, 4))
      # trajectory[0] = transformation_matrix[:3, :]

      motionlist = []

      testname = self.datastr + '_' + self.model_name.split('.')[0]
      # if save_flow:
      #    flowdir = 'results/' + testname + '_flow'
      #    if not isdir(flowdir):
      #       mkdir(flowdir)
      #    flowcount = 0
      while True:
         try:
            sample = next(self.testDataiter)
         except StopIteration:
            break

         motions, flow = self.tartan.test_batch(sample)
         motionlist.extend(motions)

      poselist = ses2poses_quat(np.array(motionlist))

      # calculate ATE, RPE, KITTI-RPE
      # if pose_file.endswith('.txt'):
      #    evaluator = TartanAirEvaluator()
      #    results = evaluator.evaluate_one_trajectory(args.pose_file, poselist, scale=True,
      #                                                kittitype=(datastr == 'kitti'))
      #    if datastr == 'euroc':
      #       print("==> ATE: %.4f" % (results['ate_score']))
      #    else:
      #       print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" % (
      #       results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))
      #
      #    # save results and visualization
      #    plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname='results/' + testname + '.png',
      #              title='ATE %.4f' % (results['ate_score']))
      #    np.savetxt('results/' + testname + '.txt', results['est_aligned'])
      # else:
      np.savetxt('results/' + testname + '.txt', poselist)


if __name__ == "__main__":
   tartan = TantanOdometry()
   tartan.visual_odometry()
