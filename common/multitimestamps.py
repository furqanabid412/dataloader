import numpy as np

from common.laserscan import *
import os
import torch

class MultiTimeStamps():
  """Class that process the multi-time stamps"""
  def __init__(self, root,  # directory where data is
                 sequences,  # sequences for this data (e.g. [1,3,4,6])
                 color_map,  # colors dict bgr (e.g 10: [255, 0, 0])
                 sensor,  # sensor to parse scans from
                 gt=False,
                 multi_gt=True,
                 max_points=150000,   # max number of points present in dataset
              ):

    self.root = root
    self.sequences = sequences
    self.color_map = color_map
    self.sensor = sensor
    self.max_points = max_points

    self.sensor_img_means = sensor["img_means"]
    self.sensor_img_stds = sensor["img_stds"]

    self.gt = gt
    self.multi_gt =multi_gt

    self.calibrations = []
    self.times = []
    self.poses = []

    # fill in with names, checking that all sequences are complete

    for seq in self.sequences:
      # to string
      seq = '{0:02d}'.format(int(seq))

      # print("parsing seq {} poses".format(seq))
      self.calibrations.append(self.parse_calibration(os.path.join(self.root, seq, "calib.txt")))  # read caliberation
      self.times.append(np.loadtxt(os.path.join(self.root, seq, 'times.txt'), dtype=np.float32))  # read times
      poses_f64 = self.parse_poses(os.path.join(self.root, seq, 'poses.txt'), self.calibrations[-1])
      self.poses.append([pose.astype(np.float32) for pose in poses_f64])  # read poses

    if self.gt:
      self.scan = SemLaserScan(self.color_map, project=True, H=self.sensor["img_prop"]["height"],
                               W=self.sensor["img_prop"]["width"],
                               fov_up=self.sensor["fov_up"], fov_down=self.sensor["fov_down"])
    else:
      self.scan = SemLaserScan(project=True, H=self.sensor["img_prop"]["height"],
                               W=self.sensor["img_prop"]["width"],
                               fov_up=self.sensor["fov_up"], fov_down=self.sensor["fov_down"])


  def parse_calibration(self, filename):
    calib = {}
    calib_file = open(filename)
    for line in calib_file:
      key, content = line.strip().split(":")
      values = [float(v) for v in content.strip().split()]
      pose = np.zeros((4, 4))
      pose[0, 0:4] = values[0:4]
      pose[1, 0:4] = values[4:8]
      pose[2, 0:4] = values[8:12]
      pose[3, 3] = 1.0
      calib[key] = pose
    calib_file.close()
    return calib

  def parse_poses(self, filename, calibration):
    file = open(filename)
    poses = []
    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)
    for line in file:
      values = [float(v) for v in line.strip().split()]
      pose = np.zeros((4, 4))
      pose[0, 0:4] = values[0:4]
      pose[1, 0:4] = values[4:8]
      pose[2, 0:4] = values[8:12]
      pose[3, 3] = 1.0
      poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
    return poses


  def multi_time_frames(self,scan_paths, # list of scan paths in the time frames : t,t-1,t-2....,t-n
                            label_paths, # list of label paths in the time frames : t,t-1,t-2....,t-n
                            seq,frame
                       ):

    # make sure scan_paths and label_paths is a list
    assert (isinstance(scan_paths, list))
    assert (isinstance(scan_paths, list))

    if self.gt:
      assert (len(scan_paths) == len(scan_paths))


    multi_time_proj = []
    multi_time_gt = []

    pose0 = self.poses[seq][frame]  # reference pose

    for timeframe in range(len(scan_paths)):

      if frame - timeframe >= 0:
        curr_frame = frame - timeframe
      else:
        curr_frame = 0

      curr_pose = self.poses[seq][curr_frame]

      # check whether a coordinate transformation is needed or not

      if timeframe==0 or np.array_equal(pose0,curr_pose):
        ego_motion=False
      else :
        ego_motion=True

      # opening the scan(and label) file
      self.scan.open_scan(scan_paths[timeframe],pose0, curr_pose,ego_motion=ego_motion)

      # combining the projections

      proj_range = np.copy(self.scan.proj_range)
      proj_range = np.expand_dims(proj_range,axis=0)
      proj_xyz = np.copy(self.scan.proj_xyz)
      proj_xyz = np.rollaxis(proj_xyz, 2)
      proj_remission = np.copy(self.scan.proj_remission)
      proj_remission = np.expand_dims(proj_remission, axis=0)
      proj = np.concatenate((proj_range, proj_xyz, proj_remission), axis=0)

      # re-scaling the projection
      img_mean = np.reshape(self.sensor_img_means,(5,1,1))
      img_stds = np.reshape(self.sensor_img_stds,(5,1,1))
      proj = ( proj - img_mean) / img_stds
      proj_mask = np.copy(self.scan.proj_mask)
      proj = proj * proj_mask.astype(np.float)

      if self.gt:
        # If training : then get the labels too

        self.scan.open_label(label_paths[timeframe])
        proj_labels = np.copy(self.scan.proj_sem_label)
        proj_labels = proj_labels * proj_mask
        if self.multi_gt:
          multi_time_gt.append(np.copy(proj_labels)) # for-later : not working now
      else:
        proj_labels = []

      # appending the multi projection list

      multi_time_proj.append(np.copy(proj))

      if timeframe == 0:
        # other params for post processing - only for frame "t"
        original_points = np.copy(self.scan.points)
        point_range = np.copy(self.scan.unproj_range)
        if self.gt:
          original_labels = np.copy(self.scan.sem_label)
          proj_gt = np.copy(proj_labels)
        else:
          original_labels = []
          proj_gt = []

        pixel_u=np.copy(self.scan.proj_x)
        pixel_v=np.copy(self.scan.proj_y)

    return multi_time_proj,proj_gt,original_points,point_range,original_labels,pixel_u,pixel_v







class MultiRange():
  """Class that process the multi-ranges"""

  def __init__(self, intervals,  # list of ranges
               nclasses,
               gt=False,
               multi_gt=True,
            ):
    self.intervals=intervals
    self.gt = gt
    self.multi_gt=multi_gt
    self.nclasses=nclasses
    self.only_once_gt = True


  def apply_range_limits(self, lower, upper):

    # rem = np.zeros((self.proj_H, self.proj_W), dtype=np.float32)
    # proj_range = np.zeros((self.proj_H, self.proj_W), dtype=np.float32)
    # proj_xyz = np.zeros((self.proj_H, self.proj_W, 3), dtype=np.float32)
    proj_sem_label = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)

    # getting a mask [True,False] for the range limits
    mask = np.logical_and(self.proj_range > lower, self.proj_range < upper)
    inverse_mask = np.logical_not(mask)

    rem = np.multiply(self.proj_remission, mask)
    proj_range = np.multiply(self.proj_range, mask)
    proj_xyz = np.multiply(self.proj_xyz, mask)

    # re-projecting -1
    rem[inverse_mask] = -1.0
    proj_range[inverse_mask] = -1.0
    proj_xyz[0][inverse_mask] = -1.0
    proj_xyz[1][inverse_mask] = -1.0
    proj_xyz[2][inverse_mask] = -1.0

    # self.only_once_gt is to perform multi ground truth only once
    # while range-input projection will be done for all time frames
    # but it will be done once for the labels

    if self.multi_gt and self.only_once_gt:
      proj_sem_label = self.proj_sem_label
      proj_sem_label = np.multiply(proj_sem_label, mask)
      # proj_sem_label[inverse_mask] = self.nclasses

    projection_cat = np.concatenate((np.expand_dims(rem, axis=0), np.expand_dims(proj_range, axis=0), proj_xyz),
                                    axis=0)

    return projection_cat, proj_sem_label


  def do_multi_range(self,proj,proj_gt):
    # only works when the length of intervals > 1
    # i.e. [r1,r2] or [r1,r2,r3] or so on

    self.proj_range = proj[0]
    self.proj_remission = proj [4]
    self.proj_xyz = proj [1:4]
    self.channels,self.proj_H,self.proj_W = np.shape(proj)

    if self.gt :
      self.proj_sem_label = proj_gt

    # get the minimum and maxium value of the range
    range_exclude_back = self.proj_range[self.proj_range>-1]
    min_depth = np.amin(range_exclude_back)
    max_depth = np.amax(range_exclude_back)
    max_iter, min_depth, max_depth = len(self.intervals) + 1, min_depth, max_depth

    # Divide the depth into multiple ranges
    concated_proj_range = np.zeros((0, self.channels, self.proj_H, self.proj_W), dtype=np.float32)

    if self.gt:
      concated_proj_label = np.zeros((0, self.proj_H, self.proj_W), dtype=np.int32)

    for index in range(max_iter):
      if index == 0:
        range_output, label_output = self.apply_range_limits(min_depth, self.intervals[index])  # range_output = C*H*W ; label_output = 1*H*W
      elif index == max_iter - 1:
        range_output, label_output = self.apply_range_limits(self.intervals[index - 1], max_depth)
      else:
        range_output, label_output = self.apply_range_limits(self.intervals[index - 1], self.intervals[index])

      concated_proj_range = np.concatenate((concated_proj_range, np.expand_dims(range_output, 0)), axis=0)

      if self.gt and self.multi_gt and self.only_once_gt:
        concated_proj_label = np.concatenate((concated_proj_label, np.expand_dims(label_output, 0)), axis=0)
      else:
        concated_proj_label=[]

    if self.gt and self.multi_gt:
      self.only_once_gt = False

    return concated_proj_range, concated_proj_label







