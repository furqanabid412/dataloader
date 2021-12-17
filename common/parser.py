import os
import numpy as np
import torch
from torch.utils.data import Dataset
from common.laserscan import *
from common.multitimestamps import *

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


class SemanticKitti(Dataset):

  def __init__(self, root,    # directory where data is
               sequences,     # sequences for this data (e.g. [1,3,4,6])
               labels,        # label dict: (e.g 10: "car")
               color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
               learning_map,  # classes to learn (0 to N-1 for xentropy)
               learning_map_inv,    # inverse of previous (recover labels)
               sensor,              # sensor to parse scans from
               multi_proj,        # multi projection parameters
               max_points=150000,   # max number of points present in dataset
               gt=False,       # send ground truth?
               multi_gt=False,
               ):
    # copying the params to self instance

    self.root = os.path.join(root, "sequences")
    self.sequences = sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.sensor_img_H = sensor["img_prop"]["height"]
    self.sensor_img_W = sensor["img_prop"]["width"]

    self.intervals = multi_proj["intervals"]
    self.timeframe = multi_proj["timeframes"]
    self.sensor_fov_up = sensor["fov_up"]
    self.sensor_fov_down = sensor["fov_down"]
    self.max_points = max_points
    self.multi_gt=multi_gt
    self.gt = gt

    # get number of classes (can't be len(self.learning_map) because there
    # are multiple repeated entries, so the number that matters is how many
    # there are for the xentropy)

    self.nclasses = len(self.learning_map_inv)

    # sanity checks

    # make sure directory exists
    if os.path.isdir(self.root):
      print("Sequences folder exists! Using sequences from %s" % self.root)
    else:
      raise ValueError("Sequences folder doesn't exist! Exiting...")

    # make sure labels is a dict
    assert(isinstance(self.labels, dict))

    # make sure color_map is a dict
    assert(isinstance(self.color_map, dict))

    # make sure learning_map is a dict
    assert(isinstance(self.learning_map, dict))

    # make sure sequences is a list
    assert(isinstance(self.sequences, list))

    # placeholder for filenames
    self.scan_files = []
    self.label_files = []

    self.frames_in_a_seq=[]
    frames_in_a_seq = []

    # fill in with names, checking that all sequences are complete
    for seq in self.sequences:
      # to string
      seq = '{0:02d}'.format(int(seq))

      # print("parsing seq {}".format(seq))

      # get paths for each
      scan_path = os.path.join(self.root, seq, "velodyne")
      label_path = os.path.join(self.root, seq, "labels")

      # get files
      scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
      label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(label_path)) for f in fn if is_label(f)]

      # sort for correspondance
      scan_files.sort()
      label_files.sort()

      # check all scans have labels
      if self.gt:
        assert(len(scan_files) == len(label_files))

      # append list
      self.scan_files.append(scan_files)
      self.label_files.append(label_files)
      frames_in_a_seq.append(len(scan_files))



    self.frames_in_a_seq = np.array(frames_in_a_seq).cumsum()

    # print("Using {} scans from sequences {}".format(len(self.scan_files),
    #                                                 self.sequences))



  def get_seq_and_frame(self, index):
    # function takes index and convert it to seq and frame number

    if index < self.frames_in_a_seq[0]:
      return 0, index

    else:
      seq_count = len(self.frames_in_a_seq)
      for i in range(seq_count):
        fr = index + 1
        if i < seq_count - 1 and self.frames_in_a_seq[i] < fr and self.frames_in_a_seq[i + 1] > fr:
          # print("here")
          return i + 1, index - self.frames_in_a_seq[i]

        elif i < seq_count - 1 and self.frames_in_a_seq[i] == fr:
          return i, index - self.frames_in_a_seq[i - 1]

        elif i < seq_count - 1 and fr == self.frames_in_a_seq[-1]:
          return seq_count - 1, index - self.frames_in_a_seq[-2]


  @staticmethod
  def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]

  def __len__(self):
    return self.frames_in_a_seq[-1]


  def __getitem__(self, index):

    seq,frame = self.get_seq_and_frame(index)

    scan_paths=[]
    label_paths=[]

    # get the list of filenames (scan and labels)
    # for multiple time frames

    for timeframe in range(self.timeframe):
      if frame - timeframe >= 0:
        curr_frame = frame - timeframe
      else:
        curr_frame = 0
      temp_scan_path = self.scan_files[seq][curr_frame]
      temp_label_path = self.label_files[seq][curr_frame]
      scan_paths.append(temp_scan_path)
      label_paths.append(temp_label_path)

    # multi time frames processing

    multi_scan=MultiTimeStamps(self.root,self.sequences,self.color_map,self.sensor,self.gt)

    multi_time_proj,proj_gt,original_points,point_range,\
    original_labels,pixel_u,pixel_v=multi_scan.multi_time_frames(scan_paths,label_paths,seq,frame)


    # intervals are in units of m , convert it into appropriate range

    intervals = np.copy(self.intervals)
    range_mean , range_stds = self.sensor["img_means"],self.sensor["img_means"]
    intervals = (intervals - range_mean[0]) / range_stds[0]

    # multi range processing

    multi_scan_2= MultiRange(intervals,nclasses=self.nclasses, gt=self.gt,multi_gt=self.multi_gt)
    multi_range_and_t_proj, multi_range_and_t_gt =[],[]


    for ind in range(len(multi_time_proj)):
      # run it for all time stamps
      proj=np.copy(multi_time_proj[ind])
      range_,gt_=multi_scan_2.do_multi_range(proj,proj_gt)
      multi_range_and_t_proj.append(range_)
      if ind==0:
        multi_range_and_t_gt=np.copy(gt_)

    # converting projection results into torch tensor
    proj_input=torch.tensor(multi_range_and_t_proj).clone()


    if self.gt:
      if self.multi_gt:
        proj_label = np.copy(multi_range_and_t_gt)
       # proj_label = torch.tensor(multi_range_and_t_gt).clone()
      else :
        proj_label = np.copy(proj_gt)
        # proj_label = torch.tensor(proj_gt).clone()

      # mapping classes
      proj_label = self.map(proj_label, self.learning_map)
      proj_label = torch.tensor(proj_label).clone()

    else :
      proj_label = torch.tensor([])

    # converting original point data ->  for post processing
    # Each of these are torch tensor with len=max no points i.e. 150,000

    total_points = original_points.shape[0]
    scan_points = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
    scan_points[:total_points] = torch.from_numpy(original_points)

    scan_range = torch.full([self.max_points], -1.0, dtype=torch.float)
    scan_range[:total_points] = torch.from_numpy(point_range)

    if self.gt:
      scan_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
      scan_labels[:total_points] = torch.from_numpy(original_labels)
    else:
      scan_labels = []


    point_to_pixel_u = torch.full([self.max_points], -1, dtype=torch.long)
    point_to_pixel_u[:total_points] = torch.from_numpy(pixel_u)

    point_to_pixel_v = torch.full([self.max_points], -1, dtype=torch.long)
    point_to_pixel_v[:total_points] = torch.from_numpy(pixel_v)



    return {"proj_input":proj_input,
            "proj_groundtruth":proj_label,
            "scan_points" : scan_points,
            "scan_range" : scan_range,
            "scan_labels" : scan_labels,
            "point_to_pixel_u" : point_to_pixel_u,
            "point_to_pixel_v" : point_to_pixel_v,
            }