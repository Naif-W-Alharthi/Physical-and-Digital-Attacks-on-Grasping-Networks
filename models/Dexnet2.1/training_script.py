"""
Script to train a convolutional neural network to predict grasp quality from images
Authors: Jeff Mahler

Adapted from AlexNet implementation by Michael Guerzhoy and Davi Frossard, 2016 (http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)
"""
import argparse
import copy
import cv2
import IPython
import json
import logging
import numpy as np
import cPickle as pkl
import os
import random
import scipy.misc as sm
import scipy.ndimage.filters as sf
import scipy.ndimage.morphology as snm
import scipy.stats as ss
import skimage.draw as sd
import signal
import sys
import shutil
import threading
import time
import urllib

import matplotlib.pyplot as plt
import tensorflow as tf

from core import YamlConfig
import core.utils as utils
from perception import ColorImage, DepthImage

# templates for file reading
# TODO: remove to a shared constants file
binary_im_tensor_template = 'binary_ims_raw'
depth_im_tensor_template = 'depth_ims_raw'
depth_im_table_tensor_template = 'depth_ims_raw_table'
binary_im_tf_tensor_template = 'binary_ims_tf'
color_im_tf_tensor_template = 'color_ims_tf'
gray_im_tf_tensor_template = 'gray_ims_tf'
depth_im_tf_tensor_template = 'depth_ims_tf'
color_im_tf_table_tensor_template = 'color_ims_tf_table'
depth_im_tf_table_tensor_template = 'depth_ims_tf_table'
gd_im_tf_table_tensor_template = 'gd_ims_tf_table'
rgbd_im_tf_table_tensor_template = 'rgbd_ims_tf_table'
table_mask_template = 'table_mask'
hand_poses_template = 'hand_poses'

SEED = 25794243
timeout_option = tf.RunOptions(timeout_in_ms=1000000)

# enum for image modalities
class ImageMode:
    BINARY = 'binary'
    DEPTH = 'depth'
    DEPTH_TABLE = 'depth_table'
    BINARY_TF = 'binary_tf'
    COLOR_TF = 'color_tf'
    GRAY_TF = 'gray_tf'
    DEPTH_TF = 'depth_tf'
    COLOR_TF_TABLE = 'color_tf_table'
    DEPTH_TF_TABLE = 'depth_tf_table'
    GD_TF_TABLE = 'gd_tf_table'
    RGBD_TF_TABLE = 'rgbd_tf_table'

# enum for training modes
class TrainingMode:
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'

# enums for input data modes
class InputDataMode:
    TF_IMAGE = 'tf_image'
    TF_IMAGE_PERSPECTIVE = 'tf_image_with_perspective'
    RAW_IMAGE = 'raw_image'
    RAW_IMAGE_PERSPECTIVE = 'raw_image_with_perspective'
    REGRASPING = 'regrasping'

# enums for preproc
class PreprocMode:
    NORMALIZATION = 'normalized'
    IZZYNET = 'izzynet'
    NONE = 'none'

# empty class for alexnet weights
class AlexNetWeights(object):
  def __init__(self):
    pass

def read_pose_data(pose_arr, input_data_mode):
    """ Read the pose data """
    if input_data_mode == InputDataMode.TF_IMAGE:
        return pose_arr[:,2:3]
    elif input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
        return np.c_[pose_arr[:,2:3], pose_arr[:,4:6]]
    elif input_data_mode == InputDataMode.RAW_IMAGE:
        return pose_arr[:,:4]
    elif input_data_mode == InputDataMode.RAW_IMAGE_PERSPECTIVE:
        return pose_arr[:,:6]
    elif input_data_mode == InputDataMode.REGRASPING:
        # depth, approach angle, and delta angle for reorientation
        return np.c_[pose_arr[:,2:3], pose_arr[:,4:5], pose_arr[:,6:7]]
    else:
        raise ValueError('Input data mode %s not supported' %(input_data_mode))

def class_error_rate(predictions, labels):
  """Return the classification error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == labels) /
      predictions.shape[0])

def reg_error_rate(predictions, labels):
    return np.mean((predictions - labels)**2)

def error_rate_in_batches(data_filenames, pose_filenames,
                          label_filenames, index_map, sess,
                          eval_prediction, pose_dim, data_mean, data_std, pose_mean,
                          pose_std, val_batch_size, num_categories,
                          image_mode, training_mode, preproc_mode,
                          min_metric, max_metric, metric_thresh, input_data_mode, num_tensor_channels):
  """Get all predictions for a dataset by running it in small batches."""
  error_rates = []
  data_filenames.sort(key = lambda x: int(x[-9:-4]))
  pose_filenames.sort(key = lambda x: int(x[-9:-4]))
  label_filenames.sort(key = lambda x: int(x[-9:-4]))
  for data_filename, pose_filename, label_filename in zip(data_filenames, pose_filenames, label_filenames):
    # load next file
    data = np.load(data_filename)['arr_0']
    poses = np.load(pose_filename)['arr_0']
    labels = np.load(label_filename)['arr_0']

    if len(data_mean.shape) == 0:
        data = (data - data_mean) / data_std
    else:
        for i in range(num_tensor_channels):
            data[:,:,:,i] = (data[:,:,:,i] - data_mean[i]) / data_std[i]
    poses = (poses - pose_mean) / pose_std

    data = data[index_map[data_filename],...]
    poses = read_pose_data(poses[index_map[data_filename],:], input_data_mode)
    labels = labels[index_map[data_filename],...]

    if training_mode == TrainingMode.REGRESSION:
        if preproc_mode == PreprocMode.NORMALIZATION:
            labels = (labels - min_metric) / (max_metric - min_metric)
        elif preproc_mode == PreprocMode.IZZYNET:
            labels = 2 * (labels - min_metric) / (max_metric - min_metric) - 1
    elif training_mode == TrainingMode.CLASSIFICATION:
        labels = 1 * (labels > metric_thresh)
        labels = labels.astype(np.uint8)

    # setup buffers
    size = data.shape[0]
    if size < val_batch_size:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, num_categories), dtype=np.float32)
    if training_mode == TrainingMode.REGRESSION:
        predictions = np.ndarray(shape=(size,1), dtype=np.float32)

    # make predictions
    for begin in xrange(0, size, val_batch_size):
      end = begin + val_batch_size
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={val_data_node: data[begin:end, ...],
                       val_poses_node: poses[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={val_data_node: data[-val_batch_size:, ...],
                       val_poses_node: poses[-val_batch_size:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]

    # get error rate
    if training_mode == TrainingMode.CLASSIFICATION:
        error_rates.append(class_error_rate(predictions, labels))
    else:
        error_rates.append(reg_error_rate(predictions, labels))
        
    # clean up
    del data
    del poses
    del labels

  # return average error rate over all files (assuming same size)
  return np.mean(error_rates)

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    """
    Convolution layer helper function
    From https://github.com/ethereon/caffe-tensorflow
    """
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])
    
def reduce_shape(shape):
    shape = [ x.value for x in shape[1:] ]
    f = lambda x, y: 1 if y is None else x * y
    return reduce(f, shape, 1)

def build_izzynet_weights(out_size, pose_dim, channels, im_height, im_width, image_mode, cfg):
    """ build izzynet! """
    # init pool size variables
    layer_height = im_height
    layer_width = im_width
    layer_channels = channels

    # conv1_1
    conv1_1_filt_dim = cfg['conv1_1']['filt_dim']
    conv1_1_num_filt = cfg['conv1_1']['num_filt']
    conv1_1_size = layer_height * layer_width * conv1_1_num_filt
    conv1_1_shape = [conv1_1_filt_dim, conv1_1_filt_dim, layer_channels, conv1_1_num_filt]

    conv1_1_num_inputs = conv1_1_filt_dim**2 * layer_channels
    conv1_1_std = np.sqrt(2.0 / (conv1_1_num_inputs))
    conv1_1W = tf.Variable(tf.truncated_normal(conv1_1_shape, stddev=conv1_1_std), name='conv1_1W')
    conv1_1b = tf.Variable(tf.truncated_normal([conv1_1_num_filt], stddev=conv1_1_std), name='conv1_1b')

    layer_height = layer_height / cfg['conv1_1']['pool_stride']
    layer_width = layer_width / cfg['conv1_1']['pool_stride']
    layer_channels = conv1_1_num_filt

    # conv1_2
    conv1_2_filt_dim = cfg['conv1_2']['filt_dim']
    conv1_2_num_filt = cfg['conv1_2']['num_filt']
    conv1_2_size = layer_height * layer_width * conv1_2_num_filt
    conv1_2_shape = [conv1_2_filt_dim, conv1_2_filt_dim, layer_channels, conv1_2_num_filt]

    conv1_2_num_inputs = conv1_2_filt_dim**2 * layer_channels
    conv1_2_std = np.sqrt(2.0 / (conv1_2_num_inputs))
    conv1_2W = tf.Variable(tf.truncated_normal(conv1_2_shape, stddev=conv1_2_std), name='conv1_2W')
    conv1_2b = tf.Variable(tf.truncated_normal([conv1_2_num_filt], stddev=conv1_2_std), name='conv1_2b')

    layer_height = layer_height / cfg['conv1_2']['pool_stride']
    layer_width = layer_width / cfg['conv1_2']['pool_stride']
    layer_channels = conv1_2_num_filt

    # conv2_1
    conv2_1_filt_dim = cfg['conv2_1']['filt_dim']
    conv2_1_num_filt = cfg['conv2_1']['num_filt']
    conv2_1_size = layer_height * layer_width * conv2_1_num_filt
    conv2_1_shape = [conv2_1_filt_dim, conv2_1_filt_dim, layer_channels, conv2_1_num_filt]

    conv2_1_num_inputs = conv2_1_filt_dim**2 * layer_channels
    conv2_1_std = np.sqrt(2.0 / (conv2_1_num_inputs))
    conv2_1W = tf.Variable(tf.truncated_normal(conv2_1_shape, stddev=conv2_1_std), name='conv2_1W')
    conv2_1b = tf.Variable(tf.truncated_normal([conv2_1_num_filt], stddev=conv2_1_std), name='conv2_1b')

    layer_height = layer_height / cfg['conv2_1']['pool_stride']
    layer_width = layer_width / cfg['conv2_1']['pool_stride']
    layer_channels = conv2_1_num_filt

    # conv2_2
    conv2_2_filt_dim = cfg['conv2_2']['filt_dim']
    conv2_2_num_filt = cfg['conv2_2']['num_filt']
    conv2_2_size = layer_height * layer_width * conv2_2_num_filt
    conv2_2_shape = [conv2_2_filt_dim, conv2_2_filt_dim, layer_channels, conv2_2_num_filt]

    conv2_2_num_inputs = conv2_2_filt_dim**2 * layer_channels
    conv2_2_std = np.sqrt(2.0 / (conv2_2_num_inputs))
    conv2_2W = tf.Variable(tf.truncated_normal(conv2_2_shape, stddev=conv2_2_std), name='conv2_2W')
    conv2_2b = tf.Variable(tf.truncated_normal([conv2_2_num_filt], stddev=conv2_2_std), name='conv2_2b')

    layer_height = layer_height / cfg['conv2_2']['pool_stride']
    layer_width = layer_width / cfg['conv2_2']['pool_stride']
    layer_channels = conv2_2_num_filt

    use_conv3 = False
    if 'conv3_1' in cfg.keys():
        use_conv3 = True

    if use_conv3:
        # conv3_1
        conv3_1_filt_dim = cfg['conv3_1']['filt_dim']
        conv3_1_num_filt = cfg['conv3_1']['num_filt']
        conv3_1_size = layer_height * layer_width * conv3_1_num_filt
        conv3_1_shape = [conv3_1_filt_dim, conv3_1_filt_dim, layer_channels, conv3_1_num_filt]
        
        conv3_1_num_inputs = conv3_1_filt_dim**2 * layer_channels
        conv3_1_std = np.sqrt(2.0 / (conv3_1_num_inputs))
        conv3_1W = tf.Variable(tf.truncated_normal(conv3_1_shape, stddev=conv3_1_std), name='conv3_1W')
        conv3_1b = tf.Variable(tf.truncated_normal([conv3_1_num_filt], stddev=conv3_1_std), name='conv3_1b')
        
        layer_height = layer_height / cfg['conv3_1']['pool_stride']
        layer_width = layer_width / cfg['conv3_1']['pool_stride']
        layer_channels = conv3_1_num_filt

        # conv3_2
        conv3_2_filt_dim = cfg['conv3_2']['filt_dim']
        conv3_2_num_filt = cfg['conv3_2']['num_filt']
        conv3_2_size = layer_height * layer_width * conv3_2_num_filt
        conv3_2_shape = [conv3_2_filt_dim, conv3_2_filt_dim, layer_channels, conv3_2_num_filt]
        
        conv3_2_num_inputs = conv3_2_filt_dim**2 * layer_channels
        conv3_2_std = np.sqrt(2.0 / (conv3_2_num_inputs))
        conv3_2W = tf.Variable(tf.truncated_normal(conv3_2_shape, stddev=conv3_2_std), name='conv3_2W')
        conv3_2b = tf.Variable(tf.truncated_normal([conv3_2_num_filt], stddev=conv3_2_std), name='conv3_2b')
        
        layer_height = layer_height / cfg['conv3_2']['pool_stride']
        layer_width = layer_width / cfg['conv3_2']['pool_stride']
        layer_channels = conv3_2_num_filt

    # fc3
    fc3_in_size = conv2_2_size
    if use_conv3:
        fc3_in_size = conv3_2_size
    fc3_out_size = cfg['fc3']['out_size']
    fc3_std = np.sqrt(2.0 / fc3_in_size)
    fc3W = tf.Variable(tf.truncated_normal([fc3_in_size, fc3_out_size], stddev=fc3_std), name='fc3W')
    fc3b = tf.Variable(tf.truncated_normal([fc3_out_size], stddev=fc3_std), name='fc3b')

    # pc1
    pc1_in_size = pose_dim
    pc1_out_size = cfg['pc1']['out_size']

    pc1_std = np.sqrt(2.0 / pc1_in_size)
    pc1W = tf.Variable(tf.truncated_normal([pc1_in_size, pc1_out_size],
                                        stddev=pc1_std,
                                        seed=SEED), name='pc1W')
    pc1b = tf.Variable(tf.truncated_normal([pc1_out_size],
                                        stddev=pc1_std,
                                        seed=SEED), name='pc1b')

    # pc2
    pc2_in_size = pc1_out_size
    pc2_out_size = cfg['pc2']['out_size']

    if pc2_out_size > 0:
        pc2_std = np.sqrt(2.0 / pc2_in_size)
        pc2W = tf.Variable(tf.truncated_normal([pc2_in_size, pc2_out_size],
                                               stddev=pc2_std,
                                               seed=SEED), name='pc2W')
        pc2b = tf.Variable(tf.truncated_normal([pc2_out_size],
                                               stddev=pc2_std,
                                               seed=SEED), name='pc2b')

    # fc4
    fc4_im_in_size = fc3_out_size
    if pc2_out_size == 0:
        fc4_pose_in_size = pc1_out_size
    else:
        fc4_pose_in_size = pc2_out_size
    fc4_out_size = cfg['fc4']['out_size']
    fc4_std = np.sqrt(2.0 / (fc4_im_in_size + fc4_pose_in_size))
    fc4W_im = tf.Variable(tf.truncated_normal([fc4_im_in_size, fc4_out_size], stddev=fc4_std), name='fc4W_im')
    fc4W_pose = tf.Variable(tf.truncated_normal([fc4_pose_in_size, fc4_out_size], stddev=fc4_std), name='fc4W_pose')
    fc4b = tf.Variable(tf.truncated_normal([fc4_out_size], stddev=fc4_std), name='fc4b')

    # fc5
    fc5_in_size = fc4_out_size
    fc5_out_size = cfg['fc5']['out_size']
    fc5_std = np.sqrt(2.0 / (fc5_in_size))
    fc5W = tf.Variable(tf.truncated_normal([fc5_in_size, fc5_out_size], stddev=fc5_std), name='fc5W')
    fc5b = tf.Variable(tf.constant(0.0, shape=[fc5_out_size]), name='fc5b')

    # make return object
    weights = AlexNetWeights()
    weights.conv1_1W = conv1_1W
    weights.conv1_1b = conv1_1b
    weights.conv1_2W = conv1_2W
    weights.conv1_2b = conv1_2b
    weights.conv2_1W = conv2_1W
    weights.conv2_1b = conv2_1b
    weights.conv2_2W = conv2_2W
    weights.conv2_2b = conv2_2b
    
    if use_conv3:
        weights.conv3_1W = conv3_1W
        weights.conv3_1b = conv3_1b
        weights.conv3_2W = conv3_2W
        weights.conv3_2b = conv3_2b

    weights.fc3W = fc3W
    weights.fc3b = fc3b
    weights.fc4W_im = fc4W_im
    weights.fc4W_pose = fc4W_pose
    weights.fc4b = fc4b
    weights.fc5W = fc5W
    weights.fc5b = fc5b
    weights.pc1W = pc1W
    weights.pc1b = pc1b

    if pc2_out_size > 0:
        weights.pc2W = pc2W
        weights.pc2b = pc2b
    return weights    

def build_izzynet_weights_pretrained(reader, channels, im_height, im_width, pose_dim, cfg,
                                     reinit_pc1=False, reinit_pc2=False, reinit_fc3=False,
                                     reinit_fc4=False, reinit_fc5=False):
    """ build izzynet! """
    use_conv0 = False
    if 'conv0_1' in cfg.keys():
        use_conv0 = True

    use_conv3 = False
    if 'conv3_1' in cfg.keys():
        use_conv3 = True

    use_pc2 = False
    if cfg['pc2']['out_size'] > 0:
        use_pc2 = True

    weights = AlexNetWeights()
    weights.conv1_1W = tf.Variable(reader.get_tensor("conv1_1W"), name='conv1_1W') 
    weights.conv1_1b = tf.Variable(reader.get_tensor("conv1_1b"), name='conv1_1b')
    weights.conv1_2W = tf.Variable(reader.get_tensor("conv1_2W"), name='conv1_2W') 
    weights.conv1_2b = tf.Variable(reader.get_tensor("conv1_2b"), name='conv1_2b')
    weights.conv2_1W = tf.Variable(reader.get_tensor("conv2_1W"), name='conv2_1W') 
    weights.conv2_1b = tf.Variable(reader.get_tensor("conv2_1b"), name='conv2_1b')
    weights.conv2_2W = tf.Variable(reader.get_tensor("conv2_2W"), name='conv2_2W') 
    weights.conv2_2b = tf.Variable(reader.get_tensor("conv2_2b"), name='conv2_2b')

    if use_conv3:
        weights.conv3_1W = tf.Variable(reader.get_tensor("conv3_1W"), name='conv3_1W') 
        weights.conv3_1b = tf.Variable(reader.get_tensor("conv3_1b"), name='conv3_1b')
        weights.conv3_2W = tf.Variable(reader.get_tensor("conv3_2W"), name='conv3_2W') 
        weights.conv3_2b = tf.Variable(reader.get_tensor("conv3_2b"), name='conv3_2b')
        
    if reinit_pc1:
        pc1_in_size = pose_dim
        pc1_out_size = cfg['pc1']['out_size']

        pc1_std = np.sqrt(2.0 / pc1_in_size)
        weights.pc1W = tf.Variable(tf.truncated_normal([pc1_in_size, pc1_out_size],
                                                       stddev=pc1_std,
                                                       seed=SEED), name='pc1W')
        weights.pc1b = tf.Variable(tf.truncated_normal([pc1_out_size],
                                                       stddev=pc1_std,
                                                       seed=SEED), name='pc1b')
    else:
        weights.pc1W = tf.Variable(reader.get_tensor("pc1W"), name='pc1W')
        weights.pc1b = tf.Variable(reader.get_tensor("pc1b"), name='pc1b')
        pc1_out_size = weights.pc1b.shape[0].value

    pc2_out_size = cfg['pc2']['out_size']
    if use_pc2:
        pc2_in_size = pc1_out_size
        if reinit_pc2:
            pc2_std = np.sqrt(2.0 / pc2_in_size)
            weights.pc2W = tf.Variable(tf.truncated_normal([pc2_in_size, pc2_out_size],
                                                           stddev=pc2_std,
                                                           seed=SEED), name='pc2W')
            weights.pc2b = tf.Variable(tf.truncated_normal([pc2_out_size],
                                                           stddev=pc2_std,
                                                           seed=SEED), name='pc2b')
        else:
            weights.pc2W = tf.Variable(reader.get_tensor("pc2W"), name='pc2W')
            weights.pc2b = tf.Variable(reader.get_tensor("pc2b"), name='pc2b')
            pc2_out_size = weights.pc2b.shape[0].value

    if reinit_fc3:
        loaded_fc3W = tf.Variable(reader.get_tensor("fc3W"), name='loaded_fc32W')
        fc3_in_size = loaded_fc3W.shape[0].value
        fc3_out_size = cfg['fc3']['out_size']
        fc3_std = np.sqrt(2.0 / fc3_in_size)
        weights.fc3W = tf.Variable(tf.truncated_normal([fc3_in_size, fc3_out_size], stddev=fc3_std), name='fc3W')
        weights.fc3b = tf.Variable(tf.truncated_normal([fc3_out_size], stddev=fc3_std), name='fc3b')
    else:
        weights.fc3W = tf.Variable(reader.get_tensor("fc3W"), name='fc3W')
        weights.fc3b = tf.Variable(reader.get_tensor("fc3b"), name='fc3b')
        fc3_out_size = weights.fc3b.shape[0].value

    if reinit_fc4:
        fc4_im_in_size = fc3_out_size
        if pc2_out_size == 0:
            fc4_pose_in_size = pc1_out_size
        else:
            fc4_pose_in_size = pc2_out_size
        fc4_out_size = cfg['fc4']['out_size']
        fc4_std = np.sqrt(2.0 / (fc4_im_in_size + fc4_pose_in_size))
        weights.fc4W_im = tf.Variable(tf.truncated_normal([fc4_im_in_size, fc4_out_size], stddev=fc4_std), name='fc4W_im')
        weights.fc4W_pose = tf.Variable(tf.truncated_normal([fc4_pose_in_size, fc4_out_size], stddev=fc4_std), name='fc4W_pose')
        weights.fc4b = tf.Variable(tf.truncated_normal([fc4_out_size], stddev=fc4_std), name='fc4b')
    else:
        weights.fc4W_im = tf.Variable(reader.get_tensor("fc4W_im"), name='fc4W_im')
        weights.fc4W_pose = tf.Variable(reader.get_tensor("fc4W_pose"), name='fc4W_pose')
        weights.fc4b = tf.Variable(reader.get_tensor("fc4b"), name='fc4b')
        fc4_out_size = weights.fc4b.shape[0].value

    if reinit_fc5:
        fc5_in_size = fc4_out_size
        fc5_out_size = cfg['fc5']['out_size']
        fc5_std = np.sqrt(2.0 / (fc5_in_size))
        weights.fc5W = tf.Variable(tf.truncated_normal([fc5_in_size, fc5_out_size], stddev=fc5_std), name='fc5W')
        weights.fc5b = tf.Variable(tf.constant(0.0, shape=[fc5_out_size]), name='fc5b')
    else:
        weights.fc5W = tf.Variable(reader.get_tensor("fc5W"), name='fc5W')
        weights.fc5b = tf.Variable(reader.get_tensor("fc5b"), name='fc5b')

    if use_conv0:
        eps = 0.01
        nu = 0.0001

        # init pool size variables
        layer_height = im_height
        layer_width = im_width
        layer_channels = channels
        
        # conv0_1
        conv0_1_filt_dim = cfg['conv0_1']['filt_dim']
        conv0_1_num_filt = cfg['conv0_1']['num_filt']
        conv0_1_size = layer_height * layer_width * conv0_1_num_filt
        conv0_1_shape = [conv0_1_filt_dim, conv0_1_filt_dim, layer_channels, conv0_1_num_filt]
        
        conv0_1_num_inputs = conv0_1_filt_dim**2 * layer_channels
        conv0_1_std = nu
        weights.conv0_1W = tf.Variable(tf.truncated_normal(conv0_1_shape, stddev=conv0_1_std), name='conv0_1W')
        weights.conv0_1b = tf.Variable(tf.truncated_normal([conv0_1_num_filt], stddev=conv0_1_std), name='conv0_1b')

        layer_height = layer_height / cfg['conv0_1']['pool_stride']
        layer_width = layer_width / cfg['conv0_1']['pool_stride']
        layer_channels = conv0_1_num_filt

        # conv0_2
        conv0_2_filt_dim = cfg['conv0_2']['filt_dim']
        conv0_2_num_filt = cfg['conv0_2']['num_filt']
        conv0_2_size = layer_height * layer_width * conv0_2_num_filt
        conv0_2_shape = [conv0_2_filt_dim, conv0_2_filt_dim, layer_channels, conv0_2_num_filt]
        
        conv0_2_num_inputs = conv0_2_filt_dim**2 * layer_channels
        conv0_2_std = np.sqrt(2.0 / (conv0_2_num_inputs))
        weights.conv0_2W = tf.Variable(tf.truncated_normal(conv0_2_shape, stddev=conv0_2_std), name='conv0_2W')
        weights.conv0_2b = tf.Variable(tf.truncated_normal([conv0_2_num_filt], stddev=conv0_2_std), name='conv0_2b')

        layer_height = layer_height / cfg['conv0_2']['pool_stride']
        layer_width = layer_width / cfg['conv0_2']['pool_stride']
        layer_channels = conv0_2_num_filt
        
        # conv0_3
        conv0_3_filt_dim = cfg['conv0_3']['filt_dim']
        conv0_3_num_filt = cfg['conv0_3']['num_filt']
        conv0_3_size = layer_height * layer_width * conv0_3_num_filt
        conv0_3_shape = [conv0_3_filt_dim, conv0_3_filt_dim, layer_channels, conv0_3_num_filt]
        
        conv0_3_num_inputs = conv0_3_filt_dim**2 * layer_channels
        conv0_3_std = np.sqrt(2.0 / (conv0_3_num_inputs))
        weights.conv0_3W = tf.Variable(tf.truncated_normal(conv0_3_shape, stddev=conv0_3_std), name='conv0_3W')
        weights.conv0_3b = tf.Variable(tf.truncated_normal([conv0_3_num_filt], stddev=conv0_3_std), name='conv0_3b')

    return weights    

def build_izzynet(data_node, poses_node, weights, im_height, im_width, cfg, drop_fc3=False, drop_fc4=False):
    # normalization constants
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0

    image_node = data_node

    use_conv0 = False
    if 'conv0_1' in cfg.keys():
        use_conv0 = True

    # setup padding
    paddings = [[0,0], [1,1], [1,1], [0,0]]

    if use_conv0:
        # resize
        tiny_images = tf.image.resize_images(data_node, im_height/2, im_width/2)

        # pad
        p = int(cfg['conv0_1']['filt_dim'] / 2)
        padded_data_node = data_node
        for i in range(p):
            padded_data_node = tf.pad(padded_data_node, paddings, 'SYMMETRIC')

        # conv0_1
        conv0_1h = tf.nn.relu(tf.nn.conv2d(padded_data_node, weights.conv0_1W, strides=[1,1,1,1], padding='VALID') + weights.conv0_1b)
        if cfg['conv0_1']['norm']:
            conv0_1h = tf.nn.local_response_normalization(conv0_1h,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)
        pool0_1_size = cfg['conv0_1']['pool_size']
        pool0_1_stride = cfg['conv0_1']['pool_stride']
        pool0_1 = tf.nn.max_pool(conv0_1h,
                                 ksize=[1, pool0_1_size, pool0_1_size, 1],
                                 strides=[1, pool0_1_stride, pool0_1_stride, 1],
                                 padding='SAME') 
        conv0_1_num_nodes = reduce_shape(pool0_1.get_shape())
        conv0_1_flat = tf.reshape(pool0_1, [-1, conv0_1_num_nodes])

        # conv0_2
        p = int(cfg['conv0_2']['filt_dim'] / 2)
        padded_data_node = pool0_1
        for i in range(p):
            padded_data_node = tf.pad(padded_data_node, paddings, 'SYMMETRIC')

        conv0_2h = tf.nn.relu(tf.nn.conv2d(padded_data_node, weights.conv0_2W, strides=[1,1,1,1], padding='VALID') + weights.conv0_2b)
        if cfg['conv0_2']['norm']:
            conv0_2h = tf.nn.local_response_normalization(conv0_2h,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)
        pool0_2_size = cfg['conv0_2']['pool_size']
        pool0_2_stride = cfg['conv0_2']['pool_stride']
        pool0_2 = tf.nn.max_pool(conv0_2h,
                                 ksize=[1, pool0_2_size, pool0_2_size, 1],
                                 strides=[1, pool0_2_stride, pool0_2_stride, 1],
                                 padding='SAME') 
        conv0_2_num_nodes = reduce_shape(pool0_2.get_shape())
        conv0_2_flat = tf.reshape(pool0_2, [-1, conv0_2_num_nodes])

        # conv0_3
        p = int(cfg['conv0_3']['filt_dim'] / 2)
        padded_data_node = pool0_2
        for i in range(p):
            padded_data_node = tf.pad(padded_data_node, paddings, 'SYMMETRIC')

        conv0_3h = tf.nn.conv2d(padded_data_node, weights.conv0_3W, strides=[1,1,1,1], padding='VALID') + weights.conv0_3b
        if cfg['conv0_3']['norm']:
            conv0_3h = tf.nn.local_response_normalization(conv0_3h,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)
        pool0_3_size = cfg['conv0_3']['pool_size']
        pool0_3_stride = cfg['conv0_3']['pool_stride']
        pool0_3 = tf.nn.max_pool(conv0_3h,
                                 ksize=[1, pool0_3_size, pool0_3_size, 1],
                                 strides=[1, pool0_3_stride, pool0_3_stride, 1],
                                 padding='SAME') 
        conv0_3_num_nodes = reduce_shape(pool0_3.get_shape())
        conv0_3_flat = tf.reshape(pool0_3, [-1, conv0_3_num_nodes])

        pool0_3 = tf.image.resize_images(pool0_3, im_height, im_width)

        image_node = data_node + pool0_3
    else:
        # pad
        p = int(cfg['conv1_1']['filt_dim'] / 2)
        padded_data_node = image_node
        for i in range(p):
            padded_data_node = tf.pad(padded_data_node, paddings, 'SYMMETRIC')
        image_node = padded_data_node

    # conv1_1
    conv1_1h = tf.nn.relu(tf.nn.conv2d(image_node, weights.conv1_1W, strides=[1,1,1,1], padding='VALID') + weights.conv1_1b)
    if cfg['conv1_1']['norm']:
        conv1_1h = tf.nn.local_response_normalization(conv1_1h,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
    pool1_1_size = cfg['conv1_1']['pool_size']
    pool1_1_stride = cfg['conv1_1']['pool_stride']
    pool1_1 = tf.nn.max_pool(conv1_1h,
                             ksize=[1, pool1_1_size, pool1_1_size, 1],
                             strides=[1, pool1_1_stride, pool1_1_stride, 1],
                             padding='SAME') 
    conv1_1_num_nodes = reduce_shape(pool1_1.get_shape())
    conv1_1_flat = tf.reshape(pool1_1, [-1, conv1_1_num_nodes])

    # pad
    p = int(cfg['conv1_2']['filt_dim'] / 2)
    padded_conv1_1 = pool1_1
    for i in range(p):
        padded_conv1_1 = tf.pad(padded_conv1_1, paddings, 'SYMMETRIC')

    # conv1_2
    conv1_2h = tf.nn.relu(tf.nn.conv2d(padded_conv1_1, weights.conv1_2W, strides=[1,1,1,1], padding='VALID') + weights.conv1_2b)
    if cfg['conv1_2']['norm']:
        conv1_2h = tf.nn.local_response_normalization(conv1_2h,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
    pool1_2_size = cfg['conv1_2']['pool_size']
    pool1_2_stride = cfg['conv1_2']['pool_stride']
    pool1_2 = tf.nn.max_pool(conv1_2h,
                             ksize=[1, pool1_2_size, pool1_2_size, 1],
                             strides=[1, pool1_2_stride, pool1_2_stride, 1],
                             padding='SAME') 
    conv1_2_num_nodes = reduce_shape(pool1_2.get_shape())
    conv1_2_flat = tf.reshape(pool1_2, [-1, conv1_2_num_nodes])

    # conv2_1
    if cfg['conv1_2']['num_filt'] == 0:
        # pad
        p = int(cfg['conv2_1']['filt_dim'] / 2)
        padded_conv1_1 = pool1_1
        for i in range(p):
            padded_conv1_1 = tf.pad(padded_conv1_1, paddings, 'SYMMETRIC')

        conv2_1h = tf.nn.relu(tf.nn.conv2d(padded_conv1_1, weights.conv2_1W, strides=[1,1,1,1], padding='VALID') + weights.conv2_1b)
    else:
        # pad
        p = int(cfg['conv2_1']['filt_dim'] / 2)
        padded_conv1_2 = pool1_2
        for i in range(p):
            padded_conv1_2 = tf.pad(padded_conv1_2, paddings, 'SYMMETRIC')

        conv2_1h = tf.nn.relu(tf.nn.conv2d(padded_conv1_2, weights.conv2_1W, strides=[1,1,1,1], padding='VALID') + weights.conv2_1b)

    if cfg['conv2_1']['norm']:
        conv2_1h = tf.nn.local_response_normalization(conv2_1h,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
    pool2_1_size = cfg['conv2_1']['pool_size']
    pool2_1_stride = cfg['conv2_1']['pool_stride']
    pool2_1 = tf.nn.max_pool(conv2_1h,
                             ksize=[1, pool2_1_size, pool2_1_size, 1],
                             strides=[1, pool2_1_stride, pool2_1_stride, 1],
                             padding='SAME') 
    conv2_1_num_nodes = reduce_shape(pool2_1.get_shape())
    conv2_1_flat = tf.reshape(pool2_1, [-1, conv2_1_num_nodes])

    # pad
    p = int(cfg['conv2_2']['filt_dim'] / 2)
    padded_conv2_1 = pool2_1
    for i in range(p):
        padded_conv2_1 = tf.pad(padded_conv2_1, paddings, 'SYMMETRIC')

    # conv2_2
    conv2_2h = tf.nn.relu(tf.nn.conv2d(padded_conv2_1, weights.conv2_2W, strides=[1,1,1,1], padding='VALID') + weights.conv2_2b)
    if cfg['conv2_2']['norm']:
        conv2_2h = tf.nn.local_response_normalization(conv2_2h,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
    pool2_2_size = cfg['conv2_2']['pool_size']
    pool2_2_stride = cfg['conv2_2']['pool_stride']
    pool2_2 = tf.nn.max_pool(conv2_2h,
                             ksize=[1, pool2_2_size, pool2_2_size, 1],
                             strides=[1, pool2_2_stride, pool2_2_stride, 1],
                             padding='SAME') 
    conv2_2_num_nodes = reduce_shape(pool2_2.get_shape())
    conv2_2_flat = tf.reshape(pool2_2, [-1, conv2_2_num_nodes])

    # conv3
    use_conv3 = False
    if 'conv3_1' in cfg.keys():
        use_conv3 = True

    if use_conv3:
        # conv3_1
        if cfg['conv2_2']['num_filt'] == 0:
            # pad
            p = int(cfg['conv3_1']['filt_dim'] / 2)
            padded_conv2_1 = pool2_1
            for i in range(p):
                padded_conv2_1 = tf.pad(padded_conv2_1, paddings, 'SYMMETRIC')

            conv3_1h = tf.nn.relu(tf.nn.conv2d(padded_conv2_1, weights.conv3_1W, strides=[1,1,1,1], padding='VALID') + weights.conv3_1b)
        else:
            # pad
            p = int(cfg['conv3_1']['filt_dim'] / 2)
            padded_conv2_2 = pool2_2
            for i in range(p):
                padded_conv2_2 = tf.pad(padded_conv2_2, paddings, 'SYMMETRIC')

            conv3_1h = tf.nn.relu(tf.nn.conv2d(padded_conv2_2, weights.conv3_1W, strides=[1,1,1,1], padding='VALID') + weights.conv3_1b)
            

        if cfg['conv3_1']['norm']:
            conv3_1h = tf.nn.local_response_normalization(conv3_1h,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)
        pool3_1_size = cfg['conv3_1']['pool_size']
        pool3_1_stride = cfg['conv3_1']['pool_stride']
        pool3_1 = tf.nn.max_pool(conv3_1h,
                                 ksize=[1, pool3_1_size, pool3_1_size, 1],
                                 strides=[1, pool3_1_stride, pool3_1_stride, 1],
                                 padding='SAME') 
        conv3_1_num_nodes = reduce_shape(pool3_1.get_shape())
        conv3_1_flat = tf.reshape(pool3_1, [-1, conv3_1_num_nodes])

        # pad
        p = int(cfg['conv3_2']['filt_dim'] / 2)
        padded_conv3_1 = pool3_1
        for i in range(p):
            padded_conv3_1 = tf.pad(padded_conv3_1, paddings, 'SYMMETRIC')
        
        # conv3_2
        conv3_2h = tf.nn.relu(tf.nn.conv2d(padded_conv3_1, weights.conv3_2W, strides=[1,1,1,1], padding='VALID') + weights.conv3_2b)
        if cfg['conv3_2']['norm']:
            conv3_2h = tf.nn.local_response_normalization(conv3_2h,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)
        pool3_2_size = cfg['conv3_2']['pool_size']
        pool3_2_stride = cfg['conv3_2']['pool_stride']
        pool3_2 = tf.nn.max_pool(conv3_2h,
                                 ksize=[1, pool3_2_size, pool3_2_size, 1],
                                 strides=[1, pool3_2_stride, pool3_2_stride, 1],
                                 padding='SAME') 
        conv3_2_num_nodes = reduce_shape(pool3_2.get_shape())
        conv3_2_flat = tf.reshape(pool3_2, [-1, conv3_2_num_nodes])
        

    # fc3
    if use_conv3:
        if cfg['conv3_2']['num_filt'] == 0:
            fc3 = tf.nn.relu(tf.matmul(conv3_1_flat, weights.fc3W) + 
                             weights.fc3b)
        else:
            fc3 = tf.nn.relu(tf.matmul(conv3_2_flat, weights.fc3W) + 
                             weights.fc3b)            
    else:
        if cfg['conv2_2']['num_filt'] == 0:
            fc3 = tf.nn.relu(tf.matmul(conv2_1_flat, weights.fc3W) + 
                             weights.fc3b)
        else:
            fc3 = tf.nn.relu(tf.matmul(conv2_2_flat, weights.fc3W) + 
                             weights.fc3b) 
    if drop_fc3:
        fc3 = tf.nn.dropout(fc3, cfg['fc3']['drop_rate'])

    # pc1
    pc1 = tf.nn.relu(tf.matmul(poses_node, weights.pc1W) +
                     weights.pc1b)

    if cfg['pc2']['out_size'] == 0:
        # fc4
        fc4 = tf.nn.relu(tf.matmul(fc3, weights.fc4W_im) +
                         tf.matmul(pc1, weights.fc4W_pose) + 
                         weights.fc4b)
    else:
        # pc2
        pc2 = tf.nn.relu(tf.matmul(pc1, weights.pc2W) +
                         weights.pc2b)

        # fc4
        fc4 = tf.nn.relu(tf.matmul(fc3, weights.fc4W_im) +
                         tf.matmul(pc2, weights.fc4W_pose) + 
                         weights.fc4b)        

    if drop_fc4:
        fc4 = tf.nn.dropout(fc4, cfg['fc4']['drop_rate'])
    
    # fc5
    fc5 = tf.matmul(fc4, weights.fc5W) + weights.fc5b

    return fc5, image_node

if __name__ == '__main__':
    # set up logger
    logging.getLogger().setLevel(logging.INFO)
    
    # parse args
    parser = argparse.ArgumentParser(description='Train a CNN for grasp quality prediction')
    parser.add_argument('--config_filename', type=str, default='cfg/tools/train_grasp_quality_cnn.yaml', help='configuration file to use')
    args = parser.parse_args()
    config_filename = args.config_filename

    # open config file
    cfg = YamlConfig(config_filename)
    debug = cfg['debug']

    # set random seed for deterministic execution
    if debug:
        np.random.seed(SEED)
        random.seed(SEED)
        debug_num_files = cfg['debug_num_files']

    # setup output directory
    output_dir = cfg['output_dir']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    experiment_id = utils.gen_experiment_id()
    experiment_dir = os.path.join(output_dir, 'model_%s' %(experiment_id))
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    filter_dir = os.path.join(experiment_dir, 'filters')
    if not os.path.exists(filter_dir):
        os.mkdir(filter_dir)

    use_conv0 = False
    if 'conv0_1' in cfg['architecture'].keys():
        use_conv0 = True

    # copy config
    out_config_filename = os.path.join(experiment_dir, 'config.yaml')
    shutil.copyfile(config_filename, out_config_filename)
    this_filename = sys.argv[0]
    out_train_filename = os.path.join(experiment_dir, 'training_script.py')
    shutil.copyfile(this_filename, out_train_filename)
    out_architecture_filename = os.path.join(experiment_dir, 'architecture.json')
    json.dump(cfg['architecture'], open(out_architecture_filename, 'w'))

    print 'Saving model to %s' %(experiment_dir)

    # read params
    data_dir = cfg['dataset_dir']
    pretrained_model_dir = cfg['pretrained_model_dir']
    image_mode = cfg['image_mode']
    train_pct = cfg['train_pct']
    total_pct = cfg['total_pct']

    train_batch_size = cfg['train_batch_size']
    val_batch_size = cfg['val_batch_size']
    num_epochs = cfg['num_epochs']
    eval_frequency = cfg['eval_frequency']
    save_frequency = cfg['save_frequency']
    log_frequency = cfg['log_frequency']
    vis_frequency = cfg['vis_frequency']

    queue_capacity = cfg['queue_capacity']
    queue_sleep = cfg['queue_sleep']

    train_l2_regularizer = cfg['train_l2_regularizer']
    base_lr = cfg['base_lr']
    decay_step = cfg['decay_step']
    decay_rate = cfg['decay_rate']
    momentum_rate = cfg['momentum_rate']
    max_training_examples_per_load = cfg['max_training_examples_per_load']

    target_metric_name = cfg['target_metric_name']
    metric_thresh = cfg['metric_thresh']
    training_mode = cfg['training_mode']
    preproc_mode = cfg['preproc_mode']

    if train_pct < 0 or train_pct > 1:
        raise ValueError('Train percentage must be in range [0,1]')

    if total_pct < 0 or total_pct > 1:
        raise ValueError('Train percentage must be in range [0,1]')

    # setup denoising and synthetic data
    if cfg['multiplicative_denoising']:
        gamma_shape = cfg['gamma_shape']
        gamma_scale = 1.0 / gamma_shape

    # read in filenames
    logging.info('Reading filenames')
    all_filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    if image_mode == ImageMode.BINARY:
        im_filenames = [f for f in all_filenames if f.find(binary_im_tensor_template) > -1]
    elif image_mode == ImageMode.DEPTH:
        im_filenames = [f for f in all_filenames if f.find(depth_im_tensor_template) > -1]
    elif image_mode == ImageMode.DEPTH_TABLE:
        im_filenames = [f for f in all_filenames if f.find(depth_im_table_tensor_template) > -1]
    elif image_mode == ImageMode.BINARY_TF:
        im_filenames = [f for f in all_filenames if f.find(binary_im_tf_tensor_template) > -1]
    elif image_mode == ImageMode.COLOR_TF:
        im_filenames = [f for f in all_filenames if f.find(color_im_tf_tensor_template) > -1]
    elif image_mode == ImageMode.GRAY_TF:
        im_filenames = [f for f in all_filenames if f.find(gray_im_tf_tensor_template) > -1]
    elif image_mode == ImageMode.DEPTH_TF:
        im_filenames = [f for f in all_filenames if f.find(depth_im_tf_tensor_template) > -1]
    elif image_mode == ImageMode.COLOR_TF_TABLE:
        im_filenames = [f for f in all_filenames if f.find(color_im_tf_table_tensor_template) > -1]
    elif image_mode == ImageMode.DEPTH_TF_TABLE:
        im_filenames = [f for f in all_filenames if f.find(depth_im_tf_table_tensor_template) > -1]
    elif image_mode == ImageMode.GD_TF_TABLE:
        im_filenames = [f for f in all_filenames if f.find(gd_im_tf_table_tensor_template) > -1]
    elif image_mode == ImageMode.RGBD_TF_TABLE:
        im_filenames = [f for f in all_filenames if f.find(rgbd_im_tf_table_tensor_template) > -1]
    else:
        raise ValueError('Image mode %s not supported.' %(image_mode))

    pose_filenames = [f for f in all_filenames if f.find(hand_poses_template) > -1]
    label_filenames = [f for f in all_filenames if f.find('/' + target_metric_name) > -1]

    if debug:
        random.shuffle(im_filenames)
        random.shuffle(pose_filenames)
        random.shuffle(label_filenames)
        im_filenames = im_filenames[:debug_num_files]
        pose_filenames = pose_filenames[:debug_num_files]
        label_filenames = label_filenames[:debug_num_files]

    im_filenames.sort(key = lambda x: int(x[-9:-4]))
    pose_filenames.sort(key = lambda x: int(x[-9:-4]))
    label_filenames.sort(key = lambda x: int(x[-9:-4]))

    # check valid filenames
    if len(im_filenames) == 0 or len(label_filenames) == 0 or len(label_filenames) == 0:
        raise ValueError('No training files found')

    # subsample files
    num_files = len(im_filenames)
    num_files_used = int(total_pct * num_files)
    filename_indices = np.random.choice(num_files, size=num_files_used, replace=False)
    filename_indices.sort()
    im_filenames = [im_filenames[k] for k in filename_indices]
    pose_filenames = [pose_filenames[k] for k in filename_indices]
    label_filenames = [label_filenames[k] for k in filename_indices]

    # get parameters
    train_im_data = np.load(im_filenames[0])['arr_0']
    pose_data = np.load(pose_filenames[0])['arr_0']
    metric_data = np.load(label_filenames[0])['arr_0']
    images_per_file = train_im_data.shape[0]
    im_height = train_im_data.shape[1]
    im_width = train_im_data.shape[2]
    im_channels = train_im_data.shape[3]
    im_center = np.array([float(im_height-1)/2, float(im_width-1)/2])
    num_tensor_channels = cfg['num_tensor_channels']

    # set the model for the input data
    pose_shape = pose_data.shape[1]
    input_data_mode = cfg['input_data_mode']
    if input_data_mode == InputDataMode.TF_IMAGE:
        pose_dim = 1 # depth
    elif input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
        pose_dim = 3 # depth, cx, cy
    elif input_data_mode == InputDataMode.RAW_IMAGE:
        pose_dim = 4 # u, v, theta, depth
    elif input_data_mode == InputDataMode.RAW_IMAGE_PERSPECTIVE:
        pose_dim = 6 # u, v, theta, depth cx, cy
    elif input_data_mode == InputDataMode.REGRASPING:
        pose_dim = 3 # depth, phi, psi
    else:
        raise ValueError('Input data mode %s not understood' %(input_data_mode))

    num_files = len(im_filenames)
    num_random_files = min(num_files, cfg['num_random_files'])
    num_categories = 2

    # setup train and test indices
    num_datapoints = images_per_file * num_files
    num_train = int(train_pct * num_datapoints)
    all_indices = np.arange(num_datapoints)
    np.random.shuffle(all_indices)
    train_indices = np.sort(all_indices[:num_train])
    val_indices = np.sort(all_indices[num_train:])

    # convert decay step from epochs to iters
    decay_step = decay_step * num_train
    
    # make a map of the train and test indices for each file
    # TODO: different splitting functions (object-wise, stable-pose-wise, image-wise)
    logging.info('Computing indices')
    train_index_map_filename = os.path.join(experiment_dir, 'train_indices.pkl')
    val_index_map_filename = os.path.join(experiment_dir, 'val_indices.pkl')
    if os.path.exists(train_index_map_filename):
        train_index_map = pkl.load(open(train_index_map_filename, 'r'))
        val_index_map = pkl.load(open(val_index_map_filename, 'r'))
    else:
        train_index_map = {}
        val_index_map = {}
        for i, im_filename in enumerate(im_filenames):
            lower = i * images_per_file
            upper = (i+1) * images_per_file
            im_arr = np.load(im_filename)['arr_0']
            train_index_map[im_filename] = train_indices[(train_indices >= lower) & (train_indices < upper) &  (train_indices - lower < im_arr.shape[0])] - lower
            val_index_map[im_filename] = val_indices[(val_indices >= lower) & (val_indices < upper) & (val_indices - lower < im_arr.shape[0])] - lower
        pkl.dump(train_index_map, open(train_index_map_filename, 'w'))
        pkl.dump(val_index_map, open(val_index_map_filename, 'w'))

    # compute data mean
    # TODO: different data preproc schemes
    logging.info('Computing image mean')
    mean_filename = os.path.join(experiment_dir, 'mean.npy')
    std_filename = os.path.join(experiment_dir, 'std.npy')
    if cfg['use_pretrained_weights'] and not cfg['reinit_image_mean']:
        mean_filename = os.path.join(pretrained_model_dir, 'mean.npy')
        std_filename = os.path.join(pretrained_model_dir, 'std.npy')

    if os.path.exists(mean_filename):
        data_mean = np.load(mean_filename)
        data_std = np.load(std_filename)
    else:
        data_mean = np.zeros(num_tensor_channels)
        data_std = np.zeros(num_tensor_channels)
        random_file_indices = np.random.choice(num_files, size=num_random_files, replace=False)
        num_summed = 0
        for k in random_file_indices.tolist():
            im_filename = im_filenames[k]
            im_data = np.load(im_filename)['arr_0']
            for i in range(num_tensor_channels):
                data_mean[i] += np.sum(im_data[train_index_map[im_filename], :, :, i])
            num_summed += im_data[train_index_map[im_filename], :, :, :].shape[0]
        for i in range(num_tensor_channels):
            data_mean[i] = data_mean[i] / (num_summed * im_height * im_width)
        np.save(mean_filename, data_mean)

        for k in random_file_indices.tolist():
            im_filename = im_filenames[k]
            im_data = np.load(im_filename)['arr_0']
            for i in range(num_tensor_channels):
                data_std[i] += np.sum((im_data[train_index_map[im_filename], :, :, i] - data_mean[i])**2)
        for i in range(num_tensor_channels):
            data_std[i] = np.sqrt(data_std[i] / (num_summed * im_height * im_width))
        np.save(std_filename, data_std)

    # compute pose mean
    logging.info('Computing pose mean')
    pose_mean_filename = os.path.join(experiment_dir, 'pose_mean.npy')
    pose_std_filename = os.path.join(experiment_dir, 'pose_std.npy')
    if cfg['use_pretrained_weights'] and not cfg['reinit_pose_mean']:
        pose_mean_filename = os.path.join(pretrained_model_dir, 'pose_mean.npy')
        pose_std_filename = os.path.join(pretrained_model_dir, 'pose_std.npy')

    if os.path.exists(pose_mean_filename):
        pose_mean = np.load(pose_mean_filename)
        pose_std = np.load(pose_std_filename)
    else:
        pose_mean = np.zeros(pose_shape)
        pose_std = np.zeros(pose_shape)
        num_summed = 0
        random_file_indices = np.random.choice(num_files, size=num_random_files, replace=False)
        for k in random_file_indices.tolist():
            im_filename = im_filenames[k]
            pose_filename = pose_filenames[k]
            pose_data = np.load(pose_filename)['arr_0']
            pose_mean += np.sum(pose_data[train_index_map[im_filename],:], axis=0)
            num_summed += pose_data[train_index_map[im_filename]].shape[0]
        pose_mean = pose_mean / num_summed

        for k in random_file_indices.tolist():
            im_filename = im_filenames[k]
            pose_filename = pose_filenames[k]
            pose_data = np.load(pose_filename)['arr_0']
            pose_std += np.sum((pose_data[train_index_map[im_filename],:] - pose_mean)**2, axis=0)
        pose_std = np.sqrt(pose_std / num_summed)

        pose_std[pose_std==0] = 1.0

        np.save(pose_mean_filename, pose_mean)
        np.save(pose_std_filename, pose_std)

    if pose_dim == 1 and len(pose_mean.shape) > 0:
        pose_mean = pose_mean[2]
        pose_std = pose_std[2]

    if cfg['use_pretrained_weights']:
        out_mean_filename = os.path.join(experiment_dir, 'mean.npy')
        out_std_filename = os.path.join(experiment_dir, 'std.npy')
        out_pose_mean_filename = os.path.join(experiment_dir, 'pose_mean.npy')
        out_pose_std_filename = os.path.join(experiment_dir, 'pose_std.npy')
        np.save(out_mean_filename, data_mean)
        np.save(out_std_filename, data_std)
        np.save(out_pose_mean_filename, pose_mean)
        np.save(out_pose_std_filename, pose_std)

    # compute statistics of the input metrics
    # TODO: use in normalization schemes?
    logging.info('Computing metric stats')
    all_metrics = None
    all_val_metrics = None
    for im_filename, metric_filename in zip(im_filenames, label_filenames):
        metric_data = np.load(metric_filename)['arr_0']
        indices = val_index_map[im_filename]
        val_metric_data = metric_data[indices]
        if all_metrics is None:
            all_metrics = metric_data
        else:
            all_metrics = np.r_[all_metrics, metric_data]
        if all_val_metrics is None:
            all_val_metrics = val_metric_data
        else:
            all_val_metrics = np.r_[all_val_metrics, val_metric_data]
    min_metric = np.min(all_metrics)
    max_metric = np.max(all_metrics)
    mean_metric = np.mean(all_metrics)
    median_metric = np.median(all_metrics)

    if train_pct < 1.0:
        pct_pos_val = float(np.sum(all_val_metrics > metric_thresh)) / all_val_metrics.shape[0]
        print 'Percent positive in val set:', pct_pos_val

    ### SETUP TENSORFLOW TRAINING

    # setup nodes
    train_data_batch = tf.placeholder(tf.float32, (train_batch_size, im_height, im_width, num_tensor_channels))
    train_poses_batch = tf.placeholder(tf.float32, (train_batch_size, pose_dim))
    if training_mode == TrainingMode.REGRESSION:
        train_label_dtype = tf.float32
        numpy_dtype = np.float32
    elif training_mode == TrainingMode.CLASSIFICATION:
        train_label_dtype = tf.int64
        numpy_dtype = np.int64
    else:
        raise ValueError('Training mode %s not supported' %(training_mode))
    train_labels_batch = tf.placeholder(train_label_dtype, (train_batch_size,))
    val_data_node = tf.placeholder(tf.float32, (val_batch_size, im_height, im_width, num_tensor_channels))
    val_poses_node = tf.placeholder(tf.float32, (val_batch_size, pose_dim))

    # create queue
    q = tf.FIFOQueue(queue_capacity, [tf.float32, tf.float32, train_label_dtype], shapes=[(train_batch_size, im_height, im_width, num_tensor_channels),
                                                                                          (train_batch_size, pose_dim),
                                                                                          (train_batch_size,)])
    enqueue_op = q.enqueue([train_data_batch, train_poses_batch, train_labels_batch])
    train_data_node, train_poses_node, train_labels_node = q.dequeue()

    # get conv weights
    with tf.device('/gpu:0'):
        if cfg['use_pretrained_weights']:
            model_filename = os.path.join(cfg['pretrained_model_dir'], 'model.ckpt')
            reader = tf.train.NewCheckpointReader(model_filename)
            weights = build_izzynet_weights_pretrained(reader, im_channels, im_height,
                                                       im_width, pose_dim, cfg['architecture'],
                                                       reinit_pc1=cfg['reinit_pc1'],
                                                       reinit_pc2=cfg['reinit_pc2'],
                                                       reinit_fc3=cfg['reinit_fc3'],
                                                       reinit_fc4=cfg['reinit_fc4'],
                                                       reinit_fc5=cfg['reinit_fc5'])
        else:
            weights = build_izzynet_weights(num_categories, pose_dim,
                                            im_channels, im_height, im_width,
                                            image_mode, cfg['architecture'])

    # start tf session
    start_time = time.time()
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    
    # function for loading and enqueuing training data
    term_event = threading.Event()
    term_event.clear()
    dead_event = threading.Event()
    dead_event.clear()
    def load_and_enqueue():
        train_data = np.zeros([train_batch_size, im_height, im_width, num_tensor_channels]).astype(np.float32)
        train_poses = np.zeros([train_batch_size, pose_dim]).astype(np.float32)
        label_data = np.zeros(train_batch_size).astype(numpy_dtype)
        enqueue_num = 0

        # read parameters of gaussian process
        gp_rescale_factor = cfg['gaussian_process_scaling_factor']
        gp_sample_height = int(im_height / gp_rescale_factor)
        gp_sample_width = int(im_width / gp_rescale_factor)
        gp_num_pix = gp_sample_height * gp_sample_width
        gp_sigma_color = cfg['gaussian_process_sigma_color']
        gp_sigma_depth = cfg['gaussian_process_sigma_depth']
        if image_mode == ImageMode.COLOR_TF_TABLE:
            gp_sigmas = np.array([gp_sigma_color, gp_sigma_color, gp_sigma_color])
        elif image_mode == ImageMode.DEPTH_TF_TABLE:
            gp_sigmas = np.array([gp_sigma_depth])
        elif image_mode == ImageMode.GD_TF_TABLE:
            gp_sigmas = np.array([gp_sigma_color, gp_sigma_depth])
        elif image_mode == ImageMode.RGBD_TF_TABLE:
            gp_sigmas = np.array([gp_sigma_color, gp_sigma_color, gp_sigma_color, gp_sigma_depth])
        else:
            gp_sigmas = gp_sigma_depth * np.ones(num_tensor_channels)

        while not term_event.is_set():
            time.sleep(cfg['queue_sleep'])

            # loop through data
            num_queued = 0
            start_i = 0
            end_i = 0
            while start_i < train_batch_size:
                # compute num remaining
                num_remaining = train_batch_size - num_queued
                
                # gen file index uniformly at random
                file_num = np.random.choice(num_files, size=1)[0]
                train_data_filename = im_filenames[file_num]
                train_data_arr = np.load(train_data_filename)['arr_0'].astype(np.float32)
                train_poses_arr = np.load(pose_filenames[file_num])['arr_0'].astype(np.float32)
                train_label_arr = np.load(label_filenames[file_num])['arr_0'].astype(np.float32)

                # get batch indices uniformly at random
                train_ind = train_index_map[train_data_filename]
                np.random.shuffle(train_ind)
                upper = min(num_remaining, train_ind.shape[0], max_training_examples_per_load)
                ind = train_ind[:upper]
                num_loaded = ind.shape[0]
                end_i = start_i + num_loaded

                # subsample data
                train_data_arr = train_data_arr[ind,...]
                train_poses_arr = train_poses_arr[ind,:]
                train_label_arr = train_label_arr[ind]
                num_images = train_data_arr.shape[0]

                # setup image save for debugging
                if cfg['save_training_images']:
                    debug_dir = os.path.join(experiment_dir, 'debug')
                    if not os.path.exists(debug_dir):
                        os.mkdir(debug_dir)

                    for k in range(num_images):
                        plt.figure(k)
                        plt.subplot(1,2,1)
                        plt.imshow(train_data_arr[k,:,:,0].astype(np.float32), cmap=plt.cm.gray_r)
                        plt.title('Original')
                        plt.axis('off')

                # denoising and synthetic data generation
                if cfg['multiplicative_denoising']:
                    mult_samples = ss.gamma.rvs(gamma_shape, scale=gamma_scale, size=num_loaded)
                    mult_samples = mult_samples[:,np.newaxis,np.newaxis,np.newaxis]
                    train_data_arr = train_data_arr * np.tile(mult_samples, [1, im_height, im_width, im_channels])

                # randomly dropout regions of the image for robustness
                if cfg['image_dropout']:
                    for i in range(num_images):
                        if np.random.rand() < cfg['image_dropout_rate']:
                            train_image = train_data_arr[i,:,:,0]
                            nonzero_px = np.where(train_image > 0)
                            nonzero_px = np.c_[nonzero_px[0], nonzero_px[1]]
                            num_nonzero = nonzero_px.shape[0]
                            num_dropout_regions = ss.poisson.rvs(cfg['dropout_poisson_mean'])                             
                            # sample ellipses
                            dropout_centers = np.random.choice(num_nonzero, size=num_dropout_regions)
                            x_radii = ss.gamma.rvs(cfg['dropout_radius_shape'], scale=cfg['dropout_radius_scale'], size=num_dropout_regions)
                            y_radii = ss.gamma.rvs(cfg['dropout_radius_shape'], scale=cfg['dropout_radius_scale'], size=num_dropout_regions)

                            # set interior pixels to zero
                            for j in range(num_dropout_regions):
                                ind = dropout_centers[j]
                                dropout_center = nonzero_px[ind, :]
                                x_radius = x_radii[j]
                                y_radius = y_radii[j]
                                dropout_px_y, dropout_px_x = sd.ellipse(dropout_center[0], dropout_center[1], y_radius, x_radius, shape=train_image.shape)
                                train_image[dropout_px_y, dropout_px_x] = 0.0
                            train_data_arr[i,:,:,0] = train_image

                # dropout a region around the areas of the image with high gradient
                if cfg['gradient_dropout']:
                    for i in range(num_images):
                        if np.random.rand() < cfg['gradient_dropout_rate']:
                            train_image = train_data_arr[i,:,:,0]
                            grad_mag = sf.gaussian_gradient_magnitude(train_image, sigma=cfg['gradient_dropout_sigma'])
                            thresh = ss.gamma.rvs(cfg['gradient_dropout_shape'], cfg['gradient_dropout_scale'], size=1)
                            high_gradient_px = np.where(grad_mag > thresh)
                            train_image[high_gradient_px[0], high_gradient_px[1]] = 0.0
                        train_data_arr[i,:,:,0] = train_image

                # add correlated Gaussian noise
                if cfg['gaussian_process_denoising']:
                    for i in range(num_images):
                        for j in range(num_tensor_channels):
                            if np.random.rand() < cfg['gaussian_process_rate']:
                                train_image = train_data_arr[i,:,:,j]
                                gp_noise = ss.norm.rvs(scale=gp_sigmas[j], size=gp_num_pix).reshape(gp_sample_height, gp_sample_width)
                                gp_noise = sm.imresize(gp_noise, gp_rescale_factor, interp='bicubic', mode='F')
                                train_image[train_image > 0] += gp_noise[train_image > 0]
                                train_data_arr[i,:,:,j] = train_image

                # run open and close filters to 
                if cfg['morphological']:
                    for i in range(num_images):
                        train_image = train_data_arr[i,:,:,0]
                        sample = np.random.rand()
                        morph_filter_dim = ss.poisson.rvs(cfg['morph_poisson_mean'])                         
                        if sample < cfg['morph_open_rate']:
                            train_image = snm.grey_opening(train_image, size=morph_filter_dim)
                        else:
                            closed_train_image = snm.grey_closing(train_image, size=morph_filter_dim)
                            
                            # set new closed pixels to the minimum depth, mimicing the table
                            new_nonzero_px = np.where((train_image == 0) & (closed_train_image > 0))
                            closed_train_image[new_nonzero_px[0], new_nonzero_px[1]] = np.min(train_image[train_image>0])
                            train_image = closed_train_image.copy()

                        train_data_arr[i,:,:,0] = train_image                        

                # randomly dropout borders of the image for robustness
                if cfg['border_distortion']:
                    for i in range(num_images):
                        train_image = train_data_arr[i,:,:,0]
                        grad_mag = sf.gaussian_gradient_magnitude(train_image, sigma=cfg['border_grad_sigma'])
                        high_gradient_px = np.where(grad_mag > cfg['border_grad_thresh'])
                        high_gradient_px = np.c_[high_gradient_px[0], high_gradient_px[1]]
                        num_nonzero = high_gradient_px.shape[0]
                        if num_nonzero == 0:
                            continue

                        num_dropout_regions = ss.poisson.rvs(cfg['border_poisson_mean']) 
                        # sample ellipses
                        dropout_centers = np.random.choice(num_nonzero, size=num_dropout_regions)
                        x_radii = ss.gamma.rvs(cfg['border_radius_shape'], scale=cfg['border_radius_scale'], size=num_dropout_regions)
                        y_radii = ss.gamma.rvs(cfg['border_radius_shape'], scale=cfg['border_radius_scale'], size=num_dropout_regions)

                        # set interior pixels to zero or one
                        for j in range(num_dropout_regions):
                            ind = dropout_centers[j]
                            dropout_center = high_gradient_px[ind, :]
                            x_radius = x_radii[j]
                            y_radius = y_radii[j]
                            dropout_px_y, dropout_px_x = sd.ellipse(dropout_center[0], dropout_center[1], y_radius, x_radius, shape=train_image.shape)
                            if np.random.rand() < 0.5:
                                train_image[dropout_px_y, dropout_px_x] = 0.0
                            else:
                                train_image[dropout_px_y, dropout_px_x] = train_image[dropout_center[0], dropout_center[1]]

                        train_data_arr[i,:,:,0] = train_image

                # randomly replace background pixels with constant depth
                if cfg['background_denoising']:
                    for i in range(num_images):
                        train_image = train_data_arr[i,:,:,0]                
                        if np.random.rand() < cfg['background_rate']:
                            train_image[train_image > 0] = cfg['background_min_depth'] + (cfg['background_max_depth'] - cfg['background_min_depth']) * np.random.rand()

                # symmetrize images
                if cfg['symmetrize']:
                    for i in range(num_images):
                        train_image = train_data_arr[i,:,:,:]
                        # rotate with 50% probability
                        if np.random.rand() < 0.5:
                            theta = 180.0
                            rot_map = cv2.getRotationMatrix2D(tuple(im_center), theta, 1)
                            for j in range(num_tensor_channels):
                                train_image[:,:,j] = cv2.warpAffine(train_image[:,:,j], rot_map, (im_height, im_width), flags=cv2.INTER_NEAREST)
                            if input_data_mode == InputDataMode.REGRASPING:
                                train_poses_arr[i,4] = -train_poses_arr[i,4]
                                train_poses_arr[i,6] = -train_poses_arr[i,6]

                        # reflect left right with 50% probability
                        if cfg['reflect_lr'] and np.random.rand() < 0.5:
                            for j in range(num_tensor_channels):
                                train_image[:,:,j] = np.fliplr(train_image[:,:,j])

                        # reflect up down with 50% probability
                        if cfg['reflect_ud'] and np.random.rand() < 0.5:
                            for j in range(num_tensor_channels):
                                train_image[:,:,j] = np.flipud(train_image[:,:,j])
                            if input_data_mode == InputDataMode.REGRASPING:
                                train_poses_arr[i,4] = -train_poses_arr[i,4]
                                train_poses_arr[i,6] = -train_poses_arr[i,6]
                        train_data_arr[i,:,:,:] = train_image

                # save training examples for debugging
                if cfg['save_training_images'] and enqueue_num == 0:
                    debug_dir = os.path.join(experiment_dir, 'debug')
                    for k in range(num_images):
                        filename = os.path.join(debug_dir, 'images_%05d_%05d.jpg' %(enqueue_num, k))
                        plt.figure(k)
                        plt.subplot(1,2,2)
                        plt.imshow(train_data_arr[k,:,:,0].astype(np.float32), cmap=plt.cm.gray_r)
                        plt.title('Denoised')
                        plt.axis('off')

                        plt.savefig(os.path.join(debug_dir, filename))

                # subtract mean
                if len(data_mean.shape) == 0:
                    train_data_arr = (train_data_arr - data_mean) / data_std
                else:
                    for i in range(num_tensor_channels):
                        train_data_arr[:,:,:,i] = (train_data_arr[:,:,:,i] - data_mean[i]) / data_std[i]
                train_poses_arr = (train_poses_arr - pose_mean) / pose_std

                # normalize labels?
                if training_mode == TrainingMode.REGRESSION:
                    if preproc_mode == PreprocMode.NORMALIZATION:
                        train_label_arr = (train_label_arr - min_metric) / (max_metric - min_metric)
                    elif preproc_mode == PreprocMode.IZZYNET:
                        train_label_arr = 2 * (train_label_arr - min_metric) / (max_metric - min_metric) - 1
                elif training_mode == TrainingMode.CLASSIFICATION:
                    train_label_arr = 1 * (train_label_arr > metric_thresh)
                    train_label_arr = train_label_arr.astype(numpy_dtype)

                # enqueue training data batch                
                train_data[start_i:end_i, ...] = train_data_arr
                train_poses[start_i:end_i,:] = read_pose_data(train_poses_arr, input_data_mode)
                label_data[start_i:end_i] = train_label_arr
                del train_data_arr
                del train_poses_arr
                del train_label_arr

                # update start index
                start_i = end_i
                num_queued += num_loaded
          
            # send data to queue
            if not term_event.is_set():
                sess.run(enqueue_op, feed_dict={train_data_batch: train_data,
                                                train_poses_batch: train_poses,
                                                train_labels_batch: label_data})
                enqueue_num += 1

        del train_data
        del train_poses
        del label_data
        dead_event.set()
        q.close()
        print 'Queue thread exit'
        exit(0)

    # build training and validation networks
    drop_fc3 = False
    if 'drop_fc3' in cfg.keys() and cfg['drop_fc3']:
        drop_fc3 = True
    drop_fc4 = False
    if 'drop_fc4' in cfg.keys() and cfg['drop_fc4']:
        drop_fc4 = True
    with tf.device('/gpu:0'):
        train_net_output, train_denoised_output = build_izzynet(train_data_node, train_poses_node, weights, im_height, im_width, cfg['architecture'], drop_fc3=drop_fc3, drop_fc4=drop_fc4)
        val_net_output, val_denoised_output = build_izzynet(val_data_node, val_poses_node, weights, im_height, im_width, cfg['architecture'])

        if training_mode == TrainingMode.CLASSIFICATION or preproc_mode == PreprocMode.NORMALIZATION:
            train_predictions = tf.nn.softmax(train_net_output)
            val_predictions = tf.nn.softmax(val_net_output)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_net_output, labels=train_labels_node))
        elif training_mode == TrainingMode.REGRESSION:
            train_predictions = train_net_output
            val_predictions = val_net_output
            loss = tf.nn.l2_loss(tf.sub(train_net_output, train_labels_node))

    # form loss
    layer_weights = weights.__dict__.values()
    regularizers = tf.nn.l2_loss(layer_weights[0])
    for w in layer_weights[1:]:
        regularizers = regularizers + tf.nn.l2_loss(w)
    loss += train_l2_regularizer * regularizers

    # setup learning rate
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        base_lr,                # Base learning rate.
        batch * train_batch_size,  # Current epoch.
        decay_step,          # Decay step.
        decay_rate,                # Decay rate.
        staircase=True)

    # setup variable list
    var_list = weights.__dict__.values()
    if cfg['use_pretrained_weights'] and cfg['update_fc_only']:
        var_list = [v for k, v in weights.__dict__.iteritems() if k.find('conv') == -1]
    elif cfg['use_pretrained_weights'] and cfg['update_conv0_only'] and use_conv0:
        var_list = [v for k, v in weights.__dict__.iteritems() if k.find('conv0') > -1]
    elif cfg['use_pretrained_weights'] and cfg['update_fc5_only']:
        var_list = [v for k, v in weights.__dict__.iteritems() if k.find('fc5') > -1]
    elif cfg['use_pretrained_weights'] and cfg['update_fc4_fc5_only']:
        var_list = [v for k, v in weights.__dict__.iteritems() if k.find('fc5') > -1 or k.find('fc4') > -1]

    # create optimizer
    if cfg['optimizer'] == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate,
                                               momentum_rate).minimize(loss,
                                                                       global_step=batch,
                                                                       var_list=var_list)
    elif cfg['optimizer'] == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,
                                                                   global_step=batch,
                                                                   var_list=var_list)
    elif cfg['optimizer'] == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss,
                                                                      global_step=batch,
                                                                      var_list=var_list)
    else:
        raise ValueError('Optimizer %s not supported' %(cfg['optimizer']))

    # setup data thread
    def handler(signum, frame):
        print 'caught CTRL+C, exiting...'
        term_event.set()
    signal.signal(signal.SIGINT, handler)

    try:
        start_time = time.time()

        t = threading.Thread(target=load_and_enqueue)
        t.start()

        # init and run tf sessions
        init = tf.global_variables_initializer()
        sess.run(init)
        print('Initialized!')
            
        # Loop through training steps.
        train_eval_iters = []
        train_losses = []
        train_errors = []
        total_train_errors = []
        val_eval_iters = []
        val_errors = []
        learning_rates = []
        for step in xrange(int(num_epochs * num_train) // train_batch_size):
            # check dead queue
            if dead_event.is_set():
                # close session
                sess.close()
        
                # cleanup
                for layer_weights in weights.__dict__.values():
                    del layer_weights
                del saver
                del sess
                exit(0)
            
            # run optimization
            if use_conv0:
                _, l, lr, predictions, batch_labels, output, train_images, conv1_1W, conv1_1b, conv0_1W, conv0_1b, train_denoised_images = sess.run(
                    [optimizer, loss, learning_rate, train_predictions, train_labels_node, train_net_output, train_data_node, weights.conv1_1W, weights.conv1_1b, weights.conv0_1W, weights.conv0_1b, train_denoised_output], options=timeout_option)
            else:
                _, l, lr, predictions, batch_labels, output, train_images, conv1_1W, conv1_1b = sess.run(
                    [optimizer, loss, learning_rate, train_predictions, train_labels_node, train_net_output, train_data_node, weights.conv1_1W, weights.conv1_1b], options=timeout_option)

            ex = np.exp(output - np.tile(np.max(output, axis=1)[:,np.newaxis], [1,2]))
            softmax = ex / np.tile(np.sum(ex, axis=1)[:,np.newaxis], [1,2])
            print 'Max', np.max(softmax[:,1])
            print 'Min', np.min(softmax[:,1])
            print 'Pred nonzero', np.sum(np.argmax(predictions, axis=1))
            print 'True nonzero', np.sum(batch_labels)

            # log output
            if step % log_frequency == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' %
                      (step, float(step) * train_batch_size / num_train,
                       1000 * elapsed_time / eval_frequency))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                train_error = l
                if training_mode == TrainingMode.CLASSIFICATION:
                    train_error = class_error_rate(predictions, batch_labels)
                    print('Minibatch error: %.3f%%' %train_error)
                sys.stdout.flush()

                train_eval_iters.append(step)
                train_errors.append(train_error)
                train_losses.append(l)
                learning_rates.append(lr)

            # evaluate validation error
            if step % eval_frequency == 0:
                if cfg['eval_total_train_error']:
                    train_error = error_rate_in_batches(
                        im_filenames, pose_filenames, label_filenames, train_index_map,
                        sess, val_predictions, pose_dim, data_mean, data_std, pose_mean, pose_std, val_batch_size,
                        num_categories, image_mode, training_mode,
                        preproc_mode, min_metric, max_metric, metric_thresh, input_data_mode, num_tensor_channels)
                    print('Training error: %.3f' %train_error)
                    total_train_errors.append(train_error)
                    np.save(os.path.join(experiment_dir, 'total_train_errors.npy'), total_train_errors)

                val_error = 0
                if train_pct < 1.0:
                    val_error = error_rate_in_batches(
                        im_filenames, pose_filenames, label_filenames, val_index_map,
                        sess, val_predictions, pose_dim, data_mean, data_std, pose_mean, pose_std, val_batch_size,
                        num_categories, image_mode, training_mode,
                        preproc_mode, min_metric, max_metric, metric_thresh, input_data_mode, num_tensor_channels)
                    print('Validation error: %.3f' %val_error)
                sys.stdout.flush()

                val_eval_iters.append(step)
                val_errors.append(val_error)

                # save everything!
                np.save(os.path.join(experiment_dir, 'train_eval_iters.npy'), train_eval_iters)
                np.save(os.path.join(experiment_dir, 'val_eval_iters.npy'), val_eval_iters)
                np.save(os.path.join(experiment_dir, 'train_losses.npy'), train_losses)
                np.save(os.path.join(experiment_dir, 'train_errors.npy'), train_errors)
                np.save(os.path.join(experiment_dir, 'val_errors.npy'), val_errors)
                np.save(os.path.join(experiment_dir, 'learning_rates.npy'), learning_rates)
            
            # save the model
            if step % save_frequency == 0 and step > 0:
                saver.save(sess, os.path.join(experiment_dir, 'model_%05d.ckpt' %(step)))

            # visualize
            if step % vis_frequency == 0:
                # conv1_1
                num_filt = conv1_1W.shape[3]
                d = int(np.ceil(np.sqrt(num_filt)))

                plt.clf()
                for i in range(num_filt):
                    plt.subplot(d,d,i+1)
                    plt.imshow(conv1_1W[:,:,0,i], cmap=plt.cm.gray, interpolation='nearest')
                    plt.axis('off')
                    plt.title('b=%.3f' %(conv1_1b[i]), fontsize=10)
                if cfg['show_filters']:
                    plt.show()
                else:
                    plt.savefig(os.path.join(filter_dir, 'conv1_1W_%05d.jpg' %(step)))

                # conv0_1
                if use_conv0:
                    num_filt = conv0_1W.shape[3]
                    d = int(np.ceil(np.sqrt(num_filt)))

                    plt.clf()
                    for i in range(num_filt):
                        plt.subplot(d,d,i+1)
                        plt.imshow(conv0_1W[:,:,0,i], cmap=plt.cm.gray, interpolation='nearest')
                        plt.axis('off')
                        plt.title('b=%.3f' %(conv0_1b[i]), fontsize=10)
                    if cfg['show_filters']:
                        plt.show()
                    else:
                        plt.savefig(os.path.join(filter_dir, 'conv0_1W_%05d.jpg' %(step)))

                    train_images = train_images * data_std + data_mean
                    train_denoised_images = train_denoised_images * data_std + data_mean
                    n = 5
                    plt.clf()
                    for i in range(n):
                        plt.subplot(n,2,2*i+1)
                        plt.imshow(train_images[i,:,:,0], cmap=plt.cm.gray_r)
                        plt.axis('off')
                        plt.subplot(n,2,2*i+2)
                        plt.imshow(train_denoised_images[i,:,:,0], cmap=plt.cm.gray_r)
                        plt.axis('off')
                    if cfg['show_filters']:
                        plt.show()
                    else:
                        plt.savefig(os.path.join(filter_dir, 'images_%05d.jpg' %(step)))
                        

        # get final logs
        val_error = 0
        if train_pct < 1.0:
            val_error = error_rate_in_batches(
                im_filenames, pose_filenames, label_filenames, val_index_map,
                sess, val_predictions, pose_dim, data_mean, data_std, pose_mean, pose_std, val_batch_size,
                num_categories, image_mode, training_mode,
                preproc_mode, min_metric, max_metric, metric_thresh, input_data_mode, num_tensor_channels)
            print('Final validation error: %.1f%%' %val_error)
        sys.stdout.flush()

        val_eval_iters.append(step)
        val_errors.append(val_error)

        # save everything!
        np.save(os.path.join(experiment_dir, 'train_eval_iters.npy'), train_eval_iters)
        np.save(os.path.join(experiment_dir, 'val_eval_iters.npy'), val_eval_iters)
        np.save(os.path.join(experiment_dir, 'train_losses.npy'), train_losses)
        np.save(os.path.join(experiment_dir, 'train_errors.npy'), train_errors)
        np.save(os.path.join(experiment_dir, 'val_errors.npy'), val_errors)
        saver.save(sess, os.path.join(experiment_dir, 'model.ckpt'))

        end_time = time.time()
        print('Training took %.3f sec' %(end_time-start_time))

    except Exception as e:
        term_event.set()
        sess.close() 
        for layer_weights in weights.__dict__.values():
            del layer_weights
        del saver
        del sess
        raise

    if dead_event.is_set():
        # close session
        sess.close()
        
        # cleanup
        for layer_weights in weights.__dict__.values():
            del layer_weights
        del saver
        del sess

    # close sesions
    term_event.set()
    print('Waiting for queue')
    while not dead_event.is_set():
        pass
    sess.close()
        
    # cleanup
    for layer_weights in weights.__dict__.values():
        del layer_weights
    del saver
    del sess

