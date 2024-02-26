# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents).
All Rights Reserved. Permission to use, copy, modify, and distribute this
software and its documentation for educational, research, and not-for-profit
purposes, without fee and without a signed licensing agreement, is hereby
granted, provided that the above copyright notice, this paragraph and the
following two paragraphs appear in all copies, modifications, and
distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150
Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
otl@berkeley.edu,
http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Displays robust grasps planned using a GQ-CNN grapsing policy on a set of saved
RGB-D images. The default configuration for the standard GQ-CNN policy is
`cfg/examples/cfg/examples/gqcnn_pj.yaml`. The default configuration for the
Fully-Convolutional GQ-CNN policy is `cfg/examples/fc_gqcnn_pj.yaml`.

Author
------
Jeff Mahler & Vishal Satish
"""
import argparse
import json
import os
import time

import numpy as np

from autolab_core import (YamlConfig, Logger, BinaryImage, CameraIntrinsics,
                          ColorImage, DepthImage, RgbdImage,Point, Logger, DepthImage)
from visualization import Visualizer2D as vis

from gqcnn.grasping import (RobustGraspingPolicy,
                            CrossEntropyRobustGraspingPolicy, RgbdImageState,
                            FullyConvolutionalGraspingPolicyParallelJaw,
                            FullyConvolutionalGraspingPolicySuction,Grasp2D, SuctionPoint2D)
from gqcnn.utils import GripperMode
from gqcnn.grasping.policy.enums import SamplingMethod
from gqcnn.grasping.grasp import SuctionPoint2D
from gqcnn.grasping.policy.policy import GraspingPolicy, GraspAction
from gqcnn.utils import GeneralConstants, NoValidGraspsException


from gqcnn.grasping.policy.fc_policy import FullyConvolutionalGraspingPolicy
# gqcnn/grasping/grasp_quality_function.py
from gqcnn.grasping.grasp_quality_function import GQCnnQualityFunction,FCGQCnnQualityFunction, GraspQualityFunction,GraspQualityFunctionFactory
# Set up logger.
# logger = Logger.get_logger("examples/policy.py")
logger = Logger.get_logger("examples/policy.py")

def poicy_sim(model_name,model_dir,depth_im_filename,camera_intr_filename,model_config,want_visuals=True,segmask_filename=None,config_filename = None,corled_img=None):



    if "FC-" in model_name:
        fully_conv = True
    else:
        fully_conv = False

    
    model_path = os.path.join(model_dir, model_name)
    
    model_config = json.load(open(os.path.join(model_path, "config.json"),
                                  "r"))

    try:
        gqcnn_config = model_config["gqcnn"]
        gripper_mode = gqcnn_config["gripper_mode"]
    except KeyError:
        gqcnn_config = model_config["gqcnn_config"]
        input_data_mode = gqcnn_config["input_data_mode"]
        if input_data_mode == "tf_image":
            gripper_mode = GripperMode.LEGACY_PARALLEL_JAW
        elif input_data_mode == "tf_image_suction":
            gripper_mode = GripperMode.LEGACY_SUCTION
        elif input_data_mode == "suction":
            gripper_mode = GripperMode.SUCTION
        elif input_data_mode == "multi_suction":
            gripper_mode = GripperMode.MULTI_SUCTION
        elif input_data_mode == "parallel_jaw":
            gripper_mode = GripperMode.PARALLEL_JAW
        else:
            raise ValueError(
                "Input data mode {} not supported!".format(input_data_mode))

    # Set config.
    if config_filename is None:
        if (gripper_mode == GripperMode.LEGACY_PARALLEL_JAW
                or gripper_mode == GripperMode.PARALLEL_JAW):
            if fully_conv:
                config_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "..",
                    "cfg/examples/fc_gqcnn_pj.yaml")
            else:
                config_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "..",
                    "cfg/examples/gqcnn_pj.yaml")
        elif (gripper_mode == GripperMode.LEGACY_SUCTION
              or gripper_mode == GripperMode.SUCTION):
            if fully_conv:
                config_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "..",
                    "cfg/examples/fc_gqcnn_suction.yaml")
            else:
                config_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "..",
                    "cfg/examples/gqcnn_suction.yaml")
    
    config_filename =config_filename.replace("/..","")
    config = YamlConfig(config_filename)

    inpaint_rescale_factor = config["inpaint_rescale_factor"]
    policy_config = config["policy"]

    # Make relative paths absolute.
    if "gqcnn_model" in policy_config["metric"]:
        policy_config["metric"]["gqcnn_model"] = model_path
        if not os.path.isabs(policy_config["metric"]["gqcnn_model"]):
            policy_config["metric"]["gqcnn_model"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..",policy_config["metric"]["gqcnn_model"])

    camera_intr = CameraIntrinsics.load(camera_intr_filename)

    # Read images.

    depth_data = np.load(depth_im_filename)
    depth_im = DepthImage(depth_data, frame=camera_intr.frame)
    color_im = ColorImage(np.zeros([depth_im.height, depth_im.width,
                                    3]).astype(np.uint8),
                          frame=camera_intr.frame)

    # Optionally read a segmask.
    segmask = None
    if segmask_filename is not None:

        segmask = BinaryImage.open(segmask_filename)
    valid_px_mask = depth_im.invalid_pixel_mask().inverse()
    if segmask is None:
        segmask = valid_px_mask
    else:
        segmask = segmask.mask_binary(valid_px_mask)

    # Inpaint.
    depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)

    if "input_images" in policy_config["vis"] and policy_config["vis"][
        "input_images"]:
        vis.figure(size=(10, 10))
        num_plot = 1
        if segmask is not None:
            num_plot = 2
        vis.subplot(1, num_plot, 1)
        vis.imshow(depth_im)
        if segmask is not None:
            vis.subplot(1, num_plot, 2)
            vis.imshow(segmask)
        vis.show()

    # Create state.
    rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
    state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)

    # Set input sizes for fully-convolutional policy.
    if fully_conv:
        policy_config["metric"]["fully_conv_gqcnn_config"][
            "im_height"] = depth_im.shape[0]
        policy_config["metric"]["fully_conv_gqcnn_config"][
            "im_width"] = depth_im.shape[1]

    # Init policy.

    if fully_conv:
        # TODO(vsatish): We should really be doing this in some factory policy.
        if policy_config["type"] == "fully_conv_suction":
            
            policy = FullyConvolutionalGraspingPolicySuction(policy_config)
        elif policy_config["type"] == "fully_conv_pj":
            policy = FullyConvolutionalGraspingPolicyParallelJaw(policy_config)
        else:
            raise ValueError(
                "Invalid fully-convolutional policy type: {}".format(
                    policy_config["type"]))
    else:
        policy_type = "cem"
        if "type" in policy_config:
            policy_type = policy_config["type"]
        if policy_type == "ranking":
            policy = RobustGraspingPolicy(policy_config)
        elif policy_type == "cem":

            policy = CrossEntropyRobustGraspingPolicy(policy_config)

        else:
            raise ValueError("Invalid policy type: {}".format(policy_type))

    # Query policy.
    policy_start = time.time()
    action = policy(state)
      
    # Vis final grasp
    if want_visuals == True:
        if policy_config["vis"]["final_grasp"]:
            vis.figure(size=(10, 10))
            vis.imshow(rgbd_im.depth,
                       vmin=policy_config["vis"]["vmin"],
                       vmax=policy_config["vis"]["vmax"])
            vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
            vis.title("Planned grasp at depth {0:.3f}m with Q={1:.3f}".format(
                action.grasp.depth, action.q_value))
            vis.show()

    #.quality(state,[action],None)
    return action



def phyiscal_sim(old_grasp,model_name,model_dir,depth_im_filename,camera_intr_filename,model_config,segmask_filename=None,config_filename = None):

    if "FC-" in model_name:
        fully_conv = True
    else:
        fully_conv = False
    config_filename = None
    model_path = os.path.join(model_dir, model_name) ## just model name this honestly but we might need directory later
    # print(model_path, "model path")

    model_config = json.load(open(os.path.join(model_path, "config.json"),
                                  "r"))

    try:
        gqcnn_config = model_config["gqcnn"]
        gripper_mode = gqcnn_config["gripper_mode"]
    except KeyError:
        gqcnn_config = model_config["gqcnn_config"]
        input_data_mode = gqcnn_config["input_data_mode"]
        if input_data_mode == "tf_image":
            gripper_mode = GripperMode.LEGACY_PARALLEL_JAW
        elif input_data_mode == "tf_image_suction":
            gripper_mode = GripperMode.LEGACY_SUCTION
        elif input_data_mode == "suction":
            gripper_mode = GripperMode.SUCTION
        elif input_data_mode == "multi_suction":
            gripper_mode = GripperMode.MULTI_SUCTION
        elif input_data_mode == "parallel_jaw":
            gripper_mode = GripperMode.PARALLEL_JAW
        else:
            raise ValueError(
                "Input data mode {} not supported!".format(input_data_mode))

    # Set config.
    if config_filename is None:
        if (gripper_mode == GripperMode.LEGACY_PARALLEL_JAW
                or gripper_mode == GripperMode.PARALLEL_JAW):
            if fully_conv:
                config_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "..",
                    "cfg/examples/fc_gqcnn_pj.yaml")
            else:
                config_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "..",
                    "cfg/examples/gqcnn_pj.yaml")
        elif (gripper_mode == GripperMode.LEGACY_SUCTION
              or gripper_mode == GripperMode.SUCTION):
            if fully_conv:
                config_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "..",
                    "cfg/examples/fc_gqcnn_suction.yaml")
            else:
                config_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "..",
                    "cfg/examples/gqcnn_suction.yaml")

    # Read config.
    
    config_filename = config_filename.replace("/..","")
    config = YamlConfig(config_filename)

    inpaint_rescale_factor = config["inpaint_rescale_factor"]
    policy_config = config["policy"]

    # Make relative paths absolute.
    if "gqcnn_model" in policy_config["metric"]:
        policy_config["metric"]["gqcnn_model"] = model_path
        if not os.path.isabs(policy_config["metric"]["gqcnn_model"]):
            policy_config["metric"]["gqcnn_model"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..",policy_config["metric"]["gqcnn_model"])

    camera_intr = CameraIntrinsics.load(camera_intr_filename)

    # Read images.
    depth_data = np.load(depth_im_filename,)
    depth_im = DepthImage(depth_data, frame=camera_intr.frame)
    color_im = ColorImage(np.zeros([depth_im.height, depth_im.width,
                                    3]).astype(np.uint8),
                          frame=camera_intr.frame)

    # Optionally read a segmask.
    segmask = None
    if segmask_filename is not None:

        segmask = BinaryImage.open(segmask_filename)
    valid_px_mask = depth_im.invalid_pixel_mask().inverse()
    if segmask is None:
        segmask = valid_px_mask
    else:
        segmask = segmask.mask_binary(valid_px_mask)

    # Inpaint.
    depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)

    if "input_images" in policy_config["vis"] and policy_config["vis"][
        "input_images"]:
        vis.figure(size=(10, 10))
        num_plot = 1
        if segmask is not None:
            num_plot = 2
        vis.subplot(1, num_plot, 1)
        vis.imshow(depth_im)
        if segmask is not None:
            vis.subplot(1, num_plot, 2)
            vis.imshow(segmask)
        vis.show()

    # Create state.
    rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
    state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)

    # Set input sizes for fully-convolutional policy.
    if fully_conv:

        policy_config["metric"]["fully_conv_gqcnn_config"][
            "im_height"] = depth_im.shape[0]
        policy_config["metric"]["fully_conv_gqcnn_config"][
            "im_width"] = depth_im.shape[1]


    # Init policy.

    if fully_conv:
        # TODO(vsatish): We should really be doing this in some factory policy.
        if policy_config["type"] == "fully_conv_suction":
            policy = FullyConvolutionalGraspingPolicySuction(policy_config)
        elif policy_config["type"] == "fully_conv_pj":
            policy = FullyConvolutionalGraspingPolicyParallelJaw(policy_config)

        else:
                raise ValueError(
                "Invalid fully-convolutional policy type: {}".format(
                    policy_config["type"]))
    else:
        policy_type = "cem"
        if "type" in policy_config:
            policy_type = policy_config["type"]
        if policy_type == "ranking":
            policy = RobustGraspingPolicy(policy_config)
        elif policy_type == "cem":

            policy = CrossEntropyRobustGraspingPolicy(policy_config)

        else:
            raise ValueError("Invalid policy type: {}".format(policy_type))



    if fully_conv:

        # action = FCGQCnnQualityFunction(config['policy']['metric'])  ## need to make an abstract class instance to make this work
        action = policy

        wrapped_depth, raw_depth, raw_seg, camera_intr = action._unpack_state(
            state)

        # Predict.
        images, depths = action._gen_images_and_depths(raw_depth, raw_seg)
        # depths = np.array([old_grasp.grasp.depth])
        depths = np.array(old_grasp.grasp.depth)
        if action._config["type"] == "fully_conv_suction":
            preds = action._grasp_quality_fn.quality(images, depths)
        else:
            preds = action._grasp_quality_fn.quality(images, [depths])

        # Get success probablility predictions only (this is needed because the
        # output of the network is pairs of (p_failure, p_success)).
        preds_success_only = preds[:, :, :, 1::2]
        num_actions_to_sample = action._max_grasps_to_filter


        if (action._sampling_method == SamplingMethod.TOP_K
                and action._num_vis_samples):
            action._logger.warning("FINAL GRASP RETURNED IS NOT THE BEST!")


        sampled_ind = action._sample_predictions(preds_success_only,
                                               num_actions_to_sample)


        # Mask predicted success probabilities with the cropped and downsampled
        # object segmask so we only sample grasps on the objects.

        preds_success_only = action._mask_predictions(preds_success_only,
                                                    raw_seg)

        if action._config["type"] == "fully_conv_suction":

            actions = action._get_actions(preds_success_only, sampled_ind, images,
                                        depths, camera_intr, num_actions_to_sample)
        else:
            actions = action._get_actions(preds_success_only, sampled_ind, images,
                                             np.tile(depths, (16, 1)), camera_intr, num_actions_to_sample) ##Need to expand the depths to tiled version

        # Filter grasps.
        if action._filter_grasps:
            actions = sorted(actions,
                             reverse=True,
                             key=lambda action: action.q_value)
            actions = [action._filter(actions)]


        for k in actions:

            if ([k.grasp.center[0],k.grasp.center[1]] == [old_grasp.grasp.center[0],old_grasp.grasp.center[1]]):

                    return k

            # print(k.grasp.depth == old_grasp.grasp.depth )
        just_for_formatting_parmasion = SuctionPoint2D(old_grasp.grasp.center,
                                                       axis=old_grasp.grasp.axis,
                                                       depth=old_grasp.grasp.depth,
                                                       camera_intr=old_grasp.grasp.center,
                                                       )
        formatted_parmasion = GraspAction(just_for_formatting_parmasion, 1)
        return formatted_parmasion

    else:
       final_grasp = policy._grasp_quality_fn(state,[old_grasp.grasp], params=policy._config)


       ## tidy way to intergrate it with our currnent system otherwise it would reqiure a lot of bug testing
       just_for_formatting_parmasion= SuctionPoint2D(old_grasp.grasp.center,
                                           axis=old_grasp.grasp.axis,
                                           depth=old_grasp.grasp.depth,
                                           camera_intr=old_grasp.grasp.center,
                                           )
       formatted_final_grasp = GraspAction(just_for_formatting_parmasion,final_grasp[0])



    return formatted_final_grasp

