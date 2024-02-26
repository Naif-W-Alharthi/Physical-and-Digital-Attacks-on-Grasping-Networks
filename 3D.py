import sys
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from policy_inhouse import *
import os
import logging
import copy
import tensorflow as tf
# import deap
import csv
import random
import time
import threading
from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from matplotlib import pyplot as plt
import array
import ast
from textwrap import wrap
from multiprocessing import Process,Pool
# np.set_printoptions(threshold=sys.maxsize)
import warnings ## "i don't condone it but" was all I needed to hear from the stackoverflow forums
# from scoop import futures
import multiprocessing
import math
import scipy.spatial
import autolab_core.image
from scipy.interpolate import NearestNDInterpolator
import shutil
import matplotlib.patches as mpatches
##Some things to make all the text go away
from gqcnn.grasping.grasp import SuctionPoint2D
from PIL import Image

def one_pixel_attack(x,y,change_amount,np_file):
    og_data =np.load(np_file)
 
    fig, ax = plt.subplots(1, 1)
    ax.imshow(og_data, origin="lower",cmap="gist_gray")
                # from PIL import Image

    ax.axis("off")
    # plt.show()
    

    copy_data = og_data
    # print(copy_data.shape)
    x= round(np.float_(x)) ####TODO: FIX MUTATUON GIVINING US NON INTEGER NUMBERS
    y = round(np.float_(y)) ###MUTTUATION IS RETURNING FLOATS OF MY X,Y CAREFUL



    change_amount=np.float_(change_amount)

    if copy_data[y, x][0] +change_amount > 1:  ## just in case we don't wanna pass a negative depth value for future errors (Just a fail safe for now)
        copy_data[y, x] = 1  # have to push array trust bro
    else:
        if copy_data[y, x][0] < change_amount:  ## just in case we don't wanna pass a negative depth value for future errors (Just a fail safe for now)
            copy_data[y, x] = [0]
              # have to push array trust bro
        else:
            copy_data[y, x] = [copy_data[y, x][0] + change_amount]




    return copy_data


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def image_to_list_convotor(seg_mask):
    ## takes image gives numpy array for preocessing
    print(seg_mask)

    image = cv2.imread(seg_mask, 0)
    print(image)
    flippedimage = cv2.flip(image, 0)
    # img = mpimg.imread(flippedimage)
    # end
    cv2.imshow("sada", flippedimage)
    cv2.waitKey(200000)

    ist_of_valid_points_pre_poces = np.argwhere(flippedimage != 0)
    for k in ist_of_valid_points_pre_poces:
        print(k)



    return ist_of_valid_points_pre_poces


def Cricle_attack(x,y,radius,np_file):
    
    og_data = np.load(np_file)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(og_data, origin="lower",cmap="gist_gray")
                # from PIL import Image

    ax.axis("off")
    # plt.show()
    centre_x,centre_y = int(float(x)),int(float(y))
    
    copy_data = copy.deepcopy(og_data)
    limit_x  = og_data.shape[0]
    limit_y = og_data.shape[1]
    centre_x= round(centre_x)
    centre_y=round(centre_y)
    centre_z = copy.deepcopy(copy_data[centre_x, centre_y])

    radius = int(float(radius))
    for y_ in range (-radius,radius):
        for x_ in range(-radius,radius):
            distnace = y_**2+x_**2
            if distnace <radius**2:## to make sure it is in range
                # print(centre_y+y_, centre_x+x_)
                copy_data[centre_x+x_, centre_y+y_] = centre_z + math.sqrt( radius**2 - x_**2 - y_**2) * -0.003
                if copy_data[centre_x+x_, centre_y+y_][0] > 1:  ## just in case we don't wanna pass a negative depth value for future errors (Just a fail safe for now)
                    copy_data[centre_x+x_, centre_y+y_] = [1]  # have to push array trust bro

                if copy_data[centre_x+x_, centre_y+y_][0] < 0:  ## just in case we don't wanna pass a negative depth value for future errors (Just a fail safe for now)
                    copy_data[centre_x+x_, centre_y+y_] = [0]


    return copy_data





from mpl_toolkits import mplot3d
from PIL import Image as im 
import numpy as np


def depth_camera_setter(model,file_num=None,type=None):
  file_of_samples=[]

  # camera_intr_files = ["data/calib/phoxi/phoxi.intr","data/calib/primesense/primesense.intr"]  # TODO CHECK THIS ONE LATER
  if model =="FC-GQCNN-4.0-PJ" or model == "FC-GQCNN-4.0-SUCTION" or model == "GQCNN-4.0-SUCTION":
    camera_intr = "data/calib/phoxi/phoxi.intr"
  else:
    camera_intr = "data/calib/primesense/primesense.intr"

  if model== "GQCNN-4.0-SUCTION":
      pink_panther= "image_dataset_for_models/GQCNN-4.0-SUCTION"
      list_of_all = os.listdir(pink_panther)

  if model[0] == "F":
      pink_panther = "image_dataset_for_models/fcgqcnn"
      list_of_all = os.listdir(pink_panther)


  if model== "Dexnet2.1":
      pink_panther= "image_dataset_for_models/dexnet2.1"
      list_of_all = os.listdir(pink_panther)

  if model =="Dexnet2.0":
      pink_panther= "image_dataset_for_models/dexnet2.0"
      list_of_all = os.listdir(pink_panther)

    ## first thing in the list is the calibration stuff

  list_of_all.sort()
  depth_list = [pink_panther+"/"+x for x in list_of_all if "npy" in x ]
  segmak_list =  [pink_panther+"/"+x for x in list_of_all if "segmask" in x ]

  for num_of_sample in range(0,len(depth_list)):
      file_of_samples.append([depth_list[num_of_sample],segmak_list[num_of_sample]])

  return file_of_samples,camera_intr

for models in os.listdir("./logs"):
        
    for inside_models in sorted(os.listdir("./logs/" + models)):
            
        path_log_pareto = "./logs/" + models+"/"+inside_models
        curr_log_direct = "./logs/" + models + "/" + inside_models + "/end__results.csv"
        dataSets, camera_intr_files = depth_camera_setter(models)
        if "circle" in path_log_pareto:
            typeOfAttack = "circle"
        else:
            typeOfAttack = "one pixel"
        sample_num  = int(inside_models[inside_models.find('depth_') + 6])
        curr_dataSets = dataSets[sample_num]
        count =0
        if "Dex" in curr_log_direct: 
            with open(curr_log_direct, 'r') as file: ## error here logic forget to path it to the right file
                csvreader = csv.reader(file)
                for row in csvreader:
                    count =count +1
                    # sim_list.append([row[0],row[1],row[2]])
                    print(row)
                    if "one pixel" in path_log_pareto:
                        type__ = "driver"
                        lena =Cricle_attack(row[0],row[1],row[2],curr_dataSets[0])
                    else:
                        type__ = "phyiscal"
                        lena=  one_pixel_attack(row[0],row[1],row[2],curr_dataSets[0])

                    # lena = np.load("image_dataset_for_models/dexnet2.1/depth_0.npy")
                    # lena = Cricle_attack(150,150,59,lena)
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(lena, origin="lower",cmap="gist_gray")
                    # from PIL import Image

                    ax.axis("off")

                    plt.savefig(type__+str(count)+models+"base_img.png")
                    plt.clf()
                    image = cv2.imread(type__+str(count)+models+"base_img.png",cv2.IMREAD_GRAYSCALE)
                    xx = 59
                    yy = 82
                    ww =image.shape[1]-65
                    hh = image.shape[1]-226

                    crop_image = image[xx:hh, yy:ww]

                    cv2.imwrite(type__+str(count)+models+"base_img_crop.png", crop_image)
                    
                    lena = cv2.imread(type__+str(count)+models+"base_img_crop.png",cv2.IMREAD_GRAYSCALE)
                    xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]


                    
                    
                    ax = fig.add_subplot(111, projection='3d')

                    ax.set_box_aspect((9, 9, 9), zoom=4)
                    ax.set_aspect("auto")
                  
                    ax.set_axis_off()

                    ax.set_aspect('auto')
                    ax.plot_surface(xx, yy, lena ,rstride=1, cstride=1, cmap=plt.cm.cividis,antialiased=True
                                    )
                   
                    ax.view_init(azim=70, elev=-120)
                    plt.savefig(type__+str(count)+models+"3d_right_angle"+".png")
                    ax.view_init(azim=-70, elev=-120)
                    plt.savefig(type__+str(count)+models+"3d_left_angle"+".png")
                    ax.view_init(azim=160, elev=-165)
                    plt.savefig(type__+str(count)+models+"3d_top_right_angle"+".png")
                    ax.view_init(azim=-160, elev=-165)
                    plt.savefig(type__+str(count)+models+"3d_top_left_angle"+".png")

                    for k in [type__+str(count)+models+"3d_top_left_angle"+".png",type__+str(count)+models+"3d_top_right_angle"+".png",type__+str(count)+models+"3d_left_angle"+".png",type__+str(count)+models+"3d_right_angle"+".png"]:
                        print(k,"kings")
                        image = cv2.imread(k)
                        xx = 80
                        yy = 134
                        ww = image.shape[1] - 123
                        hh = image.shape[1] - 213 #right end
                        crop_image = image[xx:hh, yy:ww]

                        cv2.imwrite(k, crop_image)



cv2.imwrite("_for3d_.png", crop_image)

ax = fig.add_subplot(111, projection='3d')
# ax.set_box_aspect((1, 1, 1), zoom=1.8)
# ax.dist = 0
ax.set_box_aspect((9, 9, 9), zoom=4)
ax.set_aspect("auto")
#,rstride=2, cstride=2
ax.set_axis_off()

ax.set_aspect('auto')
ax.plot_surface(xx, yy, lena ,rstride=1, cstride=1, cmap=plt.cm.cividis,antialiased=True
                )
