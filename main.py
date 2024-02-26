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
import warnings 

import multiprocessing
import math
import scipy.spatial
import autolab_core.image
from scipy.interpolate import NearestNDInterpolator
import shutil
import matplotlib.patches as mpatches
from gqcnn.grasping.grasp import SuctionPoint2D
from PIL import Image
from matplotlib.ticker import MaxNLocator
import sys
import argparse
parser = argparse.ArgumentParser()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
matplotlib.use('agg')
warnings.filterwarnings("ignore")
tf.get_logger().setLevel(logging.ERROR) 







def sphere_maker(x,y,radius,np_file):
    """
          Preforms a one pixel attack on a certain file.
          
          Parameters
          ----------
          x : int
                location of the pixel on the x-axis.
          y : int
                location of the pixel on the y-axis.
          radius : int
                radius of the circle.
          np_file : str
                path to the numpy file.
       
          
          
         Returns
          Numpy array
            Changed numpy array
          """
    if not vaild_centre_postion([x, y]):

        x, y = pixel_to_nearest_valid_point(x, y)


    x,y,radius=limit_fuction(x,y,radius)

    centre_x,centre_y = x,y

    og_data = np.load(np_file)
    copy_data = copy.deepcopy(og_data)



    centre_x= round(centre_x)
    centre_y=round(centre_y)
    try:
        centre_z = copy.deepcopy(copy_data[centre_x, centre_y])
    except:
        if centre_x >= 386:
            centre_x = 386
        if centre_y >= 516:
            centre_y = 516

        centre_z = copy.deepcopy(copy_data[centre_x, centre_y])

    radius = round(radius)




    for y_ in range (-radius,radius):
        for x_ in range(-radius,radius):


            distnace = y_**2+x_**2
            if distnace <radius**2:## to make sure it is in range
                # print(centre_y+y_, centre_x+x_)
                copy_data[centre_x+x_, centre_y+y_] = centre_z + math.sqrt( radius**2 - x_**2 - y_**2) * -0.003


                #
                if copy_data[centre_x+x_, centre_y+y_][0] > 1:  ## just in case we don't wanna pass a negative depth value for future errors (Just a fail safe for now)
                    copy_data[centre_x+x_, centre_y+y_] = [1]  # have to push array trust bro
                # else:
                if copy_data[centre_x+x_, centre_y+y_][0] < 0:  ## just in case we don't wanna pass a negative depth value for future errors (Just a fail safe for now)
                    copy_data[centre_x+x_, centre_y+y_] = [0]
                    #     if copy_data[centre_y+y_, centre_x+x_][0] - change_amount < 0: ## just in case we don't wanna pass a negative depth value for future errors (Just a fail safe for now)
                #         copy_data[centre_y+y_, centre_x+x_] = [0]



                #
                #
                #           # have to push array trust bro
                #     else:
                #         copy_data[centre_y+y_, centre_x+x_] = [copy_data[centre_y+y_, centre_x+x_][0] + change_amount]
                # cv2.imshow("na",copy_data)
                #
                # cv2.waitKey(2)
    #

    return copy_data

def one_pixel_maker(x,y,change_amount,np_file):
    """
          Preforms a one pixel attack on a certain file.
          
          Parameters
          ----------
          x : int
                location of the pixel on the x-axis.
          y : int
                location of the pixel on the y-axis.
          change_amount : int
                the amount of change the pixel undergoes.
          np_file : str
                path to the numpy file.
       
          
          
         Returns
          Numpy array
            Changed numpy array
          """
   
    x_, y_, change_amount_ = limit_fuction(x, y, change_amount)
    og_data = np.load(np_file)

    copy_data = og_data
   
    x_= round(x_) 
    y_ = round(y_) 


  
    change_amount_=np.float_(change_amount_)

    if copy_data[y_, x_][0] +change_amount_ > 1:  
        copy_data[y_, x_] = 1  

    elif copy_data[y_, x_][0] < change_amount_:  
            copy_data[y_, x_] = [0]

              
    else:
        copy_data[y_, x_] = [copy_data[y_, x_][0] + change_amount_]

    if copy_data[y_, x_][0]<0:
        copy_data[y_, x_] = [0]

    if copy_data[y_, x_][0]>1:
        copy_data[y_, x_] = [1]


    

    return copy_data



def function_for_fitness_spliter(path_for_files, indvs):
    """
          Function for fitness and splitting based on the individual

          Parameters
          ----------
          path_for_files : str
               path for files
          indvs : list 
               a single individual with values
               
          
          
         Returns
          list 
            (old image q_value , absolute change amount)
          """

    enviro_eles=enviro_eles_global


    if typeOfAttack =="circle":



        return circle_attack(indvs[0],indvs[1],indvs[2],
                                        enviro_eles[0],enviro_eles[1],path_for_files)

    else:

        return one_pixel_attack(indvs[0],indvs[1],indvs[2],
                                          enviro_eles[0],enviro_eles[1],path_for_files)

def circle_attack(x,y,rad,file,enviroment_info,path_log_pareto):
    """
          Preforms a circle attack on a certain file.
          
          Parameters
          ----------
          x : int
                location of the pixel on the x-axis.
          y : int
                location of the pixel on the y-axis.
          rad : int
                the radius of the circle attack.
          file : str
                path to the numpy file.
          enviroment_info : list
                list with model and paths for enviroment .
          path_log_pareto : str
                path to the entire model file.
          
          
         Returns
          list 
            (old image q_value , absolute change amount)
          """
    global old_grasp_old_image



    npy_file = file[0]
    if os.path.exists(path_log_pareto+"/attacked_numpy.npy"):
        os.remove(path_log_pareto + "/attacked_numpy.npy")
        
    if ".npy" in npy_file:




        numpy_attacked_file = sphere_maker(y,x,rad,npy_file)
        np.save(path_log_pareto+'/attacked_numpy', numpy_attacked_file)



        if not vaild_centre_postion([y, x]):
            if old_grasp_old_image == None:
                old_grasp_old_image = poicy_sim(enviroment_info[0], enviroment_info[1], enviroment_info[2][0],enviroment_info[3], enviroment_info[4], enviroment_info[5], enviroment_info[6][1]).q_value

                return [old_grasp_old_image, rad]  #Catching a failed grasp
            else:
                return [old_grasp_old_image, rad]
        try:
            new_grasp_new_image = poicy_sim(enviroment_info[0], enviroment_info[1],  path_log_pareto+'/attacked_numpy.npy', enviroment_info[3],enviroment_info[4], enviroment_info[5], enviroment_info[6][1])
        except:

            return [0,rad]

        if new_grasp_new_image.q_value == None:
            return [0,rad]

        return(new_grasp_new_image.q_value,rad) 

def one_pixel_attack(x,y,change_amount,file,enviroment_info,path_log_pareto):
    """
          Preforms a one pixel attack on a certain file.
          
          Parameters
          ----------
          x : int
                location of the pixel on the x-axis.
          y : int
                location of the pixel on the y-axis.
          change_amount : int
                the amount of change the pixel undergoes.
          file : str
                path to the numpy file.
          enviroment_info : list
                list with model and paths for enviroment .
          path_log_pareto : str
                path to the entire model file.
          
          
         Returns
          list 
            (old image q_value , absolute change amount)
          """
    global method_of_attack



    npy_file = file[0]
    if os.path.exists(path_log_pareto+"/attacked_numpy.npy"):
        os.remove(path_log_pareto+"/attacked_numpy.npy")



 
    
    if ".npy" in npy_file:

        numpy_attacked_file = one_pixel_maker(x,y,change_amount,npy_file)
        np.save(path_log_pareto+'/attacked_numpy', numpy_attacked_file)
    
        new_grasp_new_image = poicy_sim(enviroment_info[0], enviroment_info[1], path_log_pareto+'/attacked_numpy.npy', enviroment_info[3],enviroment_info[4], enviroment_info[5], enviroment_info[6][1])

        new_grasp_old_image = phyiscal_sim(new_grasp_new_image, enviroment_info[0], enviroment_info[1], enviroment_info[2][0], enviroment_info[3],enviroment_info[4], enviroment_info[6][1])





        return(new_grasp_old_image.q_value,abs(change_amount))

global global_All_attacks
global_All_attacks = []

def monitorPlot(values, filename=None):
  global  global_All_attacks
  v = values

  if v.shape != (MU,2,):
      return []

  plt.clf()


  v = v[v[:,0].argsort()]

  global_All_attacks.append(v)
  plt.plot(v[:,0], v[:,1], "-ro")
  if typeOfAttack == "one pixel":
      plt.ylabel("Intensity change")
      plt.xlabel("Fitness")
  else:
      plt.ylabel("Radius")
      plt.xlabel("Fitness")


  plt.tight_layout()
  for i, j in zip(v[:,0], v[:,1]):

      anooot = f'{i:.3}'+","f'{j:.3}'
      plt.annotate((anooot), xy=(i, j))

  plt.draw()
  if filename == None:
    plt.savefig("esnsga-pareto-"+str(attack_num)+".png", transparent=True)

  else:
    plt.savefig(filename, transparent=True)
  plt.pause(0.001)
  attack_count = 0
  return []

def individual_genator(icls,scls): ### this needs some fixing, I will consider adding some tests but I am unsure if I make someone with fitness inbuilt that might be case

    if typeOfAttack != "one pixel":
        global curr_valid_pixels
        radius = random.randint(1, 20)
        random_point = np.random.randint(0,len(curr_valid_pixels))
        y,x = curr_valid_pixels[random_point]
        
        ind= icls((x,y,radius))
        ind.strategy = scls(random.uniform(MIN_STRATEGY, MAX_STRATEGY) for _ in range(3)) 
        return ind
    else:
        x = random.randint(0, 515)
        y = random.randint(0, 385)
        stronk = random.uniform(-1, 1)
       
        ind = icls((x, y, stronk))
        ind.strategy = scls(random.uniform(MIN_STRATEGY, MAX_STRATEGY) for _ in range(3)) 
        return ind

def limit_fuction(a,b,c):
    ## Limit function
    if typeOfAttack == "circle": # edited for diagarming ## RECHECK
        
        a = round(a)
        b= round(b)
        if a < 0:
            a = 0
        if b < 0:
            b = 0


        if c >20:
            c = 20

        if c<1:
            c = 1

        return (a,b,c)
    else:
        print("one pixel lim")
        if a < 0:
            a = 0
        if b < 0:
            b = 0
        if b > 385:

            b = 385

        if a > 515:

            a = 515

        if c > 0 and c < 0.01:


            c = 0.01

        if c < 0 and c > -0.01:
            c= -0.01

        if c > 1:
            c = 1

        if c < -1:
            c = -1

        return (a, b, c)

def statAvg(pop):
  fit = np.array([pop[i].fitness.values for i in range(len(pop))])
  return np.mean(fit, axis=0)
def statMin(pop):
  fit = np.array([pop[i].fitness.values for i in range(len(pop))])
  return np.min(fit, axis=0)
def statMax(pop):
  fit = np.array([pop[i].fitness.values for i in range(len(pop))])
  return np.max(fit, axis=0)
list_of_accurs=[]
list_of_accurs_1=[]
list_of_accurs_2=[]
attack_num=0
def statLog(path_log_pareto,pop):
  """
          makes an imposed parteo curve graph from the file path
          
          Parameters
          ----------
          path_log_pareto : str
               path to file of the model.
          pop : int
                population of the generation
          
         Returns
         Not applicable
          """  
  f = []
  global  list_of_accurs 
  global list_of_accurs_1
  global list_of_accurs_2
  global attack_num

  for ind in pop:
     f.append(ind.fitness.values)
     list_of_accurs_1.append(ind.fitness.values[0])
     list_of_accurs_2.append(ind.fitness.values[1])
  plt.clf()

  
  fit = np.array(f)
   
  
  LOG_DIR = path_log_pareto
  monitorPlot(fit, filename = LOG_DIR+"/esnsga-pareto-"+str(attack_num)+"-graph.png") ## check monitor plot function
  # save population
  if attack_num == NGEN-1:
    fname = "%s/end__results.csv" % (LOG_DIR)


  else:
    fname = "%s/esnsga-pareto-%s-pop.csv" % (LOG_DIR, str(attack_num)) ### like actaully what?
  f = open(fname, "w")
  writer = csv.writer(f, lineterminator = "\n")
  for ind in pop:
      ext = array.array('d',ind) 
      ext.extend(ind.strategy)
      writer.writerow(ext)
  f.close()
 

  if attack_num == NGEN:
    fname = "%s/end__pop.csv" % (LOG_DIR)



  else:
    fname = "%s/esnsga-pareto-%s-values-real.csv" % (LOG_DIR, str(attack_num))
  f = open(fname, "w")
  writer = csv.writer(f, lineterminator = "\n")
  for i in range(len(pop)):
      ind = pop[i]
      writer.writerow(fit[i])
  f.close()
  fitness_graphing_function(LOG_DIR + "/esnsga-pareto-" + str(attack_num)+"-graph_zeal", method_of_attack=method_of_attack)


  if attack_num == NGEN:
    attack_num = 0
  else:
    attack_num = attack_num + 1
  return []

def checkStrategy(minstrategy):
  def decorator(func):
      def wrappper(*args, **kargs):
          children = func(*args, **kargs)
          for child in children:
              for i, s in enumerate(child.strategy):
                  if s < minstrategy:
                      child.strategy[i] = minstrategy
          return children
      return wrappper
  return decorator

def checkBounds():
  def decorator(func):
      def wrapper(*args, **kargs):
          offspring = func(*args, **kargs)
          
          for child in offspring:
              

              child[0],child[1],child[2]=limit_fuction(child[0],child[1],child[2])






          return offspring
      return wrapper
  return decorator


def global_enviroment_setter_for_diagarms():
    """
        Creates and maintains all the diagarm making by extracting global variables from the paths 
          
   
          
         Returns
         Not applicable
          """
    #state setter, it makes the globals and sets the values based on past experiments, a life saver for time if a diagarm wants to be changed)
    global global_All_attacks
    global list_of_accurs_1
    global list_of_accurs_2
    global enviro_eles_global
    global typeOfAttack
    global attack_num
    global old_grasp_old_image
    old_grasp_old_image = None

    for models in os.listdir("./logs"):
        for inside_models in sorted(os.listdir("./logs/" + models)):
            
            path_log_pareto = "./logs/" + models+"/"+inside_models
            curr_log_direct = "./logs/" + models + "/" + inside_models + "/end__pop.csv"
            dataSets, camera_intr_files = depth_camera_setter(models)
            if "circle" in path_log_pareto:
                typeOfAttack = "circle"
            else:
                typeOfAttack = "one pixel"
            sample_num  = int(inside_models[inside_models.find('depth_') + 6])
            curr_dataSets = dataSets[sample_num]
            depth_segment_filenames_mult = [curr_dataSets]
            
            pareto_lists=[]
            config_path = "./models" + models + "/config.json"
            enviro_eles_global = [depth_segment_filenames_mult[0],[models, 'models/', depth_segment_filenames_mult[0], camera_intr_files, config_path, False, depth_segment_filenames_mult[0]]]
            if os.path.exists(curr_log_direct):

                for each_subfile in sorted(os.listdir("./logs/" + models + "/" + inside_models)):

                    if "values-real" in each_subfile:
                        pareto_lists.append(each_subfile)

                pareto_lists.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
                pareto_lists.append("end__pop.csv")

                for popfiles in pareto_lists:

                        batch_list = list()
                        with open("./logs/" + models + "/" + inside_models +"/"+ popfiles, 'r') as file: 
                            csvreader = csv.reader(file)
                            
                            for row in csvreader:
                                row =np.array([float(row[0]),float(row[1])])
                                list_of_accurs_1.append(row[0])
                                list_of_accurs_2.append(row[1])
                                batch_list.append(row) 
                        batch_list = np.array(batch_list)
                        batch_list = batch_list[batch_list[:, 0].argsort()]
                        if not len(batch_list) >11:
                            
                        
                            global_All_attacks.append(np.array(batch_list))

            global_All_attacks = np.array(global_All_attacks)




            
        
            global curr_valid_pixels
            curr_valid_pixels = image_to_list_numpy_array(depth_segment_filenames_mult[0][1])

            
            
            zoom_reigion_mass_feeder(path_log_pareto)

            clear_global_varaibles_function() 
            
          



    




def imposed_pareto_curve(file_path):
  """
          Makes an imposed parteo curve graph from the file path
          
          Parameters
          ----------
          file_path : str
               path to file of the model.
          
          
         Returns
         Not applicable
          """
  global global_All_attacks
  global list_of_accurs_1
  global list_of_accurs_2
  global enviro_eles_global

  for title_ in ["title","No_title"]:
      v=  copy.deepcopy(global_All_attacks)
      data_lister = []

       ### removing first 100

      gen= 0

      for order in range(0,2):
          gen = 0
          plt.clf()
          fig, ax = plt.subplots(figsize=(9, 7))
          ax.tick_params(axis='both', which='major', labelsize=25)
          for  values_ in v:

            values_ = values_[values_[:, 0].argsort()]

            if order ==0:

                if gen == len(v) - 1:
                    plt.plot(values_[:, 0],  values_[:, 1], label="The {} gen".format(gen), color="green",linewidth=2.5)

                else:
                    plt.plot(values_[:, 0],  values_[:, 1], label="The {} gen".format(gen),
                             alpha=(gen + 1) / (len(v) + 1), color="black")

                plt.xlabel("Grasp quality", fontsize=22)
                if "one pixel" in file_path:
                    plt.ylabel("Intensity change",fontsize=22)
                else:
                    plt.ylabel("Radius",fontsize=22)
            if order == 1:

                if gen == len(v)-1:
                      plt.plot(values_[:, 1], values_[:, 0], label="The {} gen".format(gen),color="green")

                else:
                      plt.plot(values_[:, 1], values_[:, 0], label="The {} gen".format(gen),alpha=(gen+1)/(len(v)+1),color="black" )

                plt.ylabel("Grasp quality",fontsize = 22)
                if "one pixel" in file_path:
                    plt.xlabel("Intensity change",fontsize=22)
                else:
                    plt.xlabel("Radius",fontsize=22)

            gen = gen + 1

          sample_str_file = enviro_eles_global[0][0]
          sample_str_file = sample_str_file[sample_str_file.find('/') + 1:]
          sample_str_file = sample_str_file[sample_str_file.find('/') + 1:]
          sample_str_file = sample_str_file[:-4]

          if "one pixel" in file_path:
              method_of_attack = "driver"
          else:
              method_of_attack = "phyiscal "
          if title_ == "title":
             plt.title("\n".join(wrap("Generational graph of the with model {} on {} being attacked via a {}".format( enviro_eles_global[1][0],sample_str_file,method_of_attack), 60)))


          plt.tight_layout()
          plt.draw()
          ax.tick_params(axis='both', which='major', labelsize=24)
          plt.savefig(file_path + sample_str_file+"_imposed_pareto"+str(order)+title_+method_of_attack+".png", transparent=False)
          plt.clf()





def fitness_graphing_function(file_path,method_of_attack):
  """
          Graph the fitness function of the provided attack
          
          Parameters
          ----------
          file_path : str
               path to file of the model.
          method_of_attack : str
               the type of attack
          
         Returns
         Not applicable
          """
  global list_of_accurs_1
  global list_of_accurs_2

  for title_ in ["title", "No_title"]:
      plt.clf()
      fig, ax = plt.subplots(figsize=(6, 4))
      ax.tick_params(axis='both', which='major', labelsize=15)
      lister = [list_of_accurs_1,list_of_accurs_2]
      if "one pixel" in file_path:
          template_text = "intensity change"
      else:
          template_text = "radius "
      for index1,fitness in enumerate(lister):

        split_1 = copy.deepcopy(fitness[:100])
        fitness = copy.deepcopy(fitness[100:])
        dict_for_title = {0:"Grasp quality",1:template_text}
        rest_split = [fitness[x:x + 10] for x in range(0, len(fitness), 10)]
        min_first = min(split_1)
        list_to_return = [min_first]
        avg_first = max(split_1)
        list_to_resend = [sum(split_1)/len(split_1)]
        list_to_return_avg = [avg_first]

        for index, splits in enumerate(rest_split):
          list_to_return.append(min(rest_split[index]))
          list_to_return_avg.append(max(rest_split[index]))
          list_to_resend.append(sum(rest_split[index]) / len(rest_split[index]))
        gen_liost = [x + 1 for x in range(0, len(list_to_return))]


        plt.plot(gen_liost, list_to_return, label="Min")
        plt.plot(gen_liost, list_to_return_avg, label="Max")
        plt.plot(gen_liost, list_to_resend, label="Avg")

        plt.legend()

        sample_str_file = enviro_eles_global[0][0]
        sample_str_file = sample_str_file[sample_str_file.find('/') + 1:]
        sample_str_file = sample_str_file[sample_str_file.find('/') + 1:]
        sample_str_file = sample_str_file[:-4]
        if title_ == "title":
            plt.title("\n".join(wrap("Stats of the {} with model {} on {}  {} type attack ".format(dict_for_title[index1], enviro_eles_global[1][0], sample_str_file,method_of_attack),60)))
        plt.ylabel(dict_for_title[index1],fontsize=16)
        plt.xlabel("Generation",fontsize=16)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        plt.draw()
        plt.savefig(file_path + sample_str_file+"_avg_of_"+str(dict_for_title[index1])+title_+method_of_attack+".png", transparent=True)
        plt.pause(0.001)
        plt.clf()

def depth_camera_setter(model,file_num=None,type=None):
  """
          Make a list of avaible points for the individual and check  against the segment mask
          
          Parameters
          ----------
          file_path : str
               path to file of the model.
          
         Returns
         Not applicable
          """
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

   

  list_of_all.sort()
  depth_list = [pink_panther+"/"+x for x in list_of_all if "npy" in x ]
  segmak_list =  [pink_panther+"/"+x for x in list_of_all if "segmask" in x ]

  for num_of_sample in range(0,len(depth_list)):
      file_of_samples.append([depth_list[num_of_sample],segmak_list[num_of_sample]])

  return file_of_samples,camera_intr


def zoom_reigion_mass_feeder(file_path):
    """
          Make a list of avaible points for the individual and check  against the segment mask
          
          Parameters
          ----------
          file_path : str
               path to file of the model.
          
         Returns
         Not applicable
          """
    sim_list = []
    
    if "end__results.csv" in os.listdir(file_path):
        with open(file_path+"/end__results.csv", 'r') as file: ## error here logic forget to path it to the right file
            csvreader = csv.reader(file)
            for row in csvreader:
                sim_list.append([row[0],row[1],row[2]])
               

        if "one pixel" in file_path:
            type__ = "driver"
        else:
            type__ = "phyiscal"

        for i,things in enumerate(sim_list):

            x,y,z = things
            x = round(float(x))
            y = round(float(y))
            z=np.float_(z)

            
            Zoom_reigion(enviro_eles_global[0][0],file_path+"/"+type__+"grasp_"+str(i)+"_",x,y,change_amount=z)
            

def clear_global_varaibles_function():
    
    global method_of_attack
    global global_All_attacks
    global_All_attacks = []
    global list_of_accurs
    global list_of_accurs_1
    global list_of_accurs_2
    global attack_num
    list_of_accurs = []
    list_of_accurs_1 = []
    list_of_accurs_2 = []
    attack_num = 0
    global old_grasp_old_image
    old_grasp_old_image=None


def attack_for_UI(x,y,rad,npy_file,enviroment_info,path_log_pareto):
    """
          Make a list of avaible points for the individual and check  against the segment mask
          
          Parameters
          ----------
          seg_mask : str
               path to segment mask of the model.
          
         Returns
         -------
         numpy array
                list of valid points
          """
    
    old_grasp_old_image = poicy_sim(enviroment_info[0], enviroment_info[1], enviroment_info[2][0],
                                    enviroment_info[3], enviroment_info[4], enviroment_info[5],
                                    enviroment_info[6][1])

    

    if os.path.exists(path_log_pareto+"/attacked_numpy.npy"):
        os.remove(path_log_pareto + "/attacked_numpy.npy")
        
    
    if ".npy" in npy_file:
        if "phyiscal" in path_log_pareto:
            numpy_attacked_file = sphere_maker(y,x,rad,npy_file)
            np.save(path_log_pareto+'/attacked_numpy', numpy_attacked_file)
        if "driver" in path_log_pareto:
            print("updated one pixel ")
            numpy_attacked_file = one_pixel_maker(x, y, rad, npy_file)
            np.save(path_log_pareto + '/attacked_numpy', numpy_attacked_file)


    if "phyiscal" in path_log_pareto:
        if not vaild_centre_postion([y, x]):
            the_grasp = old_grasp_old_image

        
        try:
            the_grasp = poicy_sim(enviroment_info[0], enviroment_info[1], path_log_pareto + '/attacked_numpy.npy',
                              enviroment_info[3], enviroment_info[4], enviroment_info[5], enviroment_info[6][1])
        except:
            return (0, [old_grasp_old_image.grasp.center], [old_grasp_old_image.grasp.center],
                    old_grasp_old_image.q_value)


    if "driver" in path_log_pareto:
        
            
            the_grasp = poicy_sim(enviroment_info[0], enviroment_info[1], path_log_pareto + '/attacked_numpy.npy',
                              enviroment_info[3], enviroment_info[4], enviroment_info[5], enviroment_info[6][1])
            # print(enviroment_info[1], enviroment_info[2][0], enviroment_info[3],enviroment_info[4], enviroment_info[6][1])
            the_grasp = phyiscal_sim(the_grasp, enviroment_info[0], enviroment_info[1],enviroment_info[2][0], enviroment_info[3],enviroment_info[4], enviroment_info[6][1])

            return (the_grasp.q_value,[the_grasp.grasp.center],[old_grasp_old_image.grasp.center],old_grasp_old_image.q_value)

        


    return(the_grasp.q_value,[the_grasp.grasp.center],[old_grasp_old_image.grasp.center],old_grasp_old_image.q_value) ## to enable the method of fitness to be the same it might need to be negative

def Zoom_reigion(file_name, save_path, x=None, y=None,change_amount=None,radius=None): #
    """
          Parameters
          ----------
          file_name: str
              Python string of the numpy file.
          save_path: str
              Python string for the location to save the diagarm files.
          x : int
              Python int that holds the X  value the attack will occur in.
          y : int
               Python int that holds the Y value the attack will occur in.
          change_amount : int
              Python int that determines the change of the value can be positive or negative. Only used with a one pixel attack
          radius : int
              Python int that determines the radius of the sphere attack  can only be positive . Only used with a sphere attack

          Notes
          -----
          This function assumes the end results are called end__results.csv
          Matpotlib needs the try/catch bypass

          Returns
          -------
          Not applicable
          """
    ## we might need to mulitply the difference
    ## These pieces of code will, and shall be lost to the time. Not even my IDE knows how this works#
    # x,y,rad,file,enviroment_info,path_log_pareto


####fix

    if  "circle" in save_path:
        print("Circle Pixel Attack")
        quality,center,og_center,og_quality = attack_for_UI(x,y,change_amount,enviro_eles_global[0][0],enviro_eles_global[1],save_path[:-17])
    else:
        print("Single Pixel Attack")
        quality,center,og_center,og_quality = attack_for_UI(x,y,change_amount,enviro_eles_global[0][0],enviro_eles_global[1],save_path[:-15])
    curr_quality = quality
    # temp__ = temp_ui_values




    if  "one pixel" in save_path:
        file_to_read = one_pixel_maker(x, y, change_amount, file_name)
        offset_ = 2
        print("one pixel attack 2D")
        text_print = "Change ="
    else:
        file_to_read = sphere_maker(y, x, change_amount, file_name)
        offset_ = 1 + change_amount
        text_print = "Radius ="

        print("sphere attack 2D")
    ## Zoom in before the change
    # save_path.
    
    # print(save_path.rsplit('/', 1)[-1::]+"/attacked_numpy.npy")

    img_array = copy.deepcopy(file_to_read)

    fig, (ax, oe) = plt.subplots(1, 2, sharex=True, sharey=True)

    oe.axis("off")
    ax.axis("off")
    # ok = ax.imshow(img_array, cmap='gray')

    ax.imshow(img_array, origin="lower", cmap="gray")

    if center[0][1] == og_center[0][1] and center[0][0] == og_center[0][0]:
        d =  mpatches.Wedge((og_center[0][0],og_center[0][1]), 8,   90,  270 ,color="blue",label="New grasp")
        c = mpatches.Wedge((center[0][0],center[0][1]), 8, 270, 90, color="red",label="Old grasp")
        ax.add_patch(c)
        ax.add_patch(d)
    else:
        if curr_quality != 0:
            c = plt.Circle((center[0]), radius=8.4, label='New grasp')
            ax.add_patch(c)
        d = plt.Circle((og_center[0]), radius=8, label='Old grasp', color="red")
        ax.add_patch(d)

    ny, nx, nz = img_array.shape
    a = 50
    b = 50
    try:

        z2 = np.zeros((a, b))
        z2[30:30 + ny, 30:30 + nx] = img_array

    except Exception as e:
        just_for_ = ""
    
    axins = ax.inset_axes([1, 0, 0.47, 0.47])
    axins.imshow(img_array, origin="lower", cmap="gray")

    # subregion of the original image

    x1, x2, y1, y2 = x - offset_, x + offset_, y - offset_, y + offset_

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    axins.set_xticklabels([])
    axins.set_yticklabels([])

    random__ = copy.deepcopy(np.load(file_name))

    ny2, nx2, nz2 = random__.shape
    a = 50
    b = 50
    try:

        z23 = np.zeros((a, b))
        z23[30:30 + ny2, 30:30 + nx2] = random__

    except Exception as e:
        just_for_ = ""
    # inset axes....

    axins2 = oe.inset_axes([-0.139, 0.45, 0.35, 0.35])
    axins2.imshow(random__, origin="lower", cmap="gray")
    axins2.set_title("Original",fontsize = 10)
    # subregion of the original image

    x1, x2, y1, y2 = x - offset_, x + offset_, y - offset_, y + offset_

    axins2.set_xlim(x1, x2)
    axins2.set_ylim(y1, y2)
    axins2.set_xticklabels([])
    axins2.set_yticklabels([])

    # ax.indicate_inset_zoom(axins2, edgecolor="black")

    ax.indicate_inset_zoom(axins, edgecolor="black")

    axins.axis('off')
    axins2.axis('off')
    ###################################################

    # cv2.imshow("bas23e", img_array)
    # cv2.imshow("base", random__)
    # cv2.waitKey(200000)

    sample_str_file = enviro_eles_global[0][0]
    sample_str_file = sample_str_file[sample_str_file.find('/') + 1:]
    sample_str_file = sample_str_file[sample_str_file.find('/') + 1:]
    sample_str_file = sample_str_file[:-4]
    if curr_quality >0:
            text ="Original\nquality ="+f'{og_quality:.3}'+ "\n"*4 +"Quality =" +f'{curr_quality:.3}'+ "\n" + text_print + f'{round(change_amount,3):.3}'
    else:
            text = "Original\nquality =" + f'{og_quality:.3}' + "\n" * 4 + "Quality =" + "0" + "\n" + text_print+ f'{round(change_amount,3):.3}'
        # text = "quality =" + f'{curr_quality:.3}' + "\n" + "radius =" + f'{radius:.3}'
    # f'{i:.3}'+","f'{j:.3}
    ax.annotate(text, xy=(1.45, 0.16), xycoords='axes fraction', fontsize=11) # 0.2
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
    #                     hspace=0, wspace=0)
    # plt.margins(0, 0)
    if curr_quality != 0:
        ax.legend(handles=[c,d],fontsize="9.3",loc ="lower left")
    else:
        ax.legend(handles=[d],fontsize="9.3", loc = "lower left")
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # axins.legend(loc='upper right', fontsize=19)
    plt.savefig(save_path+sample_str_file+enviro_eles_global[1][0]+".png", transparent=False,bbox_inches='tight',pad_inches=0.1)
    
def pixel_to_nearest_valid_point(x,y):
    ##
    """
          Parameters
          ----------
          seg_mask : str
               path to segment mask of the model.
          

          Notes
          -----
          Makes the directory used in the simluation instance

         Returns
         -------
         numpy array
                list of valid poinnts
          """
    global curr_valid_pixels

    xy1 = np.array([(x,y)])

    new_coordinates = curr_valid_pixels[np.argmin(scipy.spatial.distance.cdist(xy1, curr_valid_pixels), axis=1)]

    y2,x2 = new_coordinates[0][0],new_coordinates[0][1]

    return np.array([x2,y2])
# Constraint function
def vaild_centre_postion(individual):
  """       
          Parameters
          ----------
          individual : list
               the simluation individual.

           Notes
           -----
           Returns whether or not the pixel is in the correct location   .

  """
  xy =  np.array(individual)
  global curr_valid_pixels

  return  xy in curr_valid_pixels


def main(method,model):
    # MIN_STRATEGY, MAX_STRATEGY = 1, 100  # standard deviation of the mutation. orig: 50, 100
    # MU, LAMBDA = 10, 100  # mu number of individuals to select, lambda number of children to produce. orig: 20,200. good too: 10,200
    # NGEN = 20

    

    """
          Parameters
          ----------
          method : str
               choose if it is a one-pixel or circle attack
          model : str
               Name model.
               
        
               
          Notes
          -----
          Makes the directory used in the simluation instance

         Returns
         -------
               not applicable 
          """

    model_start = time.time()
    global NGEN
    
    global typeOfAttack
    global attack_num
    global old_grasp_old_image
    global curr_valid_pixels
    old_grasp_old_image = None
    if method == "driver":
        type_of_atack = "one pixel"
    else:
        type_of_atack = "circle"
    typeOfAttack = type_of_atack
    models = {"gqcnn_suction": "GQCNN-4.0-SUCTION", "fc-gqcnn_suction": "FC-GQCNN-4.0-SUCTION","fc-gqcnn_parellel_grasp": "FC-GQCNN-4.0-PJ","dex2.1":"Dexnet2.1","dex2.0":"Dexnet2.0"}

    chosen = []

    methods = {"driver": "driver attack", "physical": "physical attack"}

    global method_of_attack
    method_of_attack = methods[method]

    if model == "All":
        chosen = os.listdir('./models')

    elif model == "Lowest":
        models_dict = {}

        for k in os.listdir('./models'):
            models_dict[k] = len(os.listdir('./logs/' + k))

        for l, o in models_dict.items():
            if o == min(models_dict.values()):
                chosen.append(l)

    else:
        chosen = [models[model]]
    for models_in_path in chosen: ### TRIES EVERYTHING IN path MODELS

        dataSets, camera_intr_files = depth_camera_setter(models_in_path)
        for data_piece in dataSets:

            depth_segment_filenames_mult = [data_piece]


            curr_valid_pixels = image_to_list_numpy_array(data_piece[1])

            attack_num = 0





            ### Model GQ-Bin-picking-Eps50 abd Eos10 removed dyue to not having config.json
            model_for_evulation = models_in_path

            print()
            print() ## Allows some needed space ya feel me?
            print("----------------------------------------")
            print("A {} looking {} on {} {}/{} it is trying on {} ".format(typeOfAttack,method_of_attack,model_for_evulation ,chosen.index(model_for_evulation)+1,len(chosen),data_piece[1]))
            print("starting time is ",model_start )
            print("it has been {} since starting time ".format((time.time()-model_start)/60))
            print("-----------------------------------------------")
            models_path = "models/"

            string_for_exact_sample = depth_segment_filenames_mult[0][0][depth_segment_filenames_mult[0][0].index("/")+1:depth_segment_filenames_mult[0][0].index(".")]

            path_log_pareto = log_directory_creator(model_for_evulation,method_of_attack,typeOfAttack,depth_segment_filenames_mult[0][0][:-4])

            model_path_global=models_path+model_for_evulation+"/config.json"


            config_path = models_path + model_for_evulation + "/config.json"



            global enviro_eles_global

            enviro_eles_global = [depth_segment_filenames_mult[0], [model_for_evulation, 'models/', depth_segment_filenames_mult[0], camera_intr_files, config_path, False, depth_segment_filenames_mult[0]]]

           

            creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0,))
            creator.create("Individual", list, typecode="d", fitness=creator.FitnessMin, strategy=None)  ##type code?
            creator.create("Strategy", list, typecode="d")


            toolbox = base.Toolbox()
            toolbox.register("evaluate", function_for_fitness_spliter, path_log_pareto)


            toolbox.register("individual", individual_genator, creator.Individual, creator.Strategy)

            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register("mate", tools.cxESBlend, alpha=0.1)
            toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
            toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
            toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))
            toolbox.decorate("mate", checkBounds())
            toolbox.decorate("mutate", checkBounds())


            toolbox.register("select", tools.selSPEA2) ##



            mstats = tools.Statistics()
            mstats.register("avg", statAvg,)
            mstats.register("min", statMin)
            mstats.register("max", statMax)
            mstats.register("log", statLog,(path_log_pareto))


            pop = toolbox.population(n=LAMBDA)
          



            hof = tools.HallOfFame(1)
            pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, cxpb=0.6, mutpb=0.3, ngen =NGEN,
                                                         stats=mstats, halloffame=hof)
               

           
            # plt.clf()
            print("ending time is {} since starting time ".format((time.time()-model_start)/60))


            clear_global_varaibles_function()

def create_bar_plot():
    """

           Notes
           -----
           Makes the bar plot of both lowest quality and radius/Intensity change in the models, Saves to local path.

           Returns
           -------
           Not applicaible 

           """

    list_of_models_formatting = ["FC-GQCNN\n4.0-PJ", "FC-GQCNN\n4.0-SUCTION", "GQCNN-4.0\nSUCTION", "Dexnet 2.1", "Dexnet 2.0"]
    list_of_models_iterating = ["FC-GQCNN-4.0-PJ", "FC-GQCNN-4.0-SUCTION", "GQCNN-4.0-SUCTION", "Dexnet2.1", "Dexnet2.0"]
    print("Be sure to remove the sampled AVG and STD")
    list_of_baselines_avg =[0.995625,0.966,0.945,0.689,0.518]  #remove these for the ones generated in the diagarm sample
    base_std = [0.0025713566458195132, 0.021415377652518764, 0.007301506693826973, 0.16312770732159512,
                0.3374112837472985]
    for tile_ in ["title","No_title"]:


        dict_for_formatting = {0:"Quality of grasp"}
        for types in ["phyiscal","driver"]:
            fig, ax = plt.subplots(figsize = (14,10))

            
            quality =[]
            list_of_std=[]
            temp_storage = []
            list_of_mins = []
            for k in list_of_models_iterating:
                for x in os.listdir("./"+"logs"+"/"+k+"/"):
                    curr_log_direct = "./logs/"+k+"/"+x+ "/end__pop.csv"
                    if (os.path.exists(curr_log_direct) and types in curr_log_direct):

                        with open(curr_log_direct, 'r') as file:  ## error here logic forget to path it to the right file
                            csvreader = csv.reader(file)
                            for row in csvreader:
                                temp_storage.append(float(row[0]))##adding fitness to the dict
                                

                        quality.append(min(temp_storage))

                        temp_storage= []
                    else:
                        if not os.path.exists(curr_log_direct):
                            shutil.rmtree("./logs/"+k+"/"+x)

                 
                list_of_mins.append(np.mean(quality))

                list_of_std.append(np.std(quality))
                quality = []


            

            x = np.arange(5)
            for i,k in enumerate(list_of_baselines_avg):
                list_of_baselines_avg[i] = round(k,3)

            for i,k in enumerate(list_of_mins):
                list_of_mins[i] = round(k,3)


            bar_org = ax.bar(x, list_of_baselines_avg, width=0.33, yerr=base_std, label = "Originial")
            ax.bar_label(bar_org,padding=3,fontsize=22)
            attack_bar = ax.bar(x+0.35, list_of_mins, color='maroon', width=0.37, yerr=list_of_std, label="Attacked")
            ax.bar_label(attack_bar, padding=3,fontsize=22)
            
            plt.ylabel("Grasp quality",fontsize=29)
            ax.legend(loc='upper right',fontsize=19)
            print(sum(list_of_std),types)
            if tile_ != "No_title":
              plt.title("Lowest {} of models tested using a {} attack".format(dict_for_formatting[0],types))
            plt.draw()
            plt.show()
            ax.tick_params(axis='y', which='major', labelsize=29)
            ax.set_xticks(x + 0.15, list_of_models_formatting,fontsize=26)
            plt.tight_layout()
            plt.savefig("./Quality of grasp"+types +tile_, transparent=False)
            plt.clf()

def image_to_list_numpy_array(seg_mask):
    """
          Make a list of avaible points for the individual and check  against the segment mask
          
          Parameters
          ----------
          seg_mask : str
               path to segment mask of the model.
          
         Returns
         -------
         numpy array
                list of valid points
          """


    seg_mask="./"+seg_mask

    image = cv2.imread(seg_mask,0) #makes it grey and it works out well



    list_of_valid_points_pre_poces = np.argwhere(image != 0)


    return list_of_valid_points_pre_poces



def log_directory_creator(model,method_of_attack, type_of_attack,attacked_file):
    """
          Makes the directory used in the simluation instance

          Parameters
          ----------
          model : str
              Python str model name.
          method_of_attack : str
              Python str of method of attack (physical or driver).
          type_of_attack : str
              Python str of method of attack (sphere or one pixel).
          attacked_file : str
              Python str of original numpy file).

    
         Returns
         -------
         String
              The path for the simluation
          """

    amount_of_logs =0

    path = "./logs/" + model
    if not os.path.exists(path):
        os.mkdir(path)

    for file in os.listdir(path):
        if method_of_attack in file and type_of_attack in file:
            amount_of_logs = amount_of_logs + 1

    past_log_file = str(amount_of_logs - 1)
    amount_of_logs = str(amount_of_logs)

    if past_log_file == "-1":
        past_log_file = "0"

    attacked_file = attacked_file[attacked_file.find('/') + 1:]
    attacked_file = attacked_file[attacked_file.find('/') + 1:]


    if not os.path.exists(path + "/"+type_of_attack + attacked_file+ method_of_attack + "Attempt_num_" + past_log_file):
        os.mkdir(path + "/" + type_of_attack+ attacked_file+method_of_attack + "Attempt_num_" + past_log_file)
        LOG_DIR = path + "/" + type_of_attack+attacked_file+method_of_attack + "Attempt_num_" + past_log_file


    else:
        attempt_folder = os.mkdir(path + "/" +type_of_attack+ attacked_file+method_of_attack + "Attempt_num_" + amount_of_logs)
        LOG_DIR = path + "/" + type_of_attack+attacked_file +method_of_attack+ "Attempt_num_" + amount_of_logs


    path_log_pareto = LOG_DIR

    return LOG_DIR
    
parser.add_argument("-method", "--method", required=False)
parser.add_argument("-m", "--model", required=False)
parser.add_argument("-allfigs", "--allfigs", required=False)
parser.add_argument("-barplot", "--barplot", required=False)

parser.add_argument("-Mu", "--Mu", required=False)
parser.add_argument("-Lambda", "--Lambda", required=False)
parser.add_argument("-Ngen", "--NGEN", required=False)
parser.add_argument("-Min_Stra", "--MIN_STRATEGY", required=False)
parser.add_argument("-Max_Stra", "--MAX_STRATEGY", required=False)
args = parser.parse_args()
global NGEN
MIN_STRATEGY, MAX_STRATEGY = 1, 100  # standard deviation of the mutation. orig: 50, 100
MU, LAMBDA = 10, 100# mu number of individuals to select, lambda number of children to produce. orig: 20,200. good too: 10,200

NGEN = 15
try:
    MU = int(args.Mu)
except Exception:
    pass

try:
    LAMBDA = int(args.Lambda)
except Exception:
    pass

try:
    NGEN = int(args.NGEN)
except Exception:
    pass

try:
    MIN_STRATEGY = int(args.MIN_STRATEGY)
except Exception:
    pass
try:
    MAX_STRATEGY = int(args.MAX_STRATEGY)
except Exception:
    pass




try:

    main(args.method,args.model,args.style,int(args.filename))
except Exception:
    
    print("Failed")


if args.barplot == "True" or args.barplot == "true" or args.barplot == "t" or args.barplot == "T":
    try:
        create_bar_plot("logs")
    except Exception:
    
      print("Failed")

if args.allfigs == "True" or args.allfigs == "true" or args.allfigs == "t" or args.allfigs == "T":
    try:
        global_enviroment_setter_for_diagarms()
    except Exception:
    
      print("Failed")


