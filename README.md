

# Adversial attack on grasping network

## Package Overview
A package to preform digital and phyiscal attacks using SPEA2 on a grasping network[^1].

## Installation
```$ git clone https://github.com/Naif-W-Alharthi/Phyiscal-and-Digital-Attacks-on-Grasping-Networks.git ```

Change directories into the repository and run the pip installation. <br>
```$ pip install .```


## Arguments

| Variable  | Description |
| ------------- | ------------- |
| -method  | Takes a string that should be: *driver* or *physical* and will determine the type of simluation used. |
| -m  | Takes a string that selectes the model used can take *All* which will simluate the attacks using all models in /models. Can take *Lowest* which will simluate using the model with the least attempts, otherwise it will use the model provided according to model look up table  |
| -allfigs  |  Takes a string that should be *True* or *False*. Creates all the diagarm.  |
| -barplot  | Takes a string that should be *True* or *False*. Creates the bar plot of both lowest quality and radius/Intensity change in the models. |
| -Mu  |  Takes an int. Mu of the simluation.  |
| -Lambda  |  Takes an int. Lambda of the simluation  |
| -Ngen  |   Takes an int. Ngen of the simluation  |
|-Min_Stra  | Takes an int. MIN_STRATEGY of the simluation  |
|-Max_Stra |  Takes an int. MAX_STRATEGY of the simluation  |

Note: if a value is not mentioned from the ones needed to run a simluation they will be replaced with the default value. 

## Model look up table
| String  | Model used for simluaion |
| ------------- | ------------- |
| "gqcnn_suction"  | GQCNN-4.0-SUCTION  |
| fc-gqcnn_suction  | FC-GQCNN-4.0-SUCTION  |
| fc-gqcnn_parellel_grasp  | FC-GQCNN-4.0-PJ  |
| dex2.1  | Dexnet2.1  |
| dex2.0  | Dexnet2.0  |



tested with python 3.9 and above.


[^1]: [[Reference to grasping networks](https://berkeleyautomation.github.io/gqcnn/index.html#academic-use)].





