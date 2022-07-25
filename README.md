# LearningHumanoidWalking

## Requirements:
- Python version: 3.7.11  
- [Pytorch](https://pytorch.org/)
- pip install:
  - mujoco==2.1.5
  - mujoco-python-viewer
  - ray
  - transforms3d

## Usage:

Environment names supported:  

| Task Description      | Environment name |
| ----------- | ----------- |
| Basic Walking Task   | 'jvrc_walk' |
| Stepping Task (using footsteps)  | 'jvrc_step' |


### **To train:** 

```
$ python run_experiment.py train --logdir <path_to_exp_dir> --num_procs <num_of_cpu_procs> --env <name_of_environment>
```  


### **To play:** 

```
$ python run_experiment.py eval --path <path_to_exp_dir>
```

---

### **What you should see:**

*Ascending stairs:*  
![climb_up](https://user-images.githubusercontent.com/16384313/180697513-25796b1a-87e0-4ab2-9e5f-d86c58ebea36.gif)

*Descending stairs:*  
![climb_down](https://user-images.githubusercontent.com/16384313/180697788-d1a2eec0-0d3d-451a-95e0-9f0e60191c34.gif)

*Walking on curves:*  
![curve](https://user-images.githubusercontent.com/16384313/180697266-7b44beb3-38bf-4494-b568-963919dc1106.gif)
