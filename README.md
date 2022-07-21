# RL-for-humanoids

## Requirements (Incomplete):
- Python version: 3.7.11  
- Pytorch (CPU only): `$ conda install pytorch torchvision torchaudio cpuonly -c pytorch`
- pip install:
  - mujoco==2.1.5
  - mujoco-python-viewer
  - ray
  - transforms3d

## Usage:

### **To train using custom PPO implementation:** 

```sh
$ python run_experiment.py train --logdir <path_to_exp_dir> --num_procs <num_of_cpu_procs> --env <name_of_environment>
```  


### **To play a trained policy:** 

```sh
$ python run_experiment.py eval --path <path_to_exp_dir>
```
