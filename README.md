# RL-for-humanoids


### **To train using custom PPO implementation:** 

```sh
$ python run_experiment.py train --logdir <path_to_exp_dir> --num_procs <num_of_cpu_procs> --env <name_of_environment>
```  


### **To play a trained policy:** 

```sh
$ python run_experiment.py eval --path <path_to_exp_dir>
```
