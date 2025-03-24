# LearningHumanoidWalking

<p align="center">
  <a href="https://www.youtube.com/watch?v=ZgfNzGAkk2Q"><img src="https://github.com/user-attachments/assets/5211cdcd-2267-497b-bd66-ac833703a134" alt="humanoid-walk" style="width:1000px"/></a>
</p>

Code for the papers:  
- [**Robust Humanoid Walking on Compliant and Uneven Terrain with Deep Reinforcement Learning**](https://ieeexplore.ieee.org/abstract/document/10769793)  
[Rohan P. Singh](https://rohanpsingh.github.io), [Mitsuharu Morisawa](https://unit.aist.go.jp/jrl-22022/en/members/member-morisawa.html), [Mehdi Benallegue](https://unit.aist.go.jp/jrl-22022/en/members/member-benalleguem.html), [Zhaoming Xie](https://zhaomingxie.github.io/), [Fumio Kanehiro](https://unit.aist.go.jp/jrl-22022/en/members/member-kanehiro.html)

- [**Learning Bipedal Walking for Humanoids with Current Feedback**](https://arxiv.org/pdf/2303.03724.pdf)  
[Rohan P. Singh](https://rohanpsingh.github.io), [Zhaoming Xie](https://zhaomingxie.github.io/), [Pierre Gergondet](https://unit.aist.go.jp/jrl-22022/en/members/member-gergondet.html), [Fumio Kanehiro](https://unit.aist.go.jp/jrl-22022/en/members/member-kanehiro.html)

- [**Learning Bipedal Walking On Planned Footsteps For Humanoid Robots**](https://arxiv.org/pdf/2207.12644.pdf)  
[Rohan P. Singh](https://rohanpsingh.github.io), [Mehdi Benallegue](https://unit.aist.go.jp/jrl-22022/en/members/member-benalleguem.html), [Mitsuharu Morisawa](https://unit.aist.go.jp/jrl-22022/en/members/member-morisawa.html), [Rafael Cisneros](https://unit.aist.go.jp/jrl-22022/en/members/member-cisneros.html), [Fumio Kanehiro](https://unit.aist.go.jp/jrl-22022/en/members/member-kanehiro.html)


## Code structure:
A rough outline for the repository that might be useful for adding your own robot:
```
LearningHumanoidWalking/
├── envs/                <-- Actions and observation space, PD gains, simulation step, control decimation, init, ...
├── tasks/               <-- Reward function, termination conditions, and more...
├── rl/                  <-- Code for PPO, actor/critic networks, observation normalization process...
├── models/              <-- MuJoCo model files: XMLs/meshes/textures
└── scripts/             <-- Utility scripts, etc.
```

## Requirements:
- Python version: 3.12.4
- pip install:
  - mujoco==3.2.2
  - ray==2.40.0
  - pytorch=2.5.1
  - intel-openmp
  - [mujoco-python-viewer](https://github.com/rohanpsingh/mujoco-python-viewer)
  - transforms3d
  - scipy

## Usage:

Environment names supported:  

| Task Description      | Environment name |
| ----------- | ----------- |
| Basic Standing Task   | 'h1' |
| Basic Walking Task   | 'jvrc_walk' |
| Stepping Task (using footsteps)  | 'jvrc_step' |


#### **To train:** 

```
$ python run_experiment.py train --logdir <path_to_exp_dir> --num_procs <num_of_cpu_procs> --env <name_of_environment>
```  


#### **To play:**

```
$ python run_experiment.py eval --logdir <path_to_actor_pt>
```

Or, we could write a rollout script specific to each environment.    
For example, `debug_stepper.py` can be used with the `jvrc_step` environment.  
```
$ PYTHONPATH=.:$PYTHONPATH python scripts/debug_stepper.py --path <path_to_exp_dir>
```

#### **What you should see:**

*Ascending stairs:*  
![climb_up](https://user-images.githubusercontent.com/16384313/180697513-25796b1a-87e0-4ab2-9e5f-d86c58ebea36.gif)

*Descending stairs:*  
![climb_down](https://user-images.githubusercontent.com/16384313/180697788-d1a2eec0-0d3d-451a-95e0-9f0e60191c34.gif)

*Walking on curves:*  
![curve](https://user-images.githubusercontent.com/16384313/180697266-7b44beb3-38bf-4494-b568-963919dc1106.gif)


## Citation
If you find this work useful in your own research, please cite the following works:

For omnidirectional walking:
```
@inproceedings{singh2024robust,
  title={Robust Humanoid Walking on Compliant and Uneven Terrain with Deep Reinforcement Learning},
  author={Singh, Rohan P and Morisawa, Mitsuharu and Benallegue, Mehdi and Xie, Zhaoming and Kanehiro, Fumio},
  booktitle={2024 IEEE-RAS 23rd International Conference on Humanoid Robots (Humanoids)},
  pages={497--504},
  year={2024},
  organization={IEEE}
}
```

For simulating "back-emf" effect and other randomizations:
```
@article{xie2023learning,
  title={Learning bipedal walking for humanoids with current feedback},
  author={Singh, Rohan Pratap and Xie, Zhaoming and Gergondet, Pierre and Kanehiro, Fumio},
  journal={IEEE Access},
  volume={11},
  pages={82013--82023},
  year={2023},
  publisher={IEEE}
}
```

For walking on footsteps:  

```
@inproceedings{singh2022learning,
  title={Learning Bipedal Walking On Planned Footsteps For Humanoid Robots},
  author={Singh, Rohan P and Benallegue, Mehdi and Morisawa, Mitsuharu and Cisneros, Rafael and Kanehiro, Fumio},
  booktitle={2022 IEEE-RAS 21st International Conference on Humanoid Robots (Humanoids)},
  pages={686--693},
  year={2022},
  organization={IEEE}
}
```

### Credits
The code in this repository was heavily inspired from [apex](https://github.com/osudrl/apex). Clock-based reward terms and some other ideas were originally proposed by the team from OSU DRL for the Cassie robot, so please also consider citing the works of Jonah Siekmann, Helei Duan, Jeremy Dao, and others.

