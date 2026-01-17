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
├── envs/                      <-- Environment implementations
│   ├── common/
│   │   ├── base_humanoid_env.py   <-- Base class for all humanoid environments
│   │   ├── mujoco_env.py          <-- MuJoCo simulation wrapper
│   │   └── robot_interface.py     <-- Robot state/control abstraction
│   ├── jvrc/                      <-- JVRC robot environments
│   └── h1/                        <-- Unitree H1 robot environment
├── tasks/                     <-- Task definitions (rewards, termination)
├── rl/                        <-- Reinforcement learning
├── robots/                    <-- Robot abstractions (PD control, stepping logic)
├── models/                    <-- MuJoCo model files
└── tests/                     <-- Test suite
```

### Key abstractions:
- **BaseHumanoidEnv**: Common functionality for humanoid environments (observation history, action smoothing, reset logic)
- **BaseTask**: Interface for task implementations (reset, step, calc_reward, done)
- **Reward functions**: Explicit parameter functions in `tasks/rewards.py` for testability

## Requirements:
- Python version: >= 3.10
- [uv](https://docs.astral.sh/uv/) package manager

```bash
$ uv sync
```

## Usage:

Environment names supported:

| Task Description      | Environment name |
| ----------- | ----------- |
| Basic Standing Task   | 'h1' |
| Basic Walking Task   | 'jvrc_walk' |
| Stepping Task (using footsteps)  | 'jvrc_step' |


#### **To train:**

```
$ uv run run_experiment.py train --logdir <path_to_exp_dir> --num_procs <num_of_cpu_procs> --env <name_of_environment>
```

Note: Setting `RAY_ADDRESS=` ensures Ray starts a new local cluster instead of connecting to an existing one.

#### **To play:**

```
$ uv run run_experiment.py eval --logdir <path_to_actor_pt>
```

Or, we could write a rollout script specific to each environment.

## Configuration

Environment behavior is configured via YAML files in `envs/<robot>/configs/`. Key parameters:

```yaml
# Simulation
sim_dt: 0.001              # Physics timestep
control_dt: 0.025          # Control loop period
obs_history_len: 1         # Observation history length
action_smoothing: 0.5      # Action filtering coefficient

# Task parameters
task:
  goal_height: 0.80        # Target standing height
  swing_duration: 0.75     # Gait swing phase duration
  stance_duration: 0.35    # Gait stance phase duration

# Reward weights (sum to 1.0)
reward_weights:
  foot_frc_score: 0.225
  foot_vel_score: 0.225
  # ... see configs for full list
```

## Adding a new robot

1. Create `envs/<robot>/` directory with:
   - `gen_xml.py` - MJCF generation from URDF
   - `configs/base.yaml` - Configuration
   - `<robot>_env.py` - Environment class

2. Inherit from `BaseHumanoidEnv` and implement:
   ```python
   class MyRobotEnv(BaseHumanoidEnv):
       def _get_default_config_path(self) -> str: ...
       def _build_xml(self) -> str: ...
       def _setup_robot(self) -> None: ...
       def _setup_spaces(self) -> None: ...
       def _get_robot_state(self) -> np.ndarray: ...
       def _get_external_state(self) -> np.ndarray: ...
   ```

3. Register in `envs/__init__.py`:
   ```python
   ENVIRONMENTS = {
       "my_robot": (MyRobotEnv, "my_robot"),
       # ...
   }
   ```

4. Run tests: `uv run pytest tests/ -v`

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
