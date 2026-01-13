"""Environment registry for learninghumanoidwalking.

All environment classes should be imported and registered here.
This makes environments discoverable for testing and external use.
"""

from envs.jvrc import JvrcWalkEnv, JvrcStepEnv
from envs.h1 import H1Env

# Registry of all available environments
# Maps environment name -> (class, robot_name)
ENVIRONMENTS = {
    "jvrc_walk": (JvrcWalkEnv, "jvrc"),
    "jvrc_step": (JvrcStepEnv, "jvrc"),
    "h1": (H1Env, "h1"),
}

__all__ = [
    "JvrcWalkEnv",
    "JvrcStepEnv",
    "H1Env",
    "ENVIRONMENTS",
]
