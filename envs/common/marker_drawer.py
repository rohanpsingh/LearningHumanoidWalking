"""Utility for drawing markers in the MuJoCo viewer."""

import numpy as np
import mujoco


class MarkerDrawer:
    """Helper class to draw markers using mujoco.viewer's user_scn.

    Usage:
        marker_drawer = MarkerDrawer(viewer)

        # In your render loop:
        marker_drawer.reset()
        marker_drawer.add_marker(pos=[0, 0, 1], size=[0.1, 0, 0],
                                  rgba=[1, 0, 0, 1], type=mujoco.mjtGeom.mjGEOM_SPHERE)
        marker_drawer.finalize()
        viewer.sync()
    """

    def __init__(self, viewer):
        """Initialize the marker drawer.

        Args:
            viewer: A mujoco.viewer handle from launch_passive()
        """
        self.viewer = viewer
        self.geom_idx = 0

    def reset(self):
        """Reset the geometry counter before drawing a new frame."""
        self.geom_idx = 0

    def finalize(self):
        """Set the final geometry count after all markers are added."""
        self.viewer.user_scn.ngeom = self.geom_idx

    def add_marker(self, pos, size, rgba, type, mat=None, label=""):
        """Add a marker geometry to the viewer scene.

        Args:
            pos: 3D position [x, y, z]
            size: Size array (interpretation depends on geometry type)
            rgba: Color [r, g, b, a]
            type: mujoco.mjtGeom type (e.g., mjGEOM_SPHERE, mjGEOM_ARROW)
            mat: 3x3 rotation matrix (default: identity)
            label: Unused (kept for API compatibility)
        """
        if self.geom_idx >= self.viewer.user_scn.maxgeom:
            return  # Can't add more geometries

        if mat is None:
            mat = np.eye(3)

        # Convert size format for different geometry types
        if type == mujoco.mjtGeom.mjGEOM_SPHERE:
            # For sphere, size[0] is radius
            geom_size = [size[0], 0, 0]
        elif type == mujoco.mjtGeom.mjGEOM_ARROW:
            # For arrow: [shaft_radius, head_radius, shaft_length]
            geom_size = [size[0], size[1], size[2]]
        else:
            geom_size = list(size)[:3] if len(size) >= 3 else list(size) + [0] * (3 - len(size))

        mujoco.mjv_initGeom(
            self.viewer.user_scn.geoms[self.geom_idx],
            type=type,
            size=geom_size,
            pos=np.array(pos, dtype=np.float64),
            mat=np.array(mat, dtype=np.float64).flatten(),
            rgba=np.array(rgba, dtype=np.float32)
        )
        self.geom_idx += 1
