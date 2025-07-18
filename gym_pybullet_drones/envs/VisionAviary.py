import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel

class VisionAviary(BaseAviary):

    def __init__(self, gui=False, obstacles=True):
        self.IMG_RES = (256, 256)
        self.RENDERER = p.ER_BULLET_HARDWARE_OPENGL if gui else p.ER_TINY_RENDERER
        self.goal = np.array([2.0, 4.0, 1.0], dtype=np.float32)
        self.prev_distance = None

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            neighbourhood_radius=10,
            initial_xyzs=np.array([[1, 1, 1]]),
            physics='pyb',
            gui=gui,
            record=False,
            obstacles=obstacles,
            user_debug_gui=False
        )

    def _actionSpace(self):
        return spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

    def _observationSpace(self):
        return spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(3, self.IMG_RES[0], self.IMG_RES[1]), dtype=np.uint8),
            "features": spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        })

    def _preprocessAction(self, action):
        action = np.clip(action, 0.0, 1.0)
        rpm = action * self.MAX_RPM
        return rpm

    def _getDroneImages(self, drone_id=0, segmentation=False):
        pos = self._getDroneStateVector(drone_id)[:3]
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=pos,
            distance=3.0,
            yaw=0,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.IMG_RES[0]) / self.IMG_RES[1],
            nearVal=0.1,
            farVal=10.0
        )
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=self.IMG_RES[0],
            height=self.IMG_RES[1],
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=self.RENDERER,
            flags=p.ER_NO_SEGMENTATION_MASK
            )

    # από flat → σωστό RGB image (χωρίς alpha)
        rgbImg = np.reshape(rgbImg, (height, width, 4))[:, :, :3]

        return rgbImg, depthImg, segImg

    def _reset(self):
        self.prev_distance = None
        self.step_counter = 0
        self.goal = np.array([2.0, 4.0, 1.0], dtype=np.float32)
        obs = self._computeObs()
        return obs, {}
    def _computeObs(self):
        # --- RGB IMAGE ---
        rgb, _, _ = self._getDroneImages(0, segmentation=False)
        if rgb is None or rgb.shape != (256, 256, 3):
            rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb = rgb.transpose(2, 0, 1)

        # --- Features Vector ---
        drone_pos = self._getDroneStateVector(0)[0:3]
        vector_to_goal = self.goal - drone_pos
        distance = np.linalg.norm(vector_to_goal)
        unit_to_goal = vector_to_goal / (distance + 1e-6)

        heading_vector = self.getDroneAxisVector('forward', 0)
        heading_vector = heading_vector / (np.linalg.norm(heading_vector) + 1e-6)
        alignment = np.dot(unit_to_goal, heading_vector)

        features = np.concatenate([
            drone_pos,
            self.goal,
            vector_to_goal,
            [distance],
            heading_vector,
            [alignment]
        ])

        return {"image": rgb, "features": features}

    def getDroneAxisVector(self, axis='forward', drone_id=0):
        orn = p.getBasePositionAndOrientation(self.DRONE_IDS[drone_id], physicsClientId=self.CLIENT)[1]
        rot_matrix = p.getMatrixFromQuaternion(orn)
        if axis == 'forward':
            return np.array([rot_matrix[0], rot_matrix[3], rot_matrix[6]])
        elif axis == 'right':
            return np.array([rot_matrix[1], rot_matrix[4], rot_matrix[7]])
        elif axis == 'up':
            return np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])
        else:
            raise ValueError(f"Invalid axis '{axis}'. Choose 'forward', 'right', or 'up'.")

    def _computeReward(self):
        drone_pos = self._getDroneStateVector(0)[0:3]
        vector_to_goal = self.goal - drone_pos
        distance = np.linalg.norm(vector_to_goal)

        unit_to_goal = vector_to_goal / (distance + 1e-6)
        heading_vector = self.getDroneAxisVector('forward', 0)
        heading_vector = heading_vector / (np.linalg.norm(heading_vector) + 1e-6)
        alignment = np.dot(unit_to_goal, heading_vector)

        reward = 0.0
        reward += alignment * 30.0
        reward -= distance * 2.0

        previous_distance = getattr(self, "prev_distance", None)
        if previous_distance is not None:
            delta = previous_distance - distance
            reward += 10.0 * np.clip(delta, -0.5, 0.5)

        if distance < 0.3:
            reward += 100.0

        reward -= 2.0 * max(0.0, drone_pos[2] - 1.5)

        if drone_pos[2] < 0.2:
            reward -= 5.0

        if self._isCollision(0):
            reward -= 10.0

        self.prev_distance = distance
        return reward

    def _computeTerminated(self):
        pos = self._getDroneStateVector(0)[:3]
        distance = np.linalg.norm(pos - self.goal)
        if distance < 0.3:
            return True
        if pos[2] < 0.05:
            return True
        if self._isCollision(0):
            return True
        return False

    def _computeTruncated(self):
        return self.step_counter > 150
    def _computeInfo(self):
        drone_pos = self._getDroneStateVector(0)[0:3]
        dist_to_goal = np.linalg.norm(self.goal - drone_pos)
        return {
        "distance_to_goal": dist_to_goal
    }
    def _isCollision(self, drone_id=0):
        for contact in p.getContactPoints(bodyA=self.DRONE_IDS[drone_id], physicsClientId=self.CLIENT):
            other_body = contact[2]
            if other_body != self.GROUND_PLANE_ID:
                return True
        return False

    def _getDroneStateVector(self, drone_id=0):
        return np.array(p.getBasePositionAndOrientation(self.DRONE_IDS[drone_id], physicsClientId=self.CLIENT)[0])      
        


