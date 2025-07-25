import numpy as np
from gym import spaces
import pybullet as p
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel


class VisionAviary(BaseRLAviary):
    def __init__(self, gui=False, record=False, obstacles=True):
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            neighbourhood_radius=10,
            initial_xyzs=np.array([[1, 1, 1]]),
            physics="pyb",
            gui=gui,
            record=record
        )

        self._obstacles_enabled = obstacles
        if self._obstacles_enabled:
            self._addObstacles()

        self.IMG_RES = (256, 256)
        self.RENDERER = p.ER_BULLET_HARDWARE_OPENGL
        self.current_action = None
        self.action_timer = 0
        self.goal = np.array([2.0, 4.0, 1.0])

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(3, self.IMG_RES[0], self.IMG_RES[1]), dtype=np.uint8),
            "goal": spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32)
        })

    def _getDroneImages(self, drone_id=0, segmentation=False):
        pos = self._getDroneStateVector(drone_id)[:3]
        drone_orientation = self._getDroneStateVector(drone_id)[3:6]

        # Set the camera slightly behind and above the drone, facing forward
        offset = np.array([0, 0, 0.2])
        camera_pos = pos + offset
        target_pos = pos + self._getDroneForwardVector(drone_id)

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=[0, 0, 1]
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
        rgbImg = np.reshape(rgbImg, (height, width, 4))[:, :, :3]
        rgbImg = np.transpose(rgbImg, (2, 0, 1))
        return rgbImg, depthImg, segImg

    def _getDroneForwardVector(self, drone_id=0):
        orientation = self._getDroneStateVector(drone_id)[3:7]  # quaternion
        rot_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        forward_vector = rot_matrix @ np.array([1, 0, 0])
        return forward_vector

    def _computeObs(self):
        rgb, _, _ = self._getDroneImages(0)
        drone_pos = self._getDroneStateVector(0)[0:3]
        rel_goal = self.goal - drone_pos
        return {"image": rgb, "goal": rel_goal.astype(np.float32)}

    def _computeReward(self):
        drone_pos = self._getDroneStateVector(0)[0:3]
        distance = np.linalg.norm(self.goal - drone_pos)
        reward = -1.0  # step penalty
        reward -= 2.0 * distance
        if distance < 0.3:
            reward += 100.0
        if self._isCollision(0):
            reward -= 100.0
        return reward

    def _computeTerminated(self):
        drone_pos = self._getDroneStateVector(0)[0:3]
        distance = np.linalg.norm(self.goal - drone_pos)
        if distance < 0.3:
            return True
        if self._isCollision(0):
            return True
        return False

    def _computeTruncated(self):
        return self.step_counter >= 300

    def _computeInfo(self):
        drone_pos = self._getDroneStateVector(0)[0:3]
        return {"distance_to_goal": float(np.linalg.norm(self.goal - drone_pos))}

    def _preprocessAction(self, action):
        if self.action_timer > 0:
            self.action_timer -= 1
            act = self.current_action
        else:
            act = action
            self.current_action = action
            self.action_timer = int(act[7] * 40) + 10  # scaled to 10–50 steps

        dir_logits = act[:6]
        direction = np.argmax(dir_logits)
        speed = np.clip(act[6], 0.0, 1.0)

        direction_vectors = {
            0: np.array([1, 0, 0]),
            1: np.array([-1, 0, 0]),
            2: np.array([0, 1, 0]),
            3: np.array([0, -1, 0]),
            4: np.array([0, 0, 1]),
            5: np.array([0, 0, -1])
        }
        thrust_vector = direction_vectors[direction] * speed
        rpm_base = 0.6 * self.MAX_RPM
        rpm = np.array([[rpm_base + thrust_vector[0] * self.MAX_RPM * 0.4,
                         rpm_base + thrust_vector[1] * self.MAX_RPM * 0.4,
                         rpm_base + thrust_vector[2] * self.MAX_RPM * 0.4,
                         rpm_base + thrust_vector[2] * self.MAX_RPM * 0.4]])
        return rpm

    def _isCollision(self, drone_id=0):
        contacts = p.getContactPoints(bodyA=self.DRONE_IDS[drone_id], physicsClientId=self.CLIENT)
        return len(contacts) > 0

    def _addObstacles(self):
        obstacle_positions = [
            [2, 2, 0.25],
            [3, 3, 0.25],
            [1, 1, 0.25],
            [3, 1, 0.25],
            [1, 3, 0.25]
        ]
        for pos in obstacle_positions:
            p.loadURDF(
                "cube.urdf",
                pos,
                p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True,
                globalScaling=1.0,
                physicsClientId=self.CLIENT
            )
