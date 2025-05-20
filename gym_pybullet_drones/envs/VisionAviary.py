import os
import cv2
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType

class VisionAviary(BaseRLAviary):
    def __init__(self,
                 drone_model=DroneModel.CF2X,
                 num_drones=1,
                 neighbourhood_radius=10,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics=Physics.PYB,
                 freq=240,
                 aggregate_phy_steps=1,
                 gui=False,
                 record=False,
                 obstacles=True,
                 user_debug_gui=False,
                 output_folder='results',
                 obs=ObservationType.RGB,
                 act=ActionType.ONE_D_RPM
                 ):
        self.IMG_RES = np.array([256, 256])  # Αντί για tuple
        self.OBS_CAM_DIST = 3
        self.OBS_CAM_YAW = 45
        self.OBS_CAM_PITCH = -30
        self.OBS_CAM_FOV = 60
        self.MAX_RPM = 8000
        self.goal = np.array([2.0, 4.0, 1.0])
        self.spawn_range = [[0.5, 3.5], [0.5, 3.5], [0.3, 2.5]]
        self.KF = 3.16e-10      # force coefficient
        self.KM = 7.94e-13      # torque coefficient


        os.environ["PYBULLET_EGL"] = "1"
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)

    def _computeObs(self):
        rgb, _, _ = self._getDroneImages(0, segmentation=False)
    
    # Final safety check
        if rgb is None or rgb.shape != (256, 256, 3):
            print("[WARNING] RGB image has invalid shape, returning dummy image.")
            rgb = np.zeros((256, 256, 3), dtype=np.uint8)

        rgb = rgb.transpose(2, 0, 1)  # HWC -> CHW
        return rgb


    def _computeReward(self):
        drone_pos = self._getDroneStateVector(0)[0:3]
        distance = np.linalg.norm(drone_pos - self.goal)

    # Θετικό reward όσο πλησιάζει
        reward = np.exp(-distance)

    # Penalty αν πέσει πολύ χαμηλά
        if drone_pos[2] < 0.2:
            reward -= 1.0

    # Penalty για σύγκρουση
        if self._isCollision(0):
            reward -= 5.0

    # Bonus αν φτάσει πολύ κοντά στον στόχο
        if distance < 0.2:
            reward += 2.0

        return reward

    def _computeDone(self):
        pos = self._getDroneStateVector(0)[:3]
        return pos[2] < 0.05 or self._isCollision(0)

    def _computeInfo(self):
        return {}

    def _observationSpace(self):
        return spaces.Box(low=0, high=255, shape=(3, self.IMG_RES[1], self.IMG_RES[0]), dtype=np.uint8)

    def _actionSpace(self):
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def _preprocessAction(self, action):
    # Αν είναι scalar ή έχει shape (1,), κάνε broadcast
        if np.isscalar(action) or action.shape == (1,):
            rpm = np.array([[action] * 4]) * self.MAX_RPM
        else:
            try:
                action = np.reshape(action, (self.NUM_DRONES, 4))
                rpm = action * self.MAX_RPM
            except Exception as e:
                #print(f"[ERROR in _preprocessAction] Action shape: {action.shape}, error: {e}")
                raise e
        #print(f"[DEBUG] RPMs sent to drone(s): {rpm}")
        return rpm
 
    def reset(self, seed=None, options=None):
        spawn = np.random.uniform([r[0] for r in self.spawn_range], [r[1] for r in self.spawn_range])
        self.INIT_XYZS = np.array([spawn])
        return super().reset(seed=seed, options=options)

    def _addObstacles(self):
        obstacle_positions = [
            [2, 2, 0.25],
            [3, 3, 0.25],
            [1, 1, 0.25],
            [3, 1, 0.25],
            [1, 3, 0.25],
#            [2, 4, 0.25],
#            [3, 2, 0.25],
#            [4, 3, 0.25],
#            [1, 4, 0.25],
#            [4, 1, 0.25],
#            [2, 5, 0.25],
#            [5, 2, 0.25],
#            [1, 5, 0.25],
#            [5, 1, 0.25],
#            [2, 6, 0.25],
#           [6, 2, 0.25],
#            [1, 6, 0.25],
#           [6, 1, 0.25],
#            [2, 7, 0.25],
#            [7, 2, 0.25],
#            [1, 7, 0.25],
#            [7, 1, 0.25],    
 ]
        for pos in obstacle_positions:
            p.loadURDF("cube.urdf", pos,
                    p.getQuaternionFromEuler([0, 0, 0]),
                    useFixedBase=True,
                    globalScaling=1.0,
                    physicsClientId=self.CLIENT)

    # Optional: Add visual target (goal)
        p.loadURDF("sphere2.urdf", self.goal,
                p.getQuaternionFromEuler([0, 0, 0]),
                globalScaling=0.2,
                useFixedBase=True,
                physicsClientId=self.CLIENT)

    #def _addObstacles(self):
       # p.setAdditionalSearchPath(pybullet_data.getDataPath())
       # positions = [[2, 2, 0.6], [3, 3, 0.6], [1, 1, 0.6], [3, 1, 0.6], [1, 3, 0.6]]
        #for pos in positions:
        #    p.loadURDF("cube.urdf", pos, p.getQuaternionFromEuler([0, 0, 0]), globalScaling=1, physicsClientId=self.CLIENT)
        #p.loadURDF("sphere2.urdf", self.goal, p.getQuaternionFromEuler([0, 0, 0]), globalScaling=0.3, physicsClientId=self.CLIENT)

    def _isCollision(self, drone_id):
        return len(p.getContactPoints(bodyA=self.DRONE_IDS[drone_id], physicsClientId=self.CLIENT)) > 0
    
    def _computeTerminated(self):
        pos = self._getDroneStateVector(0)[0:3]
        distance = np.linalg.norm(pos - self.goal)

        if distance < 0.3:
            print("[INFO] Drone reached the goal!")
            return True

        if pos[2] < 0.05 or self._isCollision(0):
            print("[INFO] Drone crashed or fell.")
            return True

        return False

    def _computeTruncated(self):
        return self.step_counter > 300
    
    def _getDroneImages(self, drone_id, segmentation=False):
        try:
            img_arr, _, _ = self._getDroneRGB(drone_id, segmentation=segmentation)
        except Exception as e:
            print(f"[ERROR] Exception while getting drone RGB: {e}")
            img_arr = None

    # Check validity
        if img_arr is None or len(img_arr.shape) != 3 or img_arr.shape[0] == 0 or img_arr.shape[1] == 0:
            print(f"[ERROR] Invalid image received from drone {drone_id}!")
            img_arr = np.zeros((256, 256, 3), dtype=np.uint8)  # fallback image
        else:
        # Remove alpha channel if it exists
            if img_arr.shape[2] == 4:
                img_arr = img_arr[:, :, :3]

        # Resize safely
            try:
                img_arr = cv2.resize(img_arr, (256, 256))
            except Exception as e:
                print(f"[ERROR] Resize failed: {e}")
                img_arr = np.zeros((256, 256, 3), dtype=np.uint8)

        return img_arr, None, None
    
    def _getDroneRGB(self, drone_id=0, segmentation=False):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.pos[drone_id],
            distance=self.OBS_CAM_DIST,
            yaw=self.OBS_CAM_YAW,
            pitch=self.OBS_CAM_PITCH,
            roll=0,
            upAxisIndex=2
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.OBS_CAM_FOV,
            aspect=float(self.IMG_RES[0]) / self.IMG_RES[1],
            nearVal=0.1,
            farVal=100.0
        )
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.IMG_RES[0],
            height=self.IMG_RES[1],
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_array = np.reshape(rgb_img, (height, width, 4))
        return rgb_array, depth_img, seg_img
    def getDroneImage(self):
        img, _, _ = self._getDroneImages(0)
        return img
