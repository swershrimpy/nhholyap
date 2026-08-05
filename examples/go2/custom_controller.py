from unitree_sdk2py.utils.thread import RecurrentThread
import time
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)
import math
from dataclasses import dataclass
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.video.video_client import VideoClient
import numpy as np
from eval_dino import *
from networks import *
import cv2, os
from omegaconf import DictConfig, OmegaConf
from fault_diagnosis import FaultDiagnosisGo2, OnlineFaultDiagnosis, FaultParameters

base_path = os.path.dirname(__file__)

def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw

def wrap_angle(angle):
    # while angle <= -np.pi:
    #     angle += 2*np.pi
    # while angle > np.pi:
    #     angle -= 2*np.pi
    # return angle
    while angle > 0:
        angle -= 2*np.pi
    while angle <= -2*np.pi:
        angle += 2*np.pi
    return angle

class CustomController:
    def __init__(self, dds=None):
        # self.dt = 0.1
        # self.vx = 0.
        # self.vy = 0.
        # self.vyaw = 0.
        # self.tot_time = 0.
        # self.done = False
        # self.position = None
        # self.yaw = None
        self.tot_time = 0.
        self.device = "cuda"
        self.state_offset = np.array([-0.46, 0.34, 0.])

        # Observation matrix for fault diagnosis
        self.C = np.array([[-0.0329,  0.9805, -0.1938],
                           [-0.8052, -0.5551, -0.2087],
                           [-0.8518, -0.4816,  0.2061]])

        self.Init = True
        self.Init_pos = np.array([0., 0., 0.])
        self.Init_ang = 0.
        self.Mrot = None

        # self.init_vision_module()

        if dds is not None:
            ChannelFactoryInitialize(0, dds)
        else:
            ChannelFactoryInitialize(0)

        self.sport_client = SportClient()  
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()

        # print(self.sport_client.SpeedLevel(-1))

        self.video_client = VideoClient()
        self.video_client.SetTimeout(3.0)
        self.video_client.Init()

        self.state_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.state_subscriber.Init(self.high_state_handler)

        time.sleep(5.)

        # time.sleep(1.)
        # self.sport_client.StandUp()
        # time.sleep(1.)
        # self.sport_client.BalanceStand()
        # time.sleep(2.)

        print("custom controller initialization completed")

    def init_state(self):
        self.Init = False
        
        self.Init_ang = self.state[2]

        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-
                            np.sin(self.Init_ang), np.cos(self.Init_ang)]])
        self.Init_pos[0] = Mrot.item((0,0))*self.state[0] + Mrot.item((0,1))*self.state[1]
        self.Init_pos[1] = Mrot.item((1,0))*self.state[0] + Mrot.item((1,1))*self.state[1]
        self.Init_pos[2] = 0.
        self.Mrot = Mrot

    def high_state_handler(self, msg:SportModeState_):
        position = msg.position
        # # quat = msg.imu_state.quaternion
        # # yaw = quat_to_yaw(quat[0], quat[1], quat[2], quat[3])
        yaw = msg.imu_state.rpy[2]

        # if self.Init:
        #     self.Init = False
        #     self.Init_ang = yaw

        #     Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-
        #                         np.sin(self.Init_ang), np.cos(self.Init_ang)]])
        #     self.Init_pos[0] = Mrot.item((0,0))*position[0] + Mrot.item((0,1))*position[1]
        #     self.Init_pos[1] = Mrot.item((1,0))*position[0] + Mrot.item((1,1))*position[1]
        #     self.Init_pos[2] = position[2]
        #     self.Mrot = Mrot

        # x = self.Mrot.item((0,0))*position[0] + self.Mrot.item((0,1))*position[1] - self.Init_pos[0]
        # y = self.Mrot.item((1,0))*position[0] + self.Mrot.item((1,1))*position[1] - self.Init_pos[1]
        # heading = wrap_angle(yaw - self.Init_ang)

        # self.state = [x, y, heading]
        # self.vel = msg.velocity
        # # print("x: {:.3f}, y: {:.3f}, yaw: {:.3f}".format(x, y, heading))

        if self.Init:
            self.state = [position[0], position[1], yaw]
            self.vel = msg.velocity
        else:
            x = self.Mrot.item((0,0))*position[0] + self.Mrot.item((0,1))*position[1] - self.Init_pos[0]
            y = self.Mrot.item((1,0))*position[0] + self.Mrot.item((1,1))*position[1] - self.Init_pos[1]
            heading = wrap_angle(yaw - self.Init_ang)

            self.state = [x, y, heading]
            self.vel = msg.velocity
            # print("x: {:.3f}, y: {:.3f}, yaw: {:.3f}".format(x, y, heading))

    def start(self):
        self.init_state()
        plan = np.load("Go2_OF_Perception.npz")
        
        # primal_u = plan["initial_input"]
        # primal_x = plan["initial_traj"]
        primal_u = plan["nominal_input"]
        primal_x = plan["nominal_traj"]
        vx = 0.6
        # vy = -0.2
        # vyaw = -0.3
        for i in range(10):
        # for i in range(primal_u.shape[1]):
            print("input ", i)
            # self.sport_client.Move(primal_u[0,i], 0., primal_u[2,i])
            self.sport_client.Move(vx, 0., 0.)
            time.sleep(0.5)
            # time.sleep(0.6)
            # print("expected vx: ", primal_u[0,i])
            # print("actual vel: ", self.vel)
            
            # print(np.array(self.state)+np.array([0.46, -0.34, 0.]))
            # print(primal_x[:, i+1] - np.array([0.46, -0.34, 0.]))
            # print(vx*0.5*(i+1))
            # print(vy*0.5*(i+1))
            # print(vyaw*0.5*(i+1))
            # print(self.state[0]*1.20945952+0.06788091, ", ", self.state[1]*1.20945952+0.06788091, ", ", self.state[2])
            print(self.state)
        print(self.state)

        self.sport_client.StopMove()
    
    def init_vision_module(self):
        # Setting up Dino
        self.preprocess = get_dino_v2_preprocess()
        self.tf = transforms.ToPILImage()
        self.model_dino = load_dino_v2_model("vitg14").to(self.device) 
        self.model_dino.patch_embed.forward = patch_embed_forward.__get__(self.model_dino.patch_embed)
        print("dino loaded")

        params = OmegaConf.load(os.path.join(base_path, "learn_model.yaml"))
        self.model_err = SupervisedDinoObservability(params.model).to(self.device)
        checkpoint = torch.load(os.path.join(base_path, "model_go2_val.pt"), map_location=self.device)
        self.model_err.load_state_dict(checkpoint)
        # self.C = dcn(self.model_err.C / self.model_err.C.norm(dim=1, keepdim=True)).astype('float64')
        # self.C = np.array([[ 0.3794, -0.9230,  0.0633, 0.],
        #                    [-0.8704,  0.1958, -0.4518, 0.],
        #                    [-0.4470,  0.8545, -0.2647, 0.]])
        self.C = np.array([[-0.0329,  0.9805, -0.1938],
                            [-0.8052, -0.5551, -0.2087],
                            [-0.8518, -0.4816,  0.2061]])

    def extract_dino_v2_features_single(self, images_pil) -> torch.Tensor:
        images = torch.stack([self.preprocess(self.tf(image.permute(2, 0, 1))) for image in images_pil], dim=0).to(self.device)
        with torch.no_grad():
            descriptor = self.model_dino.forward_features(images)["x_norm_clstoken"].detach().cpu()

        return descriptor
        
    def dino_fn(self):
        code, data = self.video_client.GetImageSample()

        if code != 0:
            print("get image sample error. code:", code)
        else:
            image_data = np.frombuffer(bytes(data), dtype=np.uint8)
            img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
 
            rgb_im = np.array(img)
            
            x_dim, y_dim, channel = rgb_im.shape
            # Load Dino model and set up transforms
            rgb_im = np.array(rgb_im, dtype=np.uint8).reshape((x_dim, y_dim, 3))
            descriptor = self.extract_dino_v2_features_single(torch.from_numpy(np.array([rgb_im])).to(self.device))
            nn_descriptor = dcn(self.model_err(descriptor.to(self.device)))

            return nn_descriptor, rgb_im

    def start_visuomotor_control(self):
        plan = np.load("Go2_OF_Perception2.npz")
        primal_u = plan["nominal_input"]
        primal_x = plan["nominal_traj"]
        Phi_xx = plan["Phi_xx"]
        Phi_ux = plan["Phi_ux"]
        Phi_xy = plan["Phi_xy"]
        Phi_uy = plan["Phi_uy"]
        tube = plan["backoff"]
        tube_f = plan["backoff_f"]
        
        nx = primal_x.shape[0]
        nu = primal_u.shape[0]
        ny = 3
        K_mat = Phi_uy - Phi_ux @ np.linalg.inv(Phi_xx) @ Phi_xy
        
        N = primal_u.shape[1]
        visited_states = np.zeros((3, N+1))
        observations = np.zeros((3, N+1))

        obs, rgb_im = self.dino_fn()
        observations[:,0] = obs[0]
        img_path = os.path.join("obs_imgs", f"frame_{0:03d}.png")
        cv2.imwrite(img_path, rgb_im)
        
        for t in range(1, N+1):
            time.sleep(1.5)

            delta_u = np.zeros((nu))
            
            # if t == 1:
            #     # observations[:,t-1] = m.measurement(states[:,t-1], disturbance_obs[:,t-1])
            #     obs, rgb_im = self.dino_fn()
            #     observations[:,t-1] = obs[0]
            # else:
            #     # observations[:,t-1] = m.measurement(states[:,t-2], disturbance_obs[:,t-1])
            #     obs, rgb_im = self.dino_fn()
            #     observations[:,t-1] = obs[0]
            obs, rgb_im = self.dino_fn()
            observations[:,t] = obs[0]
            print("nn observation: ", observations[:,t])
            print("primal observation: ", self.C @ (primal_x[:, t-1] + self.state_offset))
            
            delta_u += K_mat[(t-1)*nu:t*nu, 0:ny] @ (observations[:,0] - self.C @ (primal_x[:,0] + self.state_offset))
            for j in range(1, t):
                delta_u += K_mat[(t-1)*nu:t*nu, j*ny:(j+1)*ny] @ (observations[:,j] - self.C @ (primal_x[:,j-1] + self.state_offset ))
            
            u = primal_u[:,t-1] + delta_u

            self.sport_client.Move(u[0], 0., u[2])
            time.sleep(0.5)
            self.sport_client.StopMove()
            
            img_path = os.path.join("obs_imgs", f"frame_{t:03d}.png")
            cv2.imwrite(img_path, rgb_im)
            # print(self.state)
            print(self.state[0]*1.20945952+0.06788091, ", ", self.state[1]*1.20945952+0.06788091, ", ", self.state[2])
        #     visited_states[:,t] = self.state.copy()

        # np.savez("visited_states.npz", traj=visited_states)

    def start_with_fault_diagnosis(self, inject_actuator_fault=False, inject_sensor_fault=False,
                                   actuator_fault_level=0.7, sensor_bias=0.1, sensor_scale=1.05):
        """
        Execute trajectory with optional fault injection and perform fault diagnosis

        Args:
            inject_actuator_fault: Whether to inject actuator fault (reduced turn rate)
            inject_sensor_fault: Whether to inject sensor fault (IMU drift)
            actuator_fault_level: Actuator effectiveness (0-1, where 1=no fault)
            sensor_bias: Sensor bias in radians
            sensor_scale: Sensor scale factor

        Returns:
            Estimated fault-induced errors [px, py, yaw]
        """
        self.init_state()
        plan = np.load(os.path.join(base_path, "Go2_OF_Perception2.npz"))
        primal_u = plan["nominal_input"]
        primal_x = plan["nominal_traj"]
        Phi_xx = plan["Phi_xx"]
        Phi_ux = plan["Phi_ux"]
        Phi_xy = plan["Phi_xy"]
        Phi_uy = plan["Phi_uy"]

        nx = primal_x.shape[0]
        nu = primal_u.shape[0]
        ny = 3
        K_mat = Phi_uy - Phi_ux @ np.linalg.inv(Phi_xx) @ Phi_xy

        N = primal_u.shape[1]
        visited_states = np.zeros((3, N+1))
        observations = np.zeros((3, N+1))
        commanded_inputs = np.zeros((3, N))

        print("\n" + "="*60)
        print("STARTING TRAJECTORY EXECUTION WITH FAULT DIAGNOSIS")
        print("="*60)
        if inject_actuator_fault:
            print(f"⚠ Injecting ACTUATOR FAULT: {(1-actuator_fault_level)*100:.1f}% turn rate reduction")
        if inject_sensor_fault:
            print(f"⚠ Injecting SENSOR FAULT: scale={sensor_scale}, bias={sensor_bias:.3f} rad")
        print("="*60 + "\n")

        # Initialize vision module for observations
        self.init_vision_module()

        # Get initial observation
        obs, rgb_im = self.dino_fn()
        observations[:,0] = obs[0]
        visited_states[:,0] = self.state.copy()

        # Save initial image
        os.makedirs("obs_imgs", exist_ok=True)
        img_path = os.path.join("obs_imgs", f"frame_{0:03d}.png")
        cv2.imwrite(img_path, rgb_im)

        for t in range(1, N+1):
            print(f"\nStep {t}/{N}")

            # Compute control input using output feedback
            delta_u = np.zeros((nu))
            delta_u += K_mat[(t-1)*nu:t*nu, 0:ny] @ (observations[:,0] - self.C @ (primal_x[:,0] + self.state_offset))
            for j in range(1, t):
                delta_u += K_mat[(t-1)*nu:t*nu, j*ny:(j+1)*ny] @ (observations[:,j] - self.C @ (primal_x[:,j-1] + self.state_offset))

            u = primal_u[:,t-1] + delta_u
            commanded_inputs[:, t-1] = u.copy()

            # Apply actuator fault if enabled
            if inject_actuator_fault:
                u_actual = u.copy()
                u_actual[2] *= actuator_fault_level  # Reduce yaw rate
            else:
                u_actual = u

            # Execute control
            print(f"  Commanded: vx={u[0]:.3f}, vy={u[1]:.3f}, omega={u[2]:.3f}")
            if inject_actuator_fault:
                print(f"  Actual:    vx={u_actual[0]:.3f}, vy={u_actual[1]:.3f}, omega={u_actual[2]:.3f}")

            self.sport_client.Move(u_actual[0], u_actual[1], u_actual[2])
            time.sleep(0.5)
            self.sport_client.StopMove()
            time.sleep(1.0)

            # Get observation and state
            obs, rgb_im = self.dino_fn()
            raw_observation = obs[0]

            # Apply sensor fault if enabled
            if inject_sensor_fault:
                # Sensor fault affects the yaw component of state
                faulty_state = self.state.copy()
                faulty_state[2] = sensor_scale * self.state[2] + sensor_bias

                # Recompute observation with faulty state
                observations[:,t] = self.C @ (faulty_state + self.state_offset)
                print(f"  True state: x={self.state[0]:.3f}, y={self.state[1]:.3f}, yaw={self.state[2]:.3f}")
                print(f"  Faulty obs corresponds to: x={faulty_state[0]:.3f}, y={faulty_state[1]:.3f}, yaw={faulty_state[2]:.3f}")
            else:
                observations[:,t] = raw_observation

            visited_states[:,t] = self.state.copy()

            # Save image
            img_path = os.path.join("obs_imgs", f"frame_{t:03d}.png")
            cv2.imwrite(img_path, rgb_im)

            print(f"  State: x={self.state[0]:.3f}, y={self.state[1]:.3f}, yaw={self.state[2]:.3f}")
            print(f"  Observation: {observations[:,t]}")

        print("\n" + "="*60)
        print("TRAJECTORY EXECUTION COMPLETED")
        print("="*60)

        # Perform fault diagnosis
        print("\nPerforming fault diagnosis...")

        # Initialize fault diagnosis
        fd = FaultDiagnosisGo2(
            primal_x,
            primal_u,
            self.C,
            self.state_offset,
            dt=0.5
        )

        # Diagnose faults
        errors, estimated_faults = fd.diagnose(observations, verbose=True)

        # Save results
        results = {
            'visited_states': visited_states,
            'observations': observations,
            'commanded_inputs': commanded_inputs,
            'estimated_errors': errors,
            'estimated_faults': {
                'actuator_effectiveness': estimated_faults.actuator_effectiveness,
                'sensor_scale': estimated_faults.sensor_scale,
                'sensor_bias': estimated_faults.sensor_bias
            }
        }

        if inject_actuator_fault or inject_sensor_fault:
            results['true_faults'] = {
                'actuator_effectiveness': actuator_fault_level if inject_actuator_fault else 1.0,
                'sensor_scale': sensor_scale if inject_sensor_fault else 1.0,
                'sensor_bias': sensor_bias if inject_sensor_fault else 0.0
            }

        np.savez("fault_diagnosis_results.npz", **results)
        print("\nResults saved to fault_diagnosis_results.npz")

        return errors

    def online_fault_monitoring(self):
        """
        Execute trajectory with real-time fault monitoring

        Returns:
            Final estimated fault-induced errors [px, py, yaw]
        """
        self.init_state()
        plan = np.load(os.path.join(base_path, "Go2_OF_Perception2.npz"))
        primal_u = plan["nominal_input"]
        primal_x = plan["nominal_traj"]
        Phi_xx = plan["Phi_xx"]
        Phi_ux = plan["Phi_ux"]
        Phi_xy = plan["Phi_xy"]
        Phi_uy = plan["Phi_uy"]

        nx = primal_x.shape[0]
        nu = primal_u.shape[0]
        ny = 3
        K_mat = Phi_uy - Phi_ux @ np.linalg.inv(Phi_xx) @ Phi_xy

        N = primal_u.shape[1]

        print("\n" + "="*60)
        print("STARTING TRAJECTORY WITH ONLINE FAULT MONITORING")
        print("="*60 + "\n")

        # Initialize vision module
        self.init_vision_module()

        # Initialize online fault diagnosis
        online_fd = OnlineFaultDiagnosis(
            primal_x,
            primal_u,
            self.C,
            self.state_offset,
            window_size=10,
            dt=0.5
        )

        # Get initial observation
        obs, rgb_im = self.dino_fn()
        observation = obs[0]
        online_fd.add_observation(observation, 0)

        os.makedirs("obs_imgs", exist_ok=True)
        img_path = os.path.join("obs_imgs", f"frame_{0:03d}.png")
        cv2.imwrite(img_path, rgb_im)

        latest_errors = None
        latest_faults = None

        for t in range(1, N+1):
            print(f"\n{'='*60}")
            print(f"Step {t}/{N}")
            print('='*60)

            # Compute control input
            delta_u = np.zeros((nu))

            for i in range(min(t, len(online_fd.observation_history))):
                obs_idx = len(online_fd.observation_history) - 1 - i
                if obs_idx >= 0:
                    obs_t = online_fd.observation_history[obs_idx]
                    time_idx = online_fd.time_history[obs_idx]

                    if time_idx == 0:
                        delta_u += K_mat[(t-1)*nu:t*nu, 0:ny] @ (obs_t - self.C @ (primal_x[:,0] + self.state_offset))
                    elif time_idx < t:
                        j = time_idx
                        delta_u += K_mat[(t-1)*nu:t*nu, j*ny:(j+1)*ny] @ (obs_t - self.C @ (primal_x[:,j-1] + self.state_offset))

            u = primal_u[:,t-1] + delta_u

            print(f"Control: vx={u[0]:.3f}, vy={u[1]:.3f}, omega={u[2]:.3f}")

            # Execute
            self.sport_client.Move(u[0], u[1], u[2])
            time.sleep(0.5)
            self.sport_client.StopMove()
            time.sleep(1.0)

            # Get observation
            obs, rgb_im = self.dino_fn()
            observation = obs[0]

            # Update online fault diagnosis
            errors, faults = online_fd.add_observation(observation, t)

            if errors is not None:
                latest_errors = errors
                latest_faults = faults

                print(f"\n🔍 REAL-TIME FAULT ESTIMATE:")
                print(f"  Position error: ({errors[0]:+.4f}, {errors[1]:+.4f}) m")
                print(f"  Heading error: {errors[2]:+.4f} rad ({np.degrees(errors[2]):+.2f}°)")
                print(f"  Actuator effectiveness: {faults.actuator_effectiveness:.3f}")
                print(f"  Sensor scale: {faults.sensor_scale:.3f}")
                print(f"  Sensor bias: {faults.sensor_bias:+.4f} rad ({np.degrees(faults.sensor_bias):+.2f}°)")

                # Check for significant faults
                if faults.actuator_effectiveness < 0.9:
                    print(f"  ⚠ WARNING: Actuator degradation detected!")
                if abs(faults.sensor_bias) > 0.1 or abs(faults.sensor_scale - 1.0) > 0.1:
                    print(f"  ⚠ WARNING: Sensor drift detected!")

            print(f"\nState: x={self.state[0]:.3f}, y={self.state[1]:.3f}, yaw={self.state[2]:.3f}")

            # Save image
            img_path = os.path.join("obs_imgs", f"frame_{t:03d}.png")
            cv2.imwrite(img_path, rgb_im)

        print("\n" + "="*60)
        print("TRAJECTORY COMPLETED")
        print("="*60)

        if latest_errors is not None:
            print("\nFinal Fault Diagnosis:")
            print(f"  Position error: ({latest_errors[0]:+.4f}, {latest_errors[1]:+.4f}) m")
            print(f"  Heading error: {latest_errors[2]:+.4f} rad ({np.degrees(latest_errors[2]):+.2f}°)")
            return latest_errors
        else:
            print("\nInsufficient data for fault diagnosis")
            return None

if __name__ == "__main__":
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        custom_controller = CustomController(sys.argv[1])
    else:
        custom_controller = CustomController()
    
    # msc = MotionSwitcherClient()
    # msc.SetTimeout(5.0)
    # msc.Init()
    # status, result = msc.CheckMode()
    # print(status)
    # print(result['name'])
    # time.sleep(5.)
    custom_controller.start()
    # custom_controller.start_visuomotor_control()
