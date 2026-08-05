from unitree_sdk2py.utils.thread import RecurrentThread
import time
import sys
import csv
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
import matplotlib.pyplot as plt
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

class SeparatingController:
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
        self.odom_samples = []
        self.latest_state = None
        self.latest_raw_position = None
        self.latest_yaw = 0.0
        self.start_time = None

        # self.init_vision_module()

        if dds is not None:
            ChannelFactoryInitialize(0, dds)
        else:
            ChannelFactoryInitialize(0)

        self.sport_client = SportClient()  
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        self.state_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.state_subscriber.Init(self.high_state_handler)

        # print(self.sport_client.SpeedLevel(-1))

        time.sleep(5.)

        # time.sleep(1.)
        # self.sport_client.StandUp()
        # time.sleep(1.)
        # self.sport_client.BalanceStand()
        # time.sleep(2.)

        print("custom controller initialization completed")

    def high_state_handler(self, msg: SportModeState_):
        position = msg.position
        yaw = msg.imu_state.rpy[2]

        if self.Init:
            self.Init = False
            self.Init_ang = yaw
            self.Mrot = np.array(
                [
                    [np.cos(self.Init_ang), np.sin(self.Init_ang)],
                    [-np.sin(self.Init_ang), np.cos(self.Init_ang)],
                ]
            )
            self.Init_pos[0] = self.Mrot[0, 0] * position[0] + self.Mrot[0, 1] * position[1]
            self.Init_pos[1] = self.Mrot[1, 0] * position[0] + self.Mrot[1, 1] * position[1]
            self.Init_pos[2] = position[2]

        x = self.Mrot[0, 0] * position[0] + self.Mrot[0, 1] * position[1] - self.Init_pos[0]
        y = self.Mrot[1, 0] * position[0] + self.Mrot[1, 1] * position[1] - self.Init_pos[1]
        z = position[2] - self.Init_pos[2]
        heading = wrap_angle(yaw - self.Init_ang)

        self.latest_state = {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "heading": float(heading),
            "vx": float(msg.velocity[0]),
            "vy": float(msg.velocity[1]),
            "vyaw": float(msg.yaw_speed),
        }
        self.latest_raw_position = [float(position[0]), float(position[1]), float(position[2])]
        self.latest_yaw = float(yaw)

        if self.start_time is not None:
            t_sec = time.time() - self.start_time
            self.odom_samples.append(
                {
                    "t_sec": float(t_sec),
                    "x": self.latest_state["x"],
                    "y": self.latest_state["y"],
                    "z": self.latest_state["z"],
                    "heading": self.latest_state["heading"],
                    "vx": self.latest_state["vx"],
                    "vy": self.latest_state["vy"],
                    "vyaw": self.latest_state["vyaw"],
                    "raw_x": self.latest_raw_position[0],
                    "raw_y": self.latest_raw_position[1],
                    "raw_z": self.latest_raw_position[2],
                    "raw_yaw": self.latest_yaw,
                }
            )

    def run_separating_controller(self):
        print("Starting open loop control")
        alpha = .7
        self.start_time = time.time()
        self.sport_client.Move(0.3735, alpha * 0.4361, alpha * 0.6000)
        time.sleep(1)
        self.sport_client.Move(0.5711, alpha * 0.5667, alpha * 0.2561)
        time.sleep(1)
        self.sport_client.StopMove()
        # Capture a little post-stop odometry too.
        time.sleep(0.5)
        self.start_time = None

    def save_odom_csv(self, out_csv_path: str):
        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
        with open(out_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "t_sec",
                    "x",
                    "y",
                    "z",
                    "heading",
                    "vx",
                    "vy",
                    "vyaw",
                    "raw_x",
                    "raw_y",
                    "raw_z",
                    "raw_yaw",
                ]
            )
            for s in self.odom_samples:
                writer.writerow(
                    [
                        s["t_sec"],
                        s["x"],
                        s["y"],
                        s["z"],
                        s["heading"],
                        s["vx"],
                        s["vy"],
                        s["vyaw"],
                        s["raw_x"],
                        s["raw_y"],
                        s["raw_z"],
                        s["raw_yaw"],
                    ]
                )
        print(f"Saved odom CSV: {out_csv_path}")

    def save_trajectory_plot(self, out_plot_path: str):
        if not self.odom_samples:
            print("No odom samples captured; skipping trajectory plot.")
            return
        x = [s["x"] for s in self.odom_samples]
        y = [s["y"] for s in self.odom_samples]

        plt.figure(figsize=(7, 7))
        plt.plot(x, y, color="tab:blue", linewidth=1.5, label="odom trajectory")
        plt.scatter([x[0]], [y[0]], marker="o", color="green", s=45, label="start")
        plt.scatter([x[-1]], [y[-1]], marker="x", color="red", s=55, label="end")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("Separating Controller - Robot Movement in 2D")
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()

        os.makedirs(os.path.dirname(out_plot_path), exist_ok=True)
        plt.savefig(out_plot_path, dpi=150)
        plt.close()
        print(f"Saved 2D trajectory plot: {out_plot_path}")


    
            

if __name__ == "__main__":
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")


    custom_controller = SeparatingController("enxa0cec85aea98")
    custom_controller.run_separating_controller()
    logs_dir = os.path.join(base_path, "dataset", "separating_controller_logs")
    ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    csv_path = os.path.join(logs_dir, f"odom_{ts}.csv")
    plot_path = os.path.join(logs_dir, f"trajectory_2d_{ts}.png")
    custom_controller.save_odom_csv(csv_path)
    custom_controller.save_trajectory_plot(plot_path)

    # print("Starting open loop control")
    # custom_controller.sport_client.Move(0.3735, 0.4361, 0.6000)
    # time.sleep(1)
    # custom_controller.sport_client.Move(0.5711, 0.5667, +0.2561)
    # time.sleep(1)
    # custom_controller.sport_client.StopMove()
    
    # msc = MotionSwitcherClient()
    # msc.SetTimeout(5.0)
    # msc.Init()
    # status, result = msc.CheckMode()
    # print(status)
    # print(result['name'])
    # time.sleep(5.)
    # custom_controller.start()
    # custom_controller.start_visuomotor_control()
