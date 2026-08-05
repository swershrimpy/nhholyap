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
from unitree_sdk2py.go2.video.video_client import VideoClient
import math
from dataclasses import dataclass
import numpy as np
import os
import cv2, glob
import csv
 
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
 
class DataLogger:
    def __init__(self, dds=None):
        self.latest_state = None
        self.previous_state = None
 
        base_path = os.path.dirname(__file__)
        self.frames_dir = os.path.join(base_path, "dataset/frames")
        self.csv_path = os.path.join(base_path, "dataset/data.csv")
 
        files = (glob.glob(self.frames_dir+"/frame_*.png"))  
        self.frame_count = len(files)
 
        self.window = "front_camera"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
 
        if dds is not None:
            ChannelFactoryInitialize(0, dds)
        else:
            ChannelFactoryInitialize(0)
        
        self.video_client = VideoClient()
        self.video_client.SetTimeout(3.0)
        self.video_client.Init()
 
        self.Init = True
        self.Init_pos = np.array([0., 0., 0.])
        self.Init_ang = 0.
        self.Mrot = None
 
        self.state_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.state_subscriber.Init(self.high_state_handler)
        
        time.sleep(2.)
 
        print("data logger initialization completed")
 
    def high_state_handler(self, msg:SportModeState_):
        # print("position: {:.3f}, {:.3f}, {:.3f}".format(msg.position[0], msg.position[1], msg.position[2]))
        # print("yaw: {:.3f}".format(quat_to_yaw(quat[0], quat[1], quat[2], quat[3])))
        position = msg.position
        # quat = msg.imu_state.quaternion
        # yaw = quat_to_yaw(quat[0], quat[1], quat[2], quat[3])
        yaw = msg.imu_state.rpy[2]
        # print(yaw)
        if self.Init:
            self.Init = False
            self.Init_ang = yaw
 
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-
                                np.sin(self.Init_ang), np.cos(self.Init_ang)]])
            self.Init_pos[0] = Mrot.item((0,0))*position[0] + Mrot.item((0,1))*position[1]
            self.Init_pos[1] = Mrot.item((1,0))*position[0] + Mrot.item((1,1))*position[1]
            self.Init_pos[2] = position[2]
            self.Mrot = Mrot
 
        x = self.Mrot.item((0,0))*position[0] + self.Mrot.item((0,1))*position[1] - self.Init_pos[0]
        y = self.Mrot.item((1,0))*position[0] + self.Mrot.item((1,1))*position[1] - self.Init_pos[1]
        heading = wrap_angle(yaw - self.Init_ang)
 
        self.previous_state = self.latest_state
        self.latest_state = {'x': x, 'y': y, 'heading': heading,
                             'vx': msg.velocity[0], 'vy': msg.velocity[1], 'vyaw': msg.yaw_speed}
 
    def get_frame(self):
        code, data = self.video_client.GetImageSample()
 
        if code != 0:
            print("get image sample error. code:", code)
        else:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q or ESC
                print("Exit requested. Shutting down.")
                cv2.destroyWindow(self.window)
                return False
            
            image_data = np.frombuffer(bytes(data), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            
            try:
                # Display image
                cv2.imshow(self.window, image)
            except cv2.error:
                print("no frame recieved")
            else:
                if key == ord('s'):
                    img_path = os.path.join(self.frames_dir, f"frame_{self.frame_count:05d}.png")
                    cv2.imwrite(img_path, image)
                    print(self.latest_state)
 
                    # Write CSV row
                    with open(self.csv_path, 'a', newline='') as f:
                        w = csv.writer(f)
                        w.writerow([
                            self.frame_count, self.latest_state['x'], self.latest_state['y'], self.latest_state['heading'],
                            self.latest_state['vx'], self.latest_state['vy'], self.latest_state['vyaw'],
                            self.previous_state['x'], self.previous_state['y'], self.previous_state['heading'],
                            self.previous_state['vx'], self.previous_state['vy'], self.previous_state['vyaw']
                        ])
                    self.frame_count += 1
                    print("Saved data to "+img_path)
        
        return True
 
if __name__ == "__main__":
 
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")
    
    if len(sys.argv)>1:
        data_logger = DataLogger(sys.argv[1])
    else:
        data_logger = DataLogger()
 
    while data_logger.get_frame():
        pass