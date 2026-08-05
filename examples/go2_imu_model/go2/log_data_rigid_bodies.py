import argparse
import csv
import math
import os
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from visualize_rigid_bodies import generate_plots_from_csv
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_


def wrap_angle(angle: float) -> float:
    while angle > 0:
        angle -= 2 * np.pi
    while angle <= -2 * np.pi:
        angle += 2 * np.pi
    return angle


@dataclass
class SportSample:
    ts_ns: int
    x: float
    y: float
    heading: float
    vx: float
    vy: float
    vyaw: float


@dataclass
class RigidSample:
    ts_ns: int
    name: str
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float


class RigidBodiesSubscriber:
    def __init__(self, topic: str, callback):
        try:
            import rclpy
            from rclpy.node import Node
            from mocap4r2_msgs.msg import RigidBodies
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 dependencies missing. Install rclpy and mocap4r2_msgs."
            ) from exc

        self._rclpy = rclpy
        self._node_cls = Node
        self._msg_type = RigidBodies
        self._topic = topic
        self._callback = callback
        self._thread = None
        self._node = None
        self._running = False

    def start(self):
        self._rclpy.init(args=None)
        self._node = self._node_cls("go2_rigid_bodies_logger")
        self._node.create_subscription(
            self._msg_type, self._topic, self._callback, 10
        )
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        while self._running and self._rclpy.ok():
            self._rclpy.spin_once(self._node, timeout_sec=0.1)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._node is not None:
            self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


class DataLogger:
    def __init__(
        self,
        dds: Optional[str],
        rigid_body_name: str,
        sync_tolerance_ms: float,
        output_csv: str,
        rigid_topic: str,
        save_interval_sec: float,
        plots_dir: str,
    ):
        self.running = True
        self.rigid_body_name = rigid_body_name
        self.tolerance_ns = int(sync_tolerance_ms * 1e6)
        self.save_interval_ns = max(0, int(save_interval_sec * 1e9))
        self.lock = threading.Lock()
        self.sport_buffer = deque(maxlen=500)
        self.rigid_buffer = deque(maxlen=500)
        self.row_count = 0
        self.skipped_count = 0
        self.latest_state: Optional[SportSample] = None
        self.previous_state: Optional[SportSample] = None
        self.last_saved_ts_ns: Optional[int] = None

        if dds is not None:
            ChannelFactoryInitialize(0, dds)
        else:
            ChannelFactoryInitialize(0)

        self._init_transform_ready = False
        self._init_pos = np.array([0.0, 0.0, 0.0])
        self._init_ang = 0.0
        self._mrot = None

        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.csv_path = output_csv
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)
        self._open_csv()

        self.state_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.state_subscriber.Init(self.high_state_handler)

        self.rigid_subscriber = RigidBodiesSubscriber(
            topic=rigid_topic, callback=self.rigid_bodies_handler
        )
        self.rigid_subscriber.start()

        print(f"Logging to: {self.csv_path}")
        print(
            f"Filtering rigid body name='{self.rigid_body_name}', sync tolerance={sync_tolerance_ms:.1f} ms"
        )
        print(
            f"Save interval: {save_interval_sec:.3f} s ({'every synchronized sample' if self.save_interval_ns == 0 else 'downsampled logging'})"
        )

    def _open_csv(self):
        write_header = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
        self.csv_file = open(self.csv_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if write_header:
            self.csv_writer.writerow(
                [
                    "index",
                    "sample_ts_ns",
                    "dt_ms",
                    "rigid_name",
                    "sport_x",
                    "sport_y",
                    "sport_heading",
                    "sport_vx",
                    "sport_vy",
                    "sport_vyaw",
                    "sport_prev_x",
                    "sport_prev_y",
                    "sport_prev_heading",
                    "sport_prev_vx",
                    "sport_prev_vy",
                    "sport_prev_vyaw",
                    "rigid_x",
                    "rigid_y",
                    "rigid_z",
                    "rigid_qx",
                    "rigid_qy",
                    "rigid_qz",
                    "rigid_qw",
                ]
            )
            self.csv_file.flush()

    @staticmethod
    def _ts_ns_from_header_or_now(msg: Any) -> int:
        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None)
        sec = getattr(stamp, "sec", None)
        nanosec = getattr(stamp, "nanosec", None)
        if isinstance(sec, int) and isinstance(nanosec, int):
            return sec * 1_000_000_000 + nanosec
        return time.time_ns()

    @staticmethod
    def _extract_name(rb: Any) -> Optional[str]:
        for attr in ("name", "rigid_body_name", "id", "rigid_body_id"):
            value = getattr(rb, attr, None)
            if value is not None:
                return str(value)
        return None

    @staticmethod
    def _extract_pose(rb: Any):
        pose = getattr(rb, "pose", None)
        if pose is not None:
            position = getattr(pose, "position", None)
            orientation = getattr(pose, "orientation", None)
            if position is not None and orientation is not None:
                return (
                    float(position.x),
                    float(position.y),
                    float(position.z),
                    float(orientation.x),
                    float(orientation.y),
                    float(orientation.z),
                    float(orientation.w),
                )
        position = getattr(rb, "position", None)
        orientation = getattr(rb, "orientation", None)
        if position is not None and orientation is not None:
            return (
                float(position.x),
                float(position.y),
                float(position.z),
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            )
        return None

    def high_state_handler(self, msg: SportModeState_):
        position = msg.position
        yaw = msg.imu_state.rpy[2]

        if not self._init_transform_ready:
            self._init_transform_ready = True
            self._init_ang = yaw
            mrot = np.matrix(
                [
                    [math.cos(self._init_ang), math.sin(self._init_ang)],
                    [-math.sin(self._init_ang), math.cos(self._init_ang)],
                ]
            )
            self._init_pos[0] = (
                mrot.item((0, 0)) * position[0] + mrot.item((0, 1)) * position[1]
            )
            self._init_pos[1] = (
                mrot.item((1, 0)) * position[0] + mrot.item((1, 1)) * position[1]
            )
            self._init_pos[2] = position[2]
            self._mrot = mrot

        x = (
            self._mrot.item((0, 0)) * position[0]
            + self._mrot.item((0, 1)) * position[1]
            - self._init_pos[0]
        )
        y = (
            self._mrot.item((1, 0)) * position[0]
            + self._mrot.item((1, 1)) * position[1]
            - self._init_pos[1]
        )
        heading = wrap_angle(yaw - self._init_ang)
        sample = SportSample(
            ts_ns=time.time_ns(),
            x=float(x),
            y=float(y),
            heading=float(heading),
            vx=float(msg.velocity[0]),
            vy=float(msg.velocity[1]),
            vyaw=float(msg.yaw_speed),
        )

        with self.lock:
            self.previous_state = self.latest_state
            self.latest_state = sample
            self.sport_buffer.append(sample)
            self._pair_and_save_locked()

    def rigid_bodies_handler(self, msg):
        rigid_bodies = getattr(msg, "rigidbodies", None)
        if rigid_bodies is None:
            rigid_bodies = getattr(msg, "rigid_bodies", None)
        if rigid_bodies is None:
            return

        chosen = None
        for rb in rigid_bodies:
            rb_name = self._extract_name(rb)
            if rb_name == self.rigid_body_name:
                chosen = rb
                break
        if chosen is None:
            return

        pose = self._extract_pose(chosen)
        if pose is None:
            return

        ts_ns = self._ts_ns_from_header_or_now(msg)
        rigid_sample = RigidSample(
            ts_ns=ts_ns,
            name=self.rigid_body_name,
            x=pose[0],
            y=pose[1],
            z=pose[2],
            qx=pose[3],
            qy=pose[4],
            qz=pose[5],
            qw=pose[6],
        )

        with self.lock:
            self.rigid_buffer.append(rigid_sample)
            self._pair_and_save_locked()

    def _pair_and_save_locked(self):
        while self.sport_buffer and self.rigid_buffer:
            sport = self.sport_buffer[0]
            rigid_idx = min(
                range(len(self.rigid_buffer)),
                key=lambda i: abs(self.rigid_buffer[i].ts_ns - sport.ts_ns),
            )
            rigid = self.rigid_buffer[rigid_idx]
            dt_ns = rigid.ts_ns - sport.ts_ns

            if abs(dt_ns) <= self.tolerance_ns:
                self.sport_buffer.popleft()
                self.rigid_buffer.remove(rigid)
                prev = self.previous_state
                if prev is None:
                    prev = sport

                if (
                    self.last_saved_ts_ns is not None
                    and self.save_interval_ns > 0
                    and sport.ts_ns - self.last_saved_ts_ns < self.save_interval_ns
                ):
                    self.skipped_count += 1
                    continue

                self.csv_writer.writerow(
                    [
                        self.row_count,
                        sport.ts_ns,
                        dt_ns / 1e6,
                        rigid.name,
                        sport.x,
                        sport.y,
                        sport.heading,
                        sport.vx,
                        sport.vy,
                        sport.vyaw,
                        prev.x,
                        prev.y,
                        prev.heading,
                        prev.vx,
                        prev.vy,
                        prev.vyaw,
                        rigid.x,
                        rigid.y,
                        rigid.z,
                        rigid.qx,
                        rigid.qy,
                        rigid.qz,
                        rigid.qw,
                    ]
                )
                self.csv_file.flush()
                self.last_saved_ts_ns = sport.ts_ns
                self.row_count += 1
                if self.row_count % 50 == 0:
                    print(f"Saved {self.row_count} synchronized samples")
                continue

            rigid_oldest = self.rigid_buffer[0]
            if rigid_oldest.ts_ns < sport.ts_ns - self.tolerance_ns:
                self.rigid_buffer.popleft()
            elif sport.ts_ns < rigid_oldest.ts_ns - self.tolerance_ns:
                self.sport_buffer.popleft()
            else:
                break

    def stop(self):
        if not self.running:
            return
        self.running = False
        try:
            self.rigid_subscriber.stop()
        finally:
            self.csv_file.close()
        try:
            generate_plots_from_csv(self.csv_path, self.plots_dir)
        except ImportError:
            print("matplotlib is not installed. Skipping plot generation.")
        except Exception as exc:
            print(f"Plot generation failed: {exc}")
        print(
            f"Stopped. Total synchronized samples saved: {self.row_count} (skipped by save interval: {self.skipped_count})"
        )


def main():
    default_csv_name = time.strftime("data_rigid_bodies_%Y-%m-%d_%H-%M-%S.csv")
    default_csv_path = os.path.join(
        os.path.dirname(__file__), "dataset", default_csv_name
    )

    parser = argparse.ArgumentParser(
        description="Log synchronized GO2 sport state + ROS2 rigid body data to CSV."
    )
    parser.add_argument("--dds", type=str, default=None, help="Network interface/IP for DDS init.")
    parser.add_argument(
        "--rigid-body-name",
        type=str,
        default="4",
        help="Rigid body name/id to save from /rigid_bodies.",
    )
    parser.add_argument(
        "--rigid-topic",
        type=str,
        default="/rigid_bodies",
        help="ROS2 topic for mocap4r2_msgs/msg/RigidBodies.",
    )
    parser.add_argument(
        "--sync-tolerance-ms",
        type=float,
        default=50.0,
        help="Max timestamp difference for synchronized samples.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=default_csv_path,
        help="Output CSV path. Defaults to a timestamped filename to avoid overwriting old logs.",
    )
    parser.add_argument(
        "--save-interval-sec",
        type=float,
        default=0.1,
        help="Minimum time interval between saved rows. 0 saves every synchronized sample.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dataset"),
        help="Directory where trajectory plots are saved as PNG files.",
    )
    args = parser.parse_args()

    logger = DataLogger(
        dds=args.dds,
        rigid_body_name=args.rigid_body_name,
        sync_tolerance_ms=args.sync_tolerance_ms,
        output_csv=args.output_csv,
        rigid_topic=args.rigid_topic,
        save_interval_sec=args.save_interval_sec,
        plots_dir=args.plots_dir,
    )

    def _shutdown_handler(_sig, _frame):
        logger.stop()

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    try:
        while logger.running:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        logger.stop()


if __name__ == "__main__":
    main()
