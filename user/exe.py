import os
import sys
import math
import time
import signal
import subprocess
from datetime import datetime
from enum import Enum

import cv2
import requests
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from pinky_interfaces.srv import SetLed

import tf2_ros

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QFont, QTextCursor, QImage, QPixmap, QPainter, QPen, QColor


class PatrolState(Enum):
    IDLE = 1
    PATROL = 2
    ALERT = 3
    RETURNING = 4
    DOCKING = 5
    ESCAPING = 6  # 좁은 공간 탈출(Costmap 초기화 + 후진) 상태


class GuiEmitter(QObject):
    log_signal = pyqtSignal(str)
    hazard_log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str, str)


class MiniMapWidget(QLabel):
    """police_map3.pgm + yaml 기반 실시간 로봇 위치/경로/마커 표시 미니맵 위젯"""
    def __init__(self, pgm_path='police_map5.pgm', yaml_path='police_map5.yaml'):
        super().__init__()
        self.setFixedSize(480, 260)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #111111; border: 2px solid #333333; border-radius: 8px;")

        # police_map5.yaml 메타데이터 기본값
        self.resolution = 0.050
        self.origin_x = -3.867
        self.origin_y = -4.315
        self.pgm_path = pgm_path

        self.load_yaml_metadata(yaml_path)
        self.base_pixmap = self.load_map_pixmap(self.pgm_path)

        self.waypoints = []
        self.robot_pose = None
        self.home_pose = None
        self.hazard_points = []
        self.trail_points = []

    def load_yaml_metadata(self, yaml_path):
        if not os.path.exists(yaml_path):
            return
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('resolution:'):
                        self.resolution = float(line.split(':')[1].strip())
                    elif line.startswith('origin:'):
                        vals = line.split(':')[1].strip().strip('[]').split(',')
                        self.origin_x = float(vals[0])
                        self.origin_y = float(vals[1])
        except Exception:
            pass

    def load_map_pixmap(self, pgm_path):
        if not os.path.exists(pgm_path):
            return None
        img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def world_to_pixel(self, wx, wy, map_w, map_h):
        px = (wx - self.origin_x) / self.resolution
        py = map_h - ((wy - self.origin_y) / self.resolution)
        return px, py

    def update_state(self, waypoints, robot_pose, home_pose, hazard_points):
        self.waypoints = waypoints
        self.robot_pose = robot_pose
        self.home_pose = home_pose
        self.hazard_points = hazard_points

        if robot_pose is not None:
            rx, ry, _ = robot_pose
            if not self.trail_points or math.hypot(rx - self.trail_points[-1][0], ry - self.trail_points[-1][1]) > 0.05:
                self.trail_points.append((rx, ry))
                if len(self.trail_points) > 300:
                    self.trail_points.pop(0)

        self.render_map()

    def render_map(self):
        if self.base_pixmap is None:
            self.setText("맵 파일(police_map5.pgm)을 찾을 수 없습니다.\nexe.py와 같은 폴더에 넣어주세요.")
            return

        canvas = self.base_pixmap.copy()
        map_w = canvas.width()
        map_h = canvas.height()

        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.waypoints:
            painter.setPen(QPen(QColor(241, 196, 15, 160), 1, Qt.PenStyle.DashLine))
            wp_pixels = [self.world_to_pixel(wx, wy, map_w, map_h) for wx, wy in self.waypoints]
            for i in range(len(wp_pixels)):
                p1 = wp_pixels[i]
                p2 = wp_pixels[(i + 1) % len(wp_pixels)]
                painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

            painter.setBrush(QColor(241, 196, 15))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            for idx, (px, py) in enumerate(wp_pixels):
                painter.drawEllipse(int(px) - 3, int(py) - 3, 6, 6)

        if len(self.trail_points) > 1:
            painter.setPen(QPen(QColor(46, 204, 113, 180), 1))
            for i in range(len(self.trail_points) - 1):
                p1 = self.world_to_pixel(*self.trail_points[i], map_w, map_h)
                p2 = self.world_to_pixel(*self.trail_points[i + 1], map_w, map_h)
                painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

        if self.home_pose is not None:
            hx, hy = self.world_to_pixel(self.home_pose[0], self.home_pose[1], map_w, map_h)
            painter.setBrush(QColor(52, 152, 219))
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawRect(int(hx) - 4, int(hy) - 4, 8, 8)

        if self.hazard_points:
            painter.setPen(QPen(QColor(231, 76, 60), 2))
            for wx, wy in self.hazard_points:
                px, py = self.world_to_pixel(wx, wy, map_w, map_h)
                painter.drawLine(int(px) - 4, int(py) - 4, int(px) + 4, int(py) + 4)
                painter.drawLine(int(px) + 4, int(py) - 4, int(px) - 4, int(py) + 4)

        if self.robot_pose is not None:
            rx, ry, yaw = self.robot_pose
            px, py = self.world_to_pixel(rx, ry, map_w, map_h)
            painter.setBrush(QColor(0, 255, 102))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawEllipse(int(px) - 5, int(py) - 5, 10, 10)

            dir_x = px + math.cos(yaw) * 10
            dir_y = py - math.sin(yaw) * 10
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawLine(int(px), int(py), int(dir_x), int(dir_y))

        painter.end()

        scaled = canvas.scaled(
            self.width() - 4,
            self.height() - 4,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        )
        self.setPixmap(scaled)


class VideoStreamThread(QThread):
    frame_received = pyqtSignal(QImage)
    log_message = pyqtSignal(str)
    marker_detected = pyqtSignal(str, int)
    marker_tracking = pyqtSignal(int, float, float, int)

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.running = False
        self.last_detected_time = {}

        self.dict_list = [
            ("4x4", cv2.aruco.DICT_4X4_250),
            ("5x5", cv2.aruco.DICT_5X5_250),
            ("6x6", cv2.aruco.DICT_6X6_250),
            ("ORIG", cv2.aruco.DICT_ARUCO_ORIGINAL),
        ]

        self.detectors = []
        self.use_new_api = hasattr(cv2.aruco, 'ArucoDetector')

        for name, dict_id in self.dict_list:
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            if self.use_new_api:
                params = cv2.aruco.DetectorParameters()
                detector = cv2.aruco.ArucoDetector(aruco_dict, params)
                self.detectors.append((name, detector))
            else:
                params = cv2.aruco.DetectorParameters_create()
                self.detectors.append((name, (aruco_dict, params)))

    def detect_aruco(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = gray.shape
        now = time.time()

        for dict_name, detector_obj in self.detectors:
            if self.use_new_api:
                corners, ids, _ = detector_obj.detectMarkers(gray)
            else:
                aruco_dict, params = detector_obj
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                for i, marker_id in enumerate(ids.flatten()):
                    m_id = int(marker_id)
                    pts = corners[i].reshape((4, 2))
                    top_left = (int(pts[0][0]), int(pts[0][1]))

                    center_x = float(np.mean(pts[:, 0]))
                    width_px = float(np.linalg.norm(pts[1] - pts[0]))
                    self.marker_tracking.emit(m_id, center_x, width_px, frame_w)

                    if m_id == 66:
                        color = (255, 200, 0)
                        label_text = f"[{dict_name}] ID: 66 (HOME | W:{int(width_px)}px)"
                    elif m_id == 42:
                        color = (0, 0, 255)
                        label_text = f"[{dict_name}] ID: 42 (HAZARD!)"
                    else:
                        color = (0, 255, 0)
                        label_text = f"[{dict_name}] ID: {m_id}"

                    cv2.putText(
                        frame,
                        label_text,
                        (top_left[0], max(top_left[1] - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2
                    )

                    key = f"{dict_name}_{m_id}"
                    if key not in self.last_detected_time or (now - self.last_detected_time[key]) > 3.0:
                        self.last_detected_time[key] = now
                        self.log_message.emit(f"🎯 ArUco 인식됨 -> 규격: {dict_name}, ID: {m_id}")
                        self.marker_detected.emit(dict_name, m_id)

        return frame

    def run(self):
        self.running = True
        time.sleep(1.5)
        bytes_data = b''

        while self.running:
            try:
                stream = requests.get(self.url, stream=True, timeout=3.0)
                for chunk in stream.iter_content(chunk_size=1024):
                    if not self.running:
                        break
                    bytes_data += chunk
                    start = bytes_data.find(b'\xff\xd8')
                    end = bytes_data.find(b'\xff\xd9')
                    if start != -1 and end != -1:
                        jpg = bytes_data[start:end + 2]
                        bytes_data = bytes_data[end + 2:]
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            frame = self.detect_aruco(frame)

                            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            h, w, ch = rgb_image.shape
                            qt_image = QImage(
                                rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888
                            ).copy()
                            self.frame_received.emit(qt_image)
            except Exception:
                if self.running:
                    time.sleep(1.0)

    def stop(self):
        self.running = False
        self.wait(1000)


class PatrolTaskManager(Node):
    def __init__(self, emitter):
        super().__init__('patrol_task_manager')
        self.emitter = emitter
        self.state = PatrolState.IDLE
        self.toggle = False
        self.goal_handle = None
        self.nav_view_process = None

        self.default_home_pose = (0.16512, 0.12211, 0.0, 1.0)
        self.home_pose = None
        self.current_robot_pose = None
        self.hazard_locations = []

        # 도킹용 변수
        self.docking_start_time = 0.0
        self.last_home_marker_seen = 0.0
        self.home_marker_cx = 160.0
        self.home_marker_width = 0.0
        self.home_frame_width = 320
        self.target_docking_width_ratio = 0.39

        # 좁은 공간 갇힘 감지 및 탈출용 변수
        self.last_progress_pose = None
        self.last_progress_time = time.time()
        self.escape_start_time = 0.0
        self.escape_return_state = PatrolState.PATROL
        self.goal_arrival_tolerance = 0.22  # 좁은 공간에서 22cm 이내 진입 시 도착 인정

        self.rotation_count = 1
        self.hazard_count_in_rotation = 0
        self.total_hazard_count = 0
        self.last_hazard_time = 0.0

        self.waypoints = [
            (0.41044, -0.20069),
            (0.978307, 0.189556),
            (0.754271, 1.726295),
            (1.19979, 1.849158),
            (1.020600, 3.04068),
            (0.719944, 2.862250)
        ]
        self.current_wp_idx = 0

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.led_client = self.create_client(SetLed, '/set_led')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Nav2 Costmap 초기화 서비스 클라이언트 (좁은 공간 갇힘 탈출용)
        self.clear_local_costmap_client = self.create_client(
            ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap'
        )
        self.clear_global_costmap_client = self.create_client(
            ClearEntireCostmap, '/global_costmap/clear_entirely_global_costmap'
        )

        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_pose_callback,
            10
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.lcd_pub = self.create_publisher(String, '/lcd_cmd', 10)
        self.bringup_pub = self.create_publisher(String, '/bringup_cmd', 10)
        self.nav_pub = self.create_publisher(String, '/nav_cmd', 10)
        self.cam_pub = self.create_publisher(String, '/cam_cmd', 10)
        self.buzzer_pub = self.create_publisher(String, '/buzzer_cmd', 10)

        self.timer = self.create_timer(0.1, self.control_loop)
        self.led_tick = 0
        self.emit_log('시스템 초기화 완료 (현재 상태: IDLE)')

    def clear_nav2_costmaps(self):
        """좁은 공간에서 팽창된 Costmap 장애물 영역을 즉시 초기화"""
        req = ClearEntireCostmap.Request()
        if self.clear_local_costmap_client.service_is_ready():
            self.clear_local_costmap_client.call_async(req)
        if self.clear_global_costmap_client.service_is_ready():
            self.clear_global_costmap_client.call_async(req)
        self.emit_log('🧹 Nav2 Local/Global Costmap 초기화 요청 전송')

    def trigger_narrow_escape(self, next_state):
        """좁은 공간에 갇혔을 때 Costmap 초기화 후 짧은 후진 탈출 기동 시작"""
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None

        self.clear_nav2_costmaps()
        self.escape_return_state = next_state
        self.escape_start_time = time.time()
        self.state = PatrolState.ESCAPING
        self.emit_log('🔄 좁은 공간 갇힘 감지 -> Costmap 초기화 및 후진 탈출 기동 시작')
        self.emitter.status_signal.emit('🔄 좁은 공간 탈출 중...', '#8e44ad')

    def play_hazard_buzzer(self):
        msg = String()
        msg.data = 'hazard'
        self.buzzer_pub.publish(msg)
        self.emit_log('🔊 라즈베리파이 위험물 경고 부저음 재생 명령 전송 (/buzzer_cmd: hazard)')

    def amcl_pose_callback(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self.current_robot_pose = (p.x, p.y, o.z, o.w)

    def get_latest_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            t = trans.transform.translation
            r = trans.transform.rotation
            return (t.x, t.y, r.z, r.w)
        except Exception:
            return self.current_robot_pose

    def get_robot_2d_with_yaw(self):
        pose = self.get_latest_robot_pose()
        if pose is None:
            return None
        x, y, oz, ow = pose
        yaw = math.atan2(2.0 * (ow * oz), 1.0 - 2.0 * (oz * oz))
        return (x, y, yaw)

    def on_aruco_tracking(self, marker_id, center_x, width_px, frame_w):
        if marker_id == 66:
            self.last_home_marker_seen = time.time()
            self.home_marker_cx = center_x
            self.home_marker_width = width_px
            self.home_frame_width = frame_w

    def on_aruco_detected(self, dict_name, marker_id):
        if marker_id == 66:
            pose = self.get_latest_robot_pose()
            if pose is not None:
                is_first = (self.home_pose is None)
                self.home_pose = pose
                if is_first:
                    self.emit_log(f'🏠 ArUco ID 66 인식! 현재 위치를 홈(출발지)으로 등록: ({pose[0]:.3f}, {pose[1]:.3f})')
                    self.emit_hazard_log(f'🏠 [홈 등록] ID 66 기준 출발지 좌표 설정 완료 ({pose[0]:.2f}, {pose[1]:.2f})')
            else:
                self.emit_log('⚠️ ArUco ID 66 인식되었으나 로봇 위치(Nav2/AMCL)를 아직 수신하지 못했습니다.')

        elif marker_id == 42:
            now = time.time()
            if now - self.last_hazard_time > 5.0:
                self.last_hazard_time = now
                self.hazard_count_in_rotation += 1
                self.total_hazard_count += 1

                self.play_hazard_buzzer()

                pose = self.get_latest_robot_pose()
                coord_str = ""
                if pose is not None:
                    self.hazard_locations.append((pose[0], pose[1]))
                    coord_str = f" [좌표: {pose[0]:.2f}, {pose[1]:.2f}]"

                msg = (
                    f'⚠️ 위험물 발견! (ID: 42){coord_str} | '
                    f'{self.rotation_count}회차 누적: {self.hazard_count_in_rotation}건 '
                    f'(전체 총 {self.total_hazard_count}건)'
                )
                self.emit_log(msg)
                self.emit_hazard_log(msg)

    def emit_log(self, message):
        now = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        formatted_msg = f'[{now}] {message}'
        self.get_logger().info(message)
        self.emitter.log_signal.emit(formatted_msg)

    def emit_hazard_log(self, message):
        now = datetime.now().strftime('%H:%M:%S')
        formatted_msg = f'[{now}] {message}'
        self.emitter.hazard_log_signal.emit(formatted_msg)

    def stop_robot_motors(self):
        """즉시 정지용 제로 속도 여러 번 발행하여 확실히 정지"""
        twist = Twist()
        for _ in range(3):
            self.cmd_vel_pub.publish(twist)

    def start_camera(self):
        msg = String()
        msg.data = 'on'
        self.cam_pub.publish(msg)
        self.emit_log('📷 라즈베리파이 카메라 실행 명령 전송 (/cam_cmd: on)')

    def stop_camera(self):
        msg = String()
        msg.data = 'off'
        self.cam_pub.publish(msg)
        self.emit_log('📴 라즈베리파이 카메라 종료 명령 전송 (/cam_cmd: off)')

    def start_nav_view(self):
        if self.nav_view_process is None or self.nav_view_process.poll() is not None:
            try:
                env = os.environ.copy()
                env['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins'

                cmd = 'ros2 launch pinky_navigation nav2_view.launch.xml'
                self.nav_view_process = subprocess.Popen(
                    cmd,
                    shell=True,
                    executable='/bin/bash',
                    env=env,
                    preexec_fn=os.setsid
                )
                self.emit_log('🖥️ PC Nav2 View (RViz2) 실행됨')
            except Exception as e:
                self.emit_log(f'⚠️ Nav2 View 실행 실패: {e}')

    def stop_nav_view(self):
        if self.nav_view_process is not None and self.nav_view_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.nav_view_process.pid), signal.SIGINT)
                self.nav_view_process = None
                self.emit_log('❎ PC Nav2 View (RViz2) 종료됨')
            except Exception as e:
                self.emit_log(f'⚠️ Nav2 View 종료 실패: {e}')

    def start_navigation(self):
        msg = String()
        msg.data = 'on'
        self.nav_pub.publish(msg)
        self.emit_log('🗺️ 라즈베리파이 Navigation 실행 명령 전송 (/nav_cmd: on)')

    def stop_navigation(self):
        msg = String()
        msg.data = 'off'
        self.nav_pub.publish(msg)
        self.emit_log('🛑 라즈베리파이 Navigation 종료 명령 전송 (/nav_cmd: off)')

    def start_bringup(self):
        msg = String()
        msg.data = 'on'
        self.bringup_pub.publish(msg)
        self.emit_log('🚀 라즈베리파이 Bringup + LED Server 실행 명령 전송')

    def stop_bringup(self):
        msg = String()
        msg.data = 'off'
        self.bringup_pub.publish(msg)
        self.emit_log('🔌 라즈베리파이 Bringup + LED Server 종료 명령 전송')

    def start_lcd_script(self):
        msg = String()
        msg.data = 'on'
        self.lcd_pub.publish(msg)
        self.emit_log('라즈베리파이 LCD 실행 명령 전송 (/lcd_cmd: on)')

    def stop_lcd_script(self):
        msg = String()
        msg.data = 'off'
        self.lcd_pub.publish(msg)
        self.emit_log('라즈베리파이 LCD 종료 명령 전송 (/lcd_cmd: off)')

    def set_led(self, pixels, r, g, b):
        if not self.led_client.service_is_ready():
            return
        req = SetLed.Request()
        req.command = 'set_pixel'
        req.pixels = pixels
        req.r, req.g, req.b = r, g, b
        self.led_client.call_async(req)

    def reset_stuck_monitor(self):
        self.last_progress_pose = self.get_latest_robot_pose()
        self.last_progress_time = time.time()

    def handle_event(self, event):
        all_pixels = [0, 1, 2, 3, 4, 5, 6, 7]

        if event == 'start' and self.state == PatrolState.IDLE:
            if self.home_pose is None:
                current = self.get_latest_robot_pose()
                if current is not None:
                    self.home_pose = current
                    self.emit_log(f'📍 초기 출발 위치 임시 저장: ({current[0]:.3f}, {current[1]:.3f}) (ID 66 인식 시 갱신됨)')

            self.clear_nav2_costmaps()
            self.state = PatrolState.PATROL
            self.reset_stuck_monitor()
            self.set_led(all_pixels, 0, 0, 255)
            self.emit_log(f'▶ 순찰 모드 시작 (PATROL - {self.rotation_count}회차 로테이션)')
            self.emit_hazard_log(f'▶ {self.rotation_count}회차 순찰 로테이션 시작')
            self.emitter.status_signal.emit('순찰 중 (PATROL)', '#2980b9')
            self.send_next_waypoint()

        elif event == 'intruder_detected' and self.state != PatrolState.ALERT:
            self.state = PatrolState.ALERT
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
                self.goal_handle = None
            self.stop_robot_motors()
            self.play_hazard_buzzer()
            self.emit_log('🚨 비상 상황 발생! 즉시 정지, 경광등 점멸, 부저 경고 및 LCD 가동 (ALERT)')
            self.emitter.status_signal.emit('🚨 비상 경광등 + LCD 가동', '#c0392b')
            self.start_lcd_script()

        elif event == 'clear' and self.state == PatrolState.ALERT:
            self.clear_nav2_costmaps()
            self.state = PatrolState.PATROL
            self.reset_stuck_monitor()
            self.stop_lcd_script()
            self.set_led(all_pixels, 0, 0, 255)
            self.emit_log('✅ 비상 상황 해제. 순찰을 재개합니다.')
            self.emitter.status_signal.emit('순찰 재개 (PATROL)', '#2980b9')
            self.send_next_waypoint()

        elif event == 'stop_here':
            # 현재 그 자리에서 즉시 정지 (IDLE)
            self.state = PatrolState.IDLE
            self.stop_lcd_script()
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
                self.goal_handle = None
            self.stop_robot_motors()
            self.set_led(all_pixels, 0, 255, 0)
            self.emit_log('⏹ 즉시 정지 요청됨 -> 현재 위치에서 주행을 멈추고 대기(IDLE)합니다.')
            self.emitter.status_signal.emit('⏹ 현재 위치 정지 (IDLE)', '#34495e')

        elif event == 'return_home':
            # 홈 좌표 복귀 후 ArUco ID 66 근접 도킹
            if self.state == PatrolState.RETURNING or self.state == PatrolState.DOCKING:
                return
            self.stop_lcd_script()
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
                self.goal_handle = None

            self.clear_nav2_costmaps()
            self.state = PatrolState.RETURNING
            self.reset_stuck_monitor()
            self.set_led(all_pixels, 255, 165, 0)
            self.emit_log('🏠 홈 복귀 요청됨 -> 홈 좌표로 1차 복귀 후 ArUco ID 66에 근접 도킹합니다.')
            self.emitter.status_signal.emit('🏠 홈 좌표로 복귀 중...', '#d35400')
            self.send_home_goal()

    def send_home_goal(self):
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.emit_log('⚠️ Nav2 서버 미연결 상태 - 즉시 ArUco ID 66 도킹 탐색으로 전환합니다.')
            self.start_aruco_docking()
            return

        target_pose = self.home_pose if self.home_pose is not None else self.default_home_pose
        x, y, oz, ow = target_pose

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.orientation.z = float(oz)
        goal_msg.pose.pose.orientation.w = float(ow)

        self.reset_stuck_monitor()
        self.emit_log(f'🏠 홈 좌표 ({x:.3f}, {y:.3f})로 1차 복귀 주행 시작')
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def start_aruco_docking(self):
        all_pixels = [0, 1, 2, 3, 4, 5, 6, 7]
        self.state = PatrolState.DOCKING
        self.docking_start_time = time.time()
        self.set_led(all_pixels, 0, 255, 255)
        self.emit_log('🎯 홈 부근 도착! ArUco ID 66 마커 정밀 근접 도킹을 시작합니다...')
        self.emitter.status_signal.emit('🎯 ArUco ID 66 근접 도킹 중...', '#16a085')

    def finish_stop_at_home(self, reason_msg='⏹ 홈(ID 66) 근접 도킹 완료! 순찰이 종료되었습니다. (IDLE)'):
        all_pixels = [0, 1, 2, 3, 4, 5, 6, 7]
        self.stop_robot_motors()
        self.state = PatrolState.IDLE
        self.set_led(all_pixels, 0, 255, 0)
        self.emit_log(reason_msg)
        self.emit_hazard_log('🏠 홈(ID 66) 근접 복귀 완료 및 순찰 종료')
        self.emitter.status_signal.emit('대기 중 (IDLE)', '#2c3e50')

    def advance_to_next_waypoint(self):
        if self.current_wp_idx == len(self.waypoints) - 1:
            rot_msg = (
                f'🔄 순찰 로테이션 {self.rotation_count}회차 완료! '
                f'(이번 바퀴 위험물 발견: {self.hazard_count_in_rotation}건)'
            )
            self.emit_log(rot_msg)
            self.emit_hazard_log(rot_msg)

            self.rotation_count += 1
            self.hazard_count_in_rotation = 0
            self.emit_hazard_log(f'▶ {self.rotation_count}회차 순찰 로테이션 시작')

        self.current_wp_idx = (self.current_wp_idx + 1) % len(self.waypoints)
        self.send_next_waypoint()

    def send_next_waypoint(self):
        if self.state != PatrolState.PATROL:
            return
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.emit_log('⚠️ Nav2 서버 미연결 상태 (이동 명령 건너뜀)')
            return

        x, y = self.waypoints[self.current_wp_idx]
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.orientation.w = 1.0

        self.reset_stuck_monitor()
        self.emit_log(f'웨이포인트 [{self.current_wp_idx}] ({x}, {y})로 이동 시작')
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        if self.state not in (PatrolState.PATROL, PatrolState.RETURNING):
            return
        self.goal_handle = future.result()
        if self.goal_handle.accepted:
            result_future = self.goal_handle.get_result_async()
            result_future.add_done_callback(self.get_result_callback)
        else:
            self.emit_log('⚠️ 이동 요청이 거부되었습니다. Costmap 초기화 후 재시도합니다.')
            self.trigger_narrow_escape(self.state)

    def get_result_callback(self, future):
        # 사용자가 이미 정지/비상 버튼을 누른 경우 콜백 무시
        if self.state not in (PatrolState.PATROL, PatrolState.RETURNING):
            return

        result_wrapper = future.result()
        status = result_wrapper.status  # 4: SUCCEEDED, 5: CANCELED, 6: ABORTED

        if status == 5:
            return

        # 좁은 공간에서 경로 계획 실패(ABORTED=6) 시 Costmap 초기화 및 탈출 후진 실행
        if status == 6:
            self.emit_log('⚠️ 좁은 구역 경로 막힘(ABORTED) 발생 -> 탈출 및 Costmap 초기화 실행')
            self.trigger_narrow_escape(self.state)
            return

        if self.state == PatrolState.RETURNING:
            self.start_aruco_docking()
            return

        if self.state == PatrolState.PATROL:
            self.emit_log(f'웨이포인트 [{self.current_wp_idx}] 도착 완료!')
            self.advance_to_next_waypoint()

    def control_loop(self):
        now = time.time()

        # 1. 비상 상태(ALERT) 경광등 제어
        if self.state == PatrolState.ALERT:
            self.led_tick += 1
            if self.led_tick % 3 == 0:
                left_pixels = [0, 1, 2, 3]
                right_pixels = [4, 5, 6, 7]
                if self.toggle:
                    self.set_led(left_pixels, 255, 0, 0)
                    self.set_led(right_pixels, 0, 0, 255)
                else:
                    self.set_led(left_pixels, 0, 0, 255)
                    self.set_led(right_pixels, 255, 0, 0)
                self.toggle = not self.toggle

        # 2. 순찰/복귀 중 좁은 공간 조기 도착 판정 및 갇힘(Stuck) 감시
        elif self.state in (PatrolState.PATROL, PatrolState.RETURNING):
            pose = self.get_latest_robot_pose()
            if pose is not None:
                rx, ry = pose[0], pose[1]

                # 좁은 공간 진입 시 목표 반경(0.22m) 내 진입하면 제자리 회전 없이 다음 단계로 즉시 전환
                if self.state == PatrolState.PATROL:
                    tx, ty = self.waypoints[self.current_wp_idx]
                    if math.hypot(tx - rx, ty - ry) <= self.goal_arrival_tolerance:
                        if self.goal_handle is not None:
                            self.goal_handle.cancel_goal_async()
                            self.goal_handle = None
                        self.emit_log(f'✅ 웨이포인트 [{self.current_wp_idx}] 근접 도달 완료! 다음 지점으로 이동')
                        self.advance_to_next_waypoint()
                        return
                elif self.state == PatrolState.RETURNING:
                    target_pose = self.home_pose if self.home_pose is not None else self.default_home_pose
                    if math.hypot(target_pose[0] - rx, target_pose[1] - ry) <= 0.25:
                        if self.goal_handle is not None:
                            self.goal_handle.cancel_goal_async()
                            self.goal_handle = None
                        self.start_aruco_docking()
                        return

                # 12초 동안 8cm 이상 이동하지 못하면 Costmap에 갇힌 것으로 판단해 탈출 기동
                if self.last_progress_pose is None:
                    self.last_progress_pose = pose
                    self.last_progress_time = now
                else:
                    moved_dist = math.hypot(rx - self.last_progress_pose[0], ry - self.last_progress_pose[1])
                    if moved_dist > 0.08:
                        self.last_progress_pose = pose
                        self.last_progress_time = now
                    elif now - self.last_progress_time > 12.0:
                        self.trigger_narrow_escape(self.state)

        # 3. 좁은 공간 탈출 기동(ESCAPING): 1.2초 후진 -> 0.8초 약간 회전 후 복귀
        elif self.state == PatrolState.ESCAPING:
            elapsed = now - self.escape_start_time
            twist = Twist()
            if elapsed < 1.2:
                twist.linear.x = -0.07
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
            elif elapsed < 2.0:
                twist.linear.x = 0.0
                twist.angular.z = 0.35
                self.cmd_vel_pub.publish(twist)
            else:
                self.stop_robot_motors()
                self.clear_nav2_costmaps()
                next_st = self.escape_return_state
                self.state = next_st
                self.reset_stuck_monitor()
                if next_st == PatrolState.PATROL:
                    self.emitter.status_signal.emit('순찰 중 (PATROL)', '#2980b9')
                    self.emit_log('▶ 탈출 완료! 다음 웨이포인트로 순찰을 이어갑니다.')
                    self.advance_to_next_waypoint()
                elif next_st == PatrolState.RETURNING:
                    self.emitter.status_signal.emit('🏠 홈 좌표로 복귀 중...', '#d35400')
                    self.emit_log('🏠 탈출 완료! 홈 복귀 주행을 재시도합니다.')
                    self.send_home_goal()

        # 4. ArUco ID 66 정밀 근접 도킹(DOCKING) 제어
        elif self.state == PatrolState.DOCKING:
            if now - self.docking_start_time > 15.0:
                self.finish_stop_at_home('⚠️ 도킹 제한 시간(15초) 초과 - 현재 위치에서 정지합니다. (IDLE)')
                return

            twist = Twist()
            if now - self.last_home_marker_seen < 0.6:
                target_width = self.home_frame_width * self.target_docking_width_ratio

                if self.home_marker_width >= target_width:
                    self.finish_stop_at_home(
                        f'⏹ ArUco ID 66 앞 근접 도킹 완료! (마커 크기: {int(self.home_marker_width)}px) (IDLE)'
                    )
                    return

                center_x = self.home_frame_width / 2.0
                error_x = (self.home_marker_cx - center_x) / center_x

                twist.angular.z = float(np.clip(-error_x * 0.65, -0.45, 0.45))

                if abs(error_x) < 0.45:
                    twist.linear.x = 0.08
                else:
                    twist.linear.x = 0.02
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.28

            self.cmd_vel_pub.publish(twist)


class PatrolDashboard(QWidget):
    def __init__(self, task_manager, emitter):
        super().__init__()
        self.task_manager = task_manager
        self.video_thread = None
        self.stream_url = 'http://192.168.14.80:5000/video_feed'
        self.init_ui()

        emitter.log_signal.connect(self.append_log)
        emitter.hazard_log_signal.connect(self.append_hazard_log)
        emitter.status_signal.connect(self.update_status_label)

        self.ros_timer = QTimer(self)
        self.ros_timer.timeout.connect(self.spin_ros)
        self.ros_timer.start(30)

        self.map_timer = QTimer(self)
        self.map_timer.timeout.connect(self.refresh_minimap)
        self.map_timer.start(100)

    def spin_ros(self):
        rclpy.spin_once(self.task_manager, timeout_sec=0)

    def refresh_minimap(self):
        robot_2d = self.task_manager.get_robot_2d_with_yaw()
        self.minimap_widget.update_state(
            waypoints=self.task_manager.waypoints,
            robot_pose=robot_2d,
            home_pose=self.task_manager.home_pose,
            hazard_points=self.task_manager.hazard_locations
        )

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def init_ui(self):
        self.setWindowTitle('순찰 로봇 통합 제어기 + 좁은 공간 탈출 & 즉시정지/홈복귀 (ESC: 종료)')
        self.resize(1020, 860)

        main_layout = QHBoxLayout()
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)

        self.status_label = QLabel('현재 상태: 대기 중 (IDLE)')
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        self.status_label.setStyleSheet(
            "background-color: #2c3e50; color: white; padding: 15px; border-radius: 8px;"
        )
        left_panel.addWidget(self.status_label)

        bringup_layout = QHBoxLayout()
        btn_bringup_on = QPushButton('🚀 로봇 Bringup 켜기')
        btn_bringup_on.setStyleSheet("background-color: #8e44ad; color: white; font-size: 13px; padding: 10px; font-weight: bold;")
        btn_bringup_on.clicked.connect(self.task_manager.start_bringup)
        bringup_layout.addWidget(btn_bringup_on)

        btn_bringup_off = QPushButton('🔌 로봇 Bringup 끄기')
        btn_bringup_off.setStyleSheet("background-color: #34495e; color: white; font-size: 13px; padding: 10px; font-weight: bold;")
        btn_bringup_off.clicked.connect(self.task_manager.stop_bringup)
        bringup_layout.addWidget(btn_bringup_off)
        left_panel.addLayout(bringup_layout)

        nav_layout = QHBoxLayout()
        btn_nav_on = QPushButton('🗺️ Navigation 켜기')
        btn_nav_on.setStyleSheet("background-color: #16a085; color: white; font-size: 13px; padding: 10px; font-weight: bold;")
        btn_nav_on.clicked.connect(self.task_manager.start_navigation)
        nav_layout.addWidget(btn_nav_on)

        btn_nav_off = QPushButton('🛑 Navigation 끄기')
        btn_nav_off.setStyleSheet("background-color: #2c3e50; color: white; font-size: 13px; padding: 10px; font-weight: bold;")
        btn_nav_off.clicked.connect(self.task_manager.stop_navigation)
        nav_layout.addWidget(btn_nav_off)
        left_panel.addLayout(nav_layout)

        view_layout = QHBoxLayout()
        btn_view_on = QPushButton('🖥️ Nav2 View 켜기')
        btn_view_on.setStyleSheet("background-color: #2980b9; color: white; font-size: 13px; padding: 10px; font-weight: bold;")
        btn_view_on.clicked.connect(self.task_manager.start_nav_view)
        view_layout.addWidget(btn_view_on)

        btn_view_off = QPushButton('❎ Nav2 View 끄기')
        btn_view_off.setStyleSheet("background-color: #34495e; color: white; font-size: 13px; padding: 10px; font-weight: bold;")
        btn_view_off.clicked.connect(self.task_manager.stop_nav_view)
        view_layout.addWidget(btn_view_off)
        left_panel.addLayout(view_layout)

        cam_layout = QHBoxLayout()
        btn_cam_on = QPushButton('📷 카메라 + ArUco 켜기')
        btn_cam_on.setStyleSheet("background-color: #d35400; color: white; font-size: 13px; padding: 10px; font-weight: bold;")
        btn_cam_on.clicked.connect(self.on_cam_on_click)
        cam_layout.addWidget(btn_cam_on)

        btn_cam_off = QPushButton('📴 카메라 끄기')
        btn_cam_off.setStyleSheet("background-color: #34495e; color: white; font-size: 13px; padding: 10px; font-weight: bold;")
        btn_cam_off.clicked.connect(self.on_cam_off_click)
        cam_layout.addWidget(btn_cam_off)
        left_panel.addLayout(cam_layout)

        # 순찰 시작 & 즉시 정지 버튼 배치
        patrol_ctrl_layout = QHBoxLayout()
        btn_start = QPushButton('▶ 순찰 시작 (START)')
        btn_start.setStyleSheet("background-color: #27ae60; color: white; font-size: 14px; padding: 11px; font-weight: bold;")
        btn_start.clicked.connect(lambda: self.task_manager.handle_event('start'))
        patrol_ctrl_layout.addWidget(btn_start)

        btn_stop_here = QPushButton('⏹ 즉시 정지 (STOP)')
        btn_stop_here.setStyleSheet("background-color: #7f8c8d; color: white; font-size: 14px; padding: 11px; font-weight: bold;")
        btn_stop_here.clicked.connect(lambda: self.task_manager.handle_event('stop_here'))
        patrol_ctrl_layout.addWidget(btn_stop_here)
        left_panel.addLayout(patrol_ctrl_layout)

        # 비상 상황 & 상황 해제 버튼
        alert_ctrl_layout = QHBoxLayout()
        btn_alert = QPushButton('🚨 비상 상황 (ALERT)')
        btn_alert.setStyleSheet("background-color: #e74c3c; color: white; font-size: 14px; padding: 11px; font-weight: bold;")
        btn_alert.clicked.connect(lambda: self.task_manager.handle_event('intruder_detected'))
        alert_ctrl_layout.addWidget(btn_alert)

        btn_clear = QPushButton('✅ 상황 해제 (CLEAR)')
        btn_clear.setStyleSheet("background-color: #f39c12; color: white; font-size: 14px; padding: 11px; font-weight: bold;")
        btn_clear.clicked.connect(lambda: self.task_manager.handle_event('clear'))
        alert_ctrl_layout.addWidget(btn_clear)
        left_panel.addLayout(alert_ctrl_layout)

        # 홈 복귀 전용 버튼 + Costmap 수동 초기화 버튼
        home_ctrl_layout = QHBoxLayout()
        btn_return_home = QPushButton('🏠 홈 복귀 (ID 66 도킹)')
        btn_return_home.setStyleSheet("background-color: #d35400; color: white; font-size: 14px; padding: 11px; font-weight: bold;")
        btn_return_home.clicked.connect(lambda: self.task_manager.handle_event('return_home'))
        home_ctrl_layout.addWidget(btn_return_home)

        btn_clear_costmap = QPushButton('🧹 Costmap 초기화')
        btn_clear_costmap.setStyleSheet("background-color: #5d6d7e; color: white; font-size: 13px; padding: 11px; font-weight: bold;")
        btn_clear_costmap.clicked.connect(self.task_manager.clear_nav2_costmaps)
        home_ctrl_layout.addWidget(btn_clear_costmap)
        left_panel.addLayout(home_ctrl_layout)

        log_header_layout = QHBoxLayout()
        log_title = QLabel('📋 실시간 시스템 이벤트 로그')
        log_title.setFont(QFont('Arial', 11, QFont.Weight.Bold))
        log_header_layout.addWidget(log_title)

        btn_clear_log = QPushButton('로그 지우기')
        btn_clear_log.setFixedWidth(90)
        btn_clear_log.clicked.connect(self.clear_log)
        log_header_layout.addWidget(btn_clear_log)
        left_panel.addLayout(log_header_layout)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont('Consolas', 10))
        self.log_box.setStyleSheet("background-color: #1e1e1e; color: #00ff66; padding: 8px; border-radius: 6px;")
        left_panel.addWidget(self.log_box)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(8)

        cam_title = QLabel('📹 실시간 로봇 카메라 & ArUco 마커 인식')
        cam_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cam_title.setFont(QFont('Arial', 11, QFont.Weight.Bold))
        right_panel.addWidget(cam_title)

        self.camera_label = QLabel('카메라 대기 중...\n[📷 카메라 + ArUco 켜기] 버튼을 눌러주세요.')
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setFixedSize(480, 300)
        self.camera_label.setStyleSheet("background-color: #111111; color: #aaaaaa; border: 2px solid #333333; border-radius: 8px;")
        right_panel.addWidget(self.camera_label)

        map_title = QLabel('🗺️ 실시간 순찰 미니맵 (🟢로봇  🟡경로  🔵홈66  ❌위험물42)')
        map_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        map_title.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        right_panel.addWidget(map_title)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.minimap_widget = MiniMapWidget(
            pgm_path=os.path.join(base_dir, 'police_map5.pgm'),
            yaml_path=os.path.join(base_dir, 'police_map5.yaml')
        )
        right_panel.addWidget(self.minimap_widget)

        hazard_header_layout = QHBoxLayout()
        hazard_title = QLabel('🚨 위험물 발견(ID 42) & 순찰 로테이션 기록')
        hazard_title.setFont(QFont('Arial', 11, QFont.Weight.Bold))
        hazard_title.setStyleSheet("color: #c0392b;")
        hazard_header_layout.addWidget(hazard_title)

        btn_clear_hazard = QPushButton('기록 지우기')
        btn_clear_hazard.setFixedWidth(90)
        btn_clear_hazard.clicked.connect(self.clear_hazard_log)
        hazard_header_layout.addWidget(btn_clear_hazard)
        right_panel.addLayout(hazard_header_layout)

        self.hazard_log_box = QTextEdit()
        self.hazard_log_box.setReadOnly(True)
        self.hazard_log_box.setFont(QFont('Consolas', 10))
        self.hazard_log_box.setStyleSheet(
            "background-color: #1a1111; color: #ff6b6b; border: 1px solid #552222; padding: 8px; border-radius: 6px;"
        )
        self.hazard_log_box.setPlaceholderText("순찰 로테이션 현황 및 ID 42 위험물 발견 기록이 여기에 표시됩니다...")
        right_panel.addWidget(self.hazard_log_box)

        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 1)
        self.setLayout(main_layout)

    def on_cam_on_click(self):
        self.task_manager.start_camera()
        self.camera_label.setText('카메라 연결 중...')
        if self.video_thread is None or not self.video_thread.isRunning():
            self.video_thread = VideoStreamThread(self.stream_url)
            self.video_thread.frame_received.connect(self.update_camera_feed)
            self.video_thread.log_message.connect(self.task_manager.emit_log)
            self.video_thread.marker_detected.connect(self.task_manager.on_aruco_detected)
            self.video_thread.marker_tracking.connect(self.task_manager.on_aruco_tracking)
            self.video_thread.start()

    def on_cam_off_click(self):
        if self.video_thread is not None and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None
        self.task_manager.stop_camera()
        self.camera_label.clear()
        self.camera_label.setText('카메라 꺼짐')

    def update_camera_feed(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.camera_label.width(),
            self.camera_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.camera_label.setPixmap(pixmap)

    def update_status_label(self, status_text, color_hex):
        self.status_label.setText(f'현재 상태: {status_text}')
        self.status_label.setStyleSheet(f"background-color: {color_hex}; color: white; padding: 15px; border-radius: 8px;")

    def append_log(self, text):
        self.log_box.append(text)
        self.log_box.moveCursor(QTextCursor.MoveOperation.End)

    def clear_log(self):
        self.log_box.clear()

    def append_hazard_log(self, text):
        self.hazard_log_box.append(text)
        self.hazard_log_box.moveCursor(QTextCursor.MoveOperation.End)

    def clear_hazard_log(self):
        self.hazard_log_box.clear()

    def closeEvent(self, event):
        if self.video_thread is not None and self.video_thread.isRunning():
            self.video_thread.stop()
        super().closeEvent(event)


def main():
    rclpy.init()
    emitter = GuiEmitter()
    task_manager = PatrolTaskManager(emitter)

    app = QApplication(sys.argv)
    dashboard = PatrolDashboard(task_manager, emitter)
    dashboard.show()

    dashboard.append_log(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 대시보드 UI 로드 완료 (ESC: 종료)")

    try:
        sys.exit(app.exec())
    finally:
        task_manager.stop_robot_motors()
        task_manager.stop_lcd_script()
        task_manager.stop_camera()
        task_manager.stop_nav_view()
        task_manager.stop_navigation()
        task_manager.stop_bringup()
        task_manager.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
