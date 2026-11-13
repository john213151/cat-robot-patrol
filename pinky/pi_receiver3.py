import os
import time
import signal
import threading
import subprocess
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    from pinkylib import Buzzer
    HAS_BUZZER = True
except ImportError:
    HAS_BUZZER = False


class PiReceiverNode(Node):
    def __init__(self):
        super().__init__('pi_receiver_node')
        self.lcd_process = None
        self.bringup_process = None
        self.led_server_process = None
        self.nav_process = None
        self.cam_process = None

        self.buzzer = None
        self.buzzer_lock = threading.Lock()
        self.siren_running = False
        self.init_buzzer()

        # 경로 설정 (최신 police_map4.yaml 반영 완료)
        self.lcd_dir = '/home/pinky/pinky_lcd/example'
        self.lcd_script = 'catpopo.py'
        self.cam_dir = '/home/pinky/camera_stream'
        self.cam_script = 'app.py'
        self.map_path = '/home/pinky/police_map4.yaml'

        # 토픽 구독자 등록
        self.lcd_sub = self.create_subscription(String, '/lcd_cmd', self.lcd_callback, 10)
        self.bringup_sub = self.create_subscription(String, '/bringup_cmd', self.bringup_callback, 10)
        self.nav_sub = self.create_subscription(String, '/nav_cmd', self.nav_callback, 10)
        self.cam_sub = self.create_subscription(String, '/cam_cmd', self.cam_callback, 10)
        self.buzzer_sub = self.create_subscription(String, '/buzzer_cmd', self.buzzer_callback, 10)

        self.get_logger().info('라즈베리파이 통합 수신기(LCD + Bringup + LED + Nav2 + Camera + Buzzer) 대기 중...')

    # --- 부저(Buzzer) 초기화 및 경고음/사이렌 제어 ---
    def init_buzzer(self):
        if not HAS_BUZZER:
            self.get_logger().warn('pinkylib.Buzzer 모듈을 찾을 수 없어 부저 기능을 비활성화합니다.')
            return
        try:
            self.buzzer = Buzzer()
            self.buzzer.buzzer_start()
            self.buzzer.set_buzzer_freq(1000)
            self.buzzer.set_buzzer_duty(0)
            self.get_logger().info('pinkylib Buzzer 초기화 완료!')
        except Exception as e:
            self.get_logger().error(f'Buzzer 초기화 실패: {e}')
            self.buzzer = None

    def buzzer_callback(self, msg):
        cmd = msg.data.strip()
        if cmd == 'siren_on':
            self.start_siren()
        elif cmd == 'siren_off':
            self.stop_siren()
        elif cmd == 'hazard':
            threading.Thread(target=self.play_hazard_alarm, daemon=True).start()

    def start_siren(self):
        if self.buzzer is None:
            return
        if not self.siren_running:
            self.siren_running = True
            threading.Thread(target=self._siren_loop, daemon=True).start()
            self.get_logger().info('🚨 비상 경광등 사이렌(삐용삐용) 연속 재생 시작!')

    def stop_siren(self):
        if self.siren_running:
            self.siren_running = False
            with self.buzzer_lock:
                if self.buzzer is not None:
                    try:
                        self.buzzer.set_buzzer_duty(0)
                    except Exception:
                        pass
            self.get_logger().info('🔕 비상 경광등 사이렌 정지!')

    def _siren_loop(self):
        """비상 상황 해제(CLEAR) 또는 정지 전까지 삐용삐용 연속 발생"""
        pattern = [
            (1046, 0.28),  # 삐~
            (784, 0.28),   # 용~
        ]
        while self.siren_running and self.buzzer is not None:
            for freq, duration in pattern:
                if not self.siren_running:
                    break
                with self.buzzer_lock:
                    try:
                        self.buzzer.set_buzzer_freq(freq)
                        self.buzzer.set_buzzer_duty(60)
                    except Exception:
                        pass
                time.sleep(duration)
        with self.buzzer_lock:
            if self.buzzer is not None:
                try:
                    self.buzzer.set_buzzer_duty(0)
                except Exception:
                    pass

    def play_hazard_alarm(self):
        """위험물(ID 42) 인식 시 삐용삐용과 명확히 구분되는 단속 경고 알람 (띡-띡-띡 3연음)"""
        if self.buzzer is None or self.siren_running:
            return
        with self.buzzer_lock:
            try:
                self.get_logger().info('⚠️ 위험물 발견 전용 알람음(띡-띡-띡) 재생!')
                for _ in range(3):
                    self.buzzer.set_buzzer_freq(1568)  # 높은 경고음
                    self.buzzer.set_buzzer_duty(65)
                    time.sleep(0.12)
                    self.buzzer.set_buzzer_duty(0)
                    time.sleep(0.08)
            except Exception as e:
                self.get_logger().error(f'부저 재생 오류: {e}')
                try:
                    self.buzzer.set_buzzer_duty(0)
                except Exception:
                    pass

    def close_buzzer(self):
        self.siren_running = False
        if self.buzzer is not None:
            try:
                self.buzzer.set_buzzer_duty(0)
                self.buzzer.close()
                self.buzzer = None
            except Exception:
                pass

    # --- 카메라 제어 로직 (~/camera_stream/app.py) ---
    def cam_callback(self, msg):
        if msg.data == 'on':
            self.start_cam()
        elif msg.data == 'off':
            self.stop_cam()

    def start_cam(self):
        if self.cam_process is None or self.cam_process.poll() is not None:
            try:
                self.cam_process = subprocess.Popen(
                    ['python3', self.cam_script],
                    cwd=self.cam_dir,
                    preexec_fn=os.setsid
                )
                self.get_logger().info('camera_stream/app.py 실행됨!')
            except Exception as e:
                self.get_logger().error(f'카메라 실행 실패: {e}')

    def stop_cam(self):
        if self.cam_process is not None and self.cam_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.cam_process.pid), signal.SIGINT)
                self.cam_process = None
                self.get_logger().info('camera_stream/app.py 종료됨!')
            except Exception as e:
                self.get_logger().error(f'카메라 종료 실패: {e}')

    # --- LCD 제어 로직 ---
    def lcd_callback(self, msg):
        if msg.data == 'on':
            self.start_lcd()
        elif msg.data == 'off':
            self.stop_lcd()

    def start_lcd(self):
        if self.lcd_process is None or self.lcd_process.poll() is not None:
            try:
                self.lcd_process = subprocess.Popen(
                    ['python3', self.lcd_script],
                    cwd=self.lcd_dir,
                    preexec_fn=os.setsid
                )
                self.get_logger().info('catpopo.py LCD 실행됨!')
            except Exception as e:
                self.get_logger().error(f'LCD 실행 실패: {e}')

    def stop_lcd(self):
        if self.lcd_process is not None and self.lcd_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.lcd_process.pid), signal.SIGTERM)
                self.lcd_process = None
                self.get_logger().info('catpopo.py LCD 종료됨!')
            except Exception as e:
                self.get_logger().error(f'LCD 종료 실패: {e}')

    # --- Bringup + LED Server 제어 로직 ---
    def bringup_callback(self, msg):
        if msg.data == 'on':
            self.start_bringup()
        elif msg.data == 'off':
            self.stop_bringup()

    def start_bringup(self):
        if self.bringup_process is None or self.bringup_process.poll() is not None:
            try:
                cmd_bringup = 'ros2 launch pinky_bringup bringup_robot.launch.xml'
                self.bringup_process = subprocess.Popen(
                    cmd_bringup,
                    shell=True,
                    executable='/bin/bash',
                    preexec_fn=os.setsid
                )
                self.get_logger().info('pinky_bringup 실행됨!')
            except Exception as e:
                self.get_logger().error(f'Bringup 실행 실패: {e}')

        if self.led_server_process is None or self.led_server_process.poll() is not None:
            try:
                cmd_led = 'ros2 run pinky_led led_server'
                self.led_server_process = subprocess.Popen(
                    cmd_led,
                    shell=True,
                    executable='/bin/bash',
                    preexec_fn=os.setsid
                )
                self.get_logger().info('pinky_led led_server 실행됨!')
            except Exception as e:
                self.get_logger().error(f'LED Server 실행 실패: {e}')

    def stop_bringup(self):
        if self.bringup_process is not None and self.bringup_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.bringup_process.pid), signal.SIGINT)
                self.bringup_process = None
                self.get_logger().info('pinky_bringup 종료됨!')
            except Exception as e:
                self.get_logger().error(f'Bringup 종료 실패: {e}')

        if self.led_server_process is not None and self.led_server_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.led_server_process.pid), signal.SIGINT)
                self.led_server_process = None
                self.get_logger().info('pinky_led led_server 종료됨!')
            except Exception as e:
                self.get_logger().error(f'LED Server 종료 실패: {e}')

    # --- Navigation 제어 로직 ---
    def nav_callback(self, msg):
        if msg.data == 'on':
            self.start_nav()
        elif msg.data == 'off':
            self.stop_nav()

    def start_nav(self):
        if self.nav_process is None or self.nav_process.poll() is not None:
            try:
                cmd_nav = f'ros2 launch pinky_navigation bringup_launch.xml map:={self.map_path}'
                self.nav_process = subprocess.Popen(
                    cmd_nav,
                    shell=True,
                    executable='/bin/bash',
                    preexec_fn=os.setsid
                )
                self.get_logger().info(f'Navigation 실행됨! (맵: {self.map_path})')
            except Exception as e:
                self.get_logger().error(f'Navigation 실행 실패: {e}')

    def stop_nav(self):
        if self.nav_process is not None and self.nav_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.nav_process.pid), signal.SIGINT)
                self.nav_process = None
                self.get_logger().info('Navigation 종료됨!')
            except Exception as e:
                self.get_logger().error(f'Navigation 종료 실패: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = PiReceiverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_buzzer()
        node.stop_lcd()
        node.stop_cam()
        node.stop_nav()
        node.stop_bringup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
