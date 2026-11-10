import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from pinky_interfaces.srv import SetLed
from enum import Enum

class PatrolState(Enum):
    IDLE = 1
    PATROL = 2
    ALERT = 3

class PatrolTaskManager(Node):
    def __init__(self):
        super().__init__('patrol_task_manager')
        self.state = PatrolState.IDLE
        self.toggle = False
        self.goal_handle = None

        # 1. 순찰 웨이포인트 좌표 설정 (x, y 좌표 예시)
        self.waypoints = [
            (1.0, 0.0),
            (2.0, 1.5),
            (0.0, 1.5),
            (0.0, 0.0)
        ]
        self.current_wp_idx = 0

        # 2. Nav2 이동 액션 클라이언트 & LED 서비스 클라이언트
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.led_client = self.create_client(SetLed, '/set_led')

        # 3. 외부 명령 및 센서 이벤트 구독
        # 예: 'start', 'stop', 'intruder_detected', 'clear'
        self.event_sub = self.create_subscription(
            String, '/patrol_event', self.event_callback, 10)

        # 0.25초 주기 제어 루프 (LED 점멸 및 상태 감시)
        self.timer = self.create_timer(0.25, self.control_loop)
        self.get_logger().info('순찰 Task Manager 준비 완료 (현재 상태: IDLE)')

    def set_led(self, pixels, r, g, b):
        if not self.led_client.service_is_ready():
            return
        req = SetLed.Request()
        req.command = 'set_pixel'
        req.pixels = pixels
        req.r, req.g, req.b = r, g, b
        self.led_client.call_async(req)

    def event_callback(self, msg):
        event = msg.data

        if event == 'start' and self.state == PatrolState.IDLE:
            self.state = PatrolState.PATROL
            self.set_led([0, 1, 2, 3, 4, 5, 6, 7], 0, 0, 255) # 순찰 중 파란불
            self.get_logger().info('순찰을 시작합니다!')
            self.send_next_waypoint()

        elif event == 'intruder_detected' and self.state == PatrolState.PATROL:
            self.state = PatrolState.ALERT
            self.get_logger().warn('침입자 감지! 주행을 중지하고 경광등을 가동합니다.')
            # 진행 중인 Nav2 주행 즉시 취소
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()

        elif event == 'clear' and self.state == PatrolState.ALERT:
            self.state = PatrolState.PATROL
            self.set_led([0, 1, 2, 3, 4, 5, 6, 7], 0, 0, 255)
            self.get_logger().info('상황 해제. 순찰을 재개합니다.')
            self.send_next_waypoint()

        elif event == 'stop':
            self.state = PatrolState.IDLE
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
            self.set_led([0, 1, 2, 3, 4, 5, 6, 7], 0, 255, 0) # 대기 중 초록불
            self.get_logger().info('순찰을 종료하고 대기 상태로 전환합니다.')

    def send_next_waypoint(self):
        if self.state != PatrolState.PATROL:
            return

        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('Nav2 액션 서버가 실행되지 않았습니다.')
            return

        x, y = self.waypoints[self.current_wp_idx]
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.orientation.w = 1.0

        self.get_logger().info(f'웨이포인트 [{self.current_wp_idx}] ({x}, {y})로 이동 중...')
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().warn('목표 지점 이동이 거부되었습니다.')
            return

        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        # 순찰 상태가 유지되고 있을 때만 다음 지점으로 무한 반복 이동
        if self.state == PatrolState.PATROL:
            self.current_wp_idx = (self.current_wp_idx + 1) % len(self.waypoints)
            self.send_next_waypoint()

    def control_loop(self):
        # ALERT 상태일 때만 좌우 교차 경광등 작동
        if self.state == PatrolState.ALERT:
            if self.toggle:
                self.set_led([0, 1, 2, 3], 255, 0, 0)
                self.set_led([4, 5, 6, 7], 0, 0, 255)
            else:
                self.set_led([0, 1, 2, 3], 0, 0, 255)
                self.set_led([4, 5, 6, 7], 255, 0, 0)
            self.toggle = not self.toggle

def main(args=None):
    rclpy.init(args=args)
    node = PatrolTaskManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
