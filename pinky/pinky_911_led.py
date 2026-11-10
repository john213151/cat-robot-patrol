import rclpy
from rclpy.node import Node
from pinky_interfaces.srv import SetLed


class LedBlinkerNode(Node):

    def __init__(self):
        super().__init__('led_blinker_node')

        # /set_led 서비스 클라이언트 생성
        self.cli = self.create_client(SetLed, '/set_led')

        # 서비스 서버가 준비될 때까지 대기
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/set_led 서비스를 기다리는 중...')

        self.toggle = False

        # 0.3초마다 깜빡이도록 타이머 설정 (경광등 속도 조절 가능)
        self.timer = self.create_timer(0.3, self.timer_callback)
        self.get_logger().info('경광등 모드를 시작합니다!')

    def timer_callback(self):
        req = SetLed.Request()
        req.command = 'set_pixel'
        req.pixels = [0, 1, 2, 3, 4, 5, 6, 7]

        if self.toggle:
            # 원래 색상 (파란색)
            req.r = 0
            req.g = 0
            req.b = 255
        else:
            
            req.r = 255
            req.g = 0
            req.b = 0

        # 비동기 서비스 요청 전송
        self.cli.call_async(req)

        # 다음 주기를 위해 상태 반전
        self.toggle = not self.toggle


def main(args=None):
    rclpy.init(args=args)
    node = LedBlinkerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('경광등 모드를 종료합니다.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
