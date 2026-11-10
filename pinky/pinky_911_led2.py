import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from pinky_interfaces.srv import SetLed

class PoliceSirenNode(Node):

    def __init__(self):
        super().__init__('police_siren_node')
        self.cli = self.create_client(SetLed, '/set_led')
        self.emergency_pub = self.create_publisher(Bool, '/emergency', 10)

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/set_led 서비스를 기다리는 중...')

        self.toggle = False
        self.timer = self.create_timer(0.25, self.timer_callback)
        self.get_logger().info('긴급 상황 발생! (LED + LCD 신호 전송)')

        # 시작 시 라즈베리파이에 LCD 켜기 신호(True) 전송
        self.publish_emergency(True)

    def publish_emergency(self, state: bool):
        msg = Bool()
        msg.data = state
        self.emergency_pub.publish(msg)

    def send_led_request(self, pixels, r, g, b):
        req = SetLed.Request()
        req.command = 'set_pixel'
        req.pixels = pixels
        req.r = r
        req.g = g
        req.b = b
        self.cli.call_async(req)

    def timer_callback(self):
        left_pixels = [0, 1, 2, 3]
        right_pixels = [4, 5, 6, 7]

        if self.toggle:
            self.send_led_request(left_pixels, 255, 0, 0)
            self.send_led_request(right_pixels, 0, 0, 255)
        else:
            self.send_led_request(left_pixels, 0, 0, 255)
            self.send_led_request(right_pixels, 255, 0, 0)

        self.toggle = not self.toggle


def main(args=None):
    rclpy.init(args=args)
    node = PoliceSirenNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 시 LED 끄기 및 라즈베리파이에 LCD 끄기 신호(False) 전송
        node.send_led_request([0, 1, 2, 3, 4, 5, 6, 7], 0, 0, 0)
        node.publish_emergency(False)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
