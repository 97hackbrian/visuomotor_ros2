import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import time

class TestGripper(Node):
    def __init__(self):
        super().__init__('test_gripper')
        self.pub = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        self.timer = self.create_timer(1.0, self.timer_cb)
        self.state = 0.0

    def timer_cb(self):
        msg = Float64MultiArray()
        msg.data = [self.state]
        self.pub.publish(msg)
        self.get_logger().info(f'Published {self.state}')
        if self.state == 0.0:
            self.state = -0.01
        else:
            self.state = 0.0

def main():
    rclpy.init()
    node = TestGripper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()
