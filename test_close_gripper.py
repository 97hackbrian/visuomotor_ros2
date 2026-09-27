import rclpy
from std_msgs.msg import Float64MultiArray

def main():
    rclpy.init()
    node = rclpy.create_node('test_gripper')
    pub = node.create_publisher(Float64MultiArray, '/position_controller/commands', 1)
    
    import time
    time.sleep(1.0) # wait for publisher to establish connection
    
    msg = Float64MultiArray()
    msg.data = [-0.01]
    pub.publish(msg)
    print("Published [-0.01] to /position_controller/commands")
    
    time.sleep(1.0)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
