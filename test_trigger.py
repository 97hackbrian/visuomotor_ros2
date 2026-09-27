import rclpy
from rclpy.node import Node
from visuomotor_msgs.srv import EpisodeTrigger

def main():
    rclpy.init()
    node = rclpy.create_node('test_trigger')
    cli = node.create_client(EpisodeTrigger, '/pickup_approximate_node/trigger_episode')
    while not cli.wait_for_service(timeout_sec=1.0):
        print('service not available, waiting again...')
    req = EpisodeTrigger.Request()
    req.command = '3'
    future = cli.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    print("Result:", future.result().message)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
