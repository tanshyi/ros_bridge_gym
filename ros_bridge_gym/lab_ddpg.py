import rclpy

from .ros_bridge import BridgeNode


class LabDDPG(BridgeNode):

    def __init__(self):
        super().__init__(name='lab_ddpg')
        self._timer = self.create_timer(1, self._timer_callback)

    def _timer_callback(self):
        self.get_logger().info(str(self._state))


def main(args=None):
    rclpy.init(args=args)

    node = LabDDPG()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
