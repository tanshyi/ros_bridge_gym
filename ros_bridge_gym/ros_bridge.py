import numpy as np

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class BridgeNode(Node):

    def __init__(self, name='bridge'):
        super().__init__(name)

        self._state = None
        self._state_sub = self.create_subscription(
            msg_type = Float64MultiArray,
            topic = 'bridge_state',
            callback = self._bridge_state,
            qos_profile = 10
        )

    def _bridge_state(self, msg):
        self._state = np.float64(msg.data)

