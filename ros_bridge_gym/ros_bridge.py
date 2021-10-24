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

        self._action = None
        self._action_pub = self.create_publisher(
            msg_type = Float64MultiArray,
            topic = 'bridge_action',
            qos_profile = 10
        )


    def _bridge_state(self, msg):
        self._state = np.float64(msg.data)

    @property
    def state(self):
        return self._state


    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, action):
        msg = Float64MultiArray()
        msg.data = action
        self._action_pub.publish(msg)
        self._action = np.float64(action)


