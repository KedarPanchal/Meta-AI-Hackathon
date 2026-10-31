"""
fly_drone: the ROS 2 side -- the node, the detector and the vehicle bridges.

The brain itself is in :mod:`fly_brain`, which has no rclpy dependency, so the controller can
be calibrated and tested without a simulator running.
"""
