#!/usr/bin/env python3

import numpy as np
import rclpy
import transforms3d
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose
import time

def slerp(q1, q2, t):
    """Spherical linear interpolation for quaternions [x, y, z, w]."""
    dot = np.sum(q1 * q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        return q1 + t * (q2 - q1)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0 * q1) + (s1 * q2)

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _alpha(self, dt, cutoff):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t):
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x
            
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev

        # Estimate derivative
        dx = (x - self.x_prev) / dt
        
        # Filter derivative
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        
        # Calculate adaptive cutoff
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx_hat)
        
        # Filter value
        a = self._alpha(dt, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat

class PoseSmoother(Node):
    def __init__(self):
        super().__init__("pose_smoother")
        
        self.declare_parameter('min_cutoff', 1.5)
        self.declare_parameter('beta', 0.007)
        self.declare_parameter('d_cutoff', 1.0)
        
        # We need two filters: one for position (3D), one for orientation (we'll filter RPY internally or apply 1Euro to RPY)
        # Filtering quaternions directly with 1Euro is mathematically unstable because they are not a vector space.
        # We will convert to euler angles, filter them, and convert back.
        self.pos_filter = OneEuroFilter()
        self.rot_filter = OneEuroFilter()
        
        self.update_params()
        
        self.last_target_time = 0
        self.TIMEOUT = 0.5 # Reset filter if no target received for 0.5s

        self.sub = self.create_subscription(
            PoseStamped, "/target_frame_raw", self.target_callback, 1
        )
        self.pub = self.create_publisher(
            PoseStamped, "/target_frame", 1
        )
        
        # Timer to dynamically update parameters from the parameter server (so user can tune in live)
        self.create_timer(1.0, self.update_params)

    def update_params(self):
        min_cutoff = self.get_parameter('min_cutoff').value
        beta = self.get_parameter('beta').value
        d_cutoff = self.get_parameter('d_cutoff').value
        
        self.pos_filter.min_cutoff = min_cutoff
        self.pos_filter.beta = beta
        self.pos_filter.d_cutoff = d_cutoff
        
        # We apply slightly different beta to rotation if needed, but for now reuse
        self.rot_filter.min_cutoff = min_cutoff
        self.rot_filter.beta = beta
        self.rot_filter.d_cutoff = d_cutoff

    def target_callback(self, msg):
        t = time.time()
        
        if t - self.last_target_time > self.TIMEOUT:
            self.pos_filter.t_prev = None
            self.rot_filter.t_prev = None
            self.get_logger().info("Filtro 1Euro reseteado (Timeout)")
            
        self.last_target_time = t
        
        # Extract pos
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        # Extract rot (quat to euler to avoid quaternion wrapping issues in linear filters)
        q = [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]
        r, p, y = transforms3d.euler.quat2euler(q)
        rot = np.array([r, p, y])
        
        # Normalize euler angles before filtering to avoid jumps from pi to -pi?
        # Actually, if the model output jumps from pi to -pi, the difference dx will be 2pi.
        # We should unwrap it relative to the previous rot filter value.
        if self.rot_filter.x_prev is not None:
            rot = np.unwrap([self.rot_filter.x_prev, rot], axis=0)[1]
            
        # Filter
        pos_hat = self.pos_filter.filter(pos, t)
        rot_hat = self.rot_filter.filter(rot, t)
        
        # Convert back
        q_hat = transforms3d.euler.euler2quat(rot_hat[0], rot_hat[1], rot_hat[2])
        
        # Publish
        out_msg = PoseStamped()
        out_msg.header = msg.header
        out_msg.pose.position.x = float(pos_hat[0])
        out_msg.pose.position.y = float(pos_hat[1])
        out_msg.pose.position.z = float(pos_hat[2])
        out_msg.pose.orientation.w = float(q_hat[0])
        out_msg.pose.orientation.x = float(q_hat[1])
        out_msg.pose.orientation.y = float(q_hat[2])
        out_msg.pose.orientation.z = float(q_hat[3])
        
        self.pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PoseSmoother()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
