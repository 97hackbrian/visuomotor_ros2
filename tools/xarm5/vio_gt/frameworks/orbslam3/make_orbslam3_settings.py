#!/usr/bin/env python3
"""Ajustes de ORB-SLAM3 (formato 1.0) para la D435i desde el calib.json de export_bag_frames.py.

Intrínsecas de fábrica de la cámara de color (640x480) y extrínseca cámara-IMU de fábrica (tf_static). Los ruidos del
IMU y los parámetros ORB son los del ejemplo oficial de ORB-SLAM3 para la D435i
(Examples/RGB-D-Inertial/RealSense_D435i.yaml). Un solo archivo sirve para RGB-D, RGB-D-inercial y monocular-inercial.
Camera.RGB = 0: el ejecutor lee las imágenes con cv::imread (BGR).

Uso: python3 make_orbslam3_settings.py <calib.json> <salida.yaml>
"""
import json
import sys

import numpy as np


def main():
    cal = json.load(open(sys.argv[1]))
    K = np.array(cal["K"]).reshape(3, 3)
    D = (list(cal["D"]) + [0.0] * 5)[:5]
    # IMU.T_b_c1: p_imu = T_b_c1 · p_cámara (cámara = óptico de color)
    T_b_c = np.linalg.inv(np.array(cal["T_opt_imu"]))
    data = ", ".join(f"{v:.9f}" for v in T_b_c.ravel())
    txt = f"""%YAML:1.0
File.version: "1.0"
Camera.type: "PinHole"
Camera1.fx: {K[0, 0]:.6f}
Camera1.fy: {K[1, 1]:.6f}
Camera1.cx: {K[0, 2]:.6f}
Camera1.cy: {K[1, 2]:.6f}
Camera1.k1: {D[0]:.9f}
Camera1.k2: {D[1]:.9f}
Camera1.p1: {D[2]:.9f}
Camera1.p2: {D[3]:.9f}
Camera1.k3: {D[4]:.9f}
Camera.width: {cal['width']}
Camera.height: {cal['height']}
Camera.fps: 30
Camera.RGB: 0
Stereo.ThDepth: 40.0
Stereo.b: 0.0745
RGBD.DepthMapFactor: {cal['depth_scale']:.1f}
IMU.T_b_c1: !!opencv-matrix
  rows: 4
  cols: 4
  dt: f
  data: [{data}]
IMU.InsertKFsWhenLost: 1
IMU.NoiseGyro: 1e-2
IMU.NoiseAcc: 1e-1
IMU.GyroWalk: 1e-6
IMU.AccWalk: 1e-4
IMU.Frequency: {round(cal['imu_rate_hz']):.1f}
ORBextractor.nFeatures: 1250
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -3.5
Viewer.ViewpointF: 500.0
"""
    open(sys.argv[2], "w").write(txt)
    print(f"-> {sys.argv[2]}")


if __name__ == "__main__":
    main()
