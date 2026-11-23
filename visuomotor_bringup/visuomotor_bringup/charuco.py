"""Tablero ChArUco fijo de la celda: configuración, construcción, imagen y pose (OpenCV 4.5 a 4.10).

El tablero es la referencia física de la herramienta de demostraciones con la D435i:
* calibración cámara-brida con el xArm5 (``tools/xarm5/calibrate_hand_eye.py``);
* referencia para medir la odometría a mano;
* guía para empezar cada demostración en la vista de HOME3.

La configuración vive en ``visuomotor_bringup/config/realsense/charuco_board.yaml``. Las medidas impresas deben
medirse con calibre (``measured: true``): las herramientas de medición se niegan a usar medidas nominales.

Se exige un número IMPAR de filas: OpenCV 4.6 cambió el patrón de los tableros con filas pares ("legacy pattern"), y
un tablero impar se ve igual en todas las versiones. Así sirve igual en el portátil y en la Jetson.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "realsense" / "charuco_board.yaml"


@dataclass
class CharucoConfig:
    dictionary: str
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    measured: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "CharucoConfig":
        cfg = cls(dictionary=str(d["dictionary"]), squares_x=int(d["squares_x"]), squares_y=int(d["squares_y"]),
                  square_length_m=float(d["square_length_m"]), marker_length_m=float(d["marker_length_m"]),
                  measured=bool(d.get("measured", False)))
        if cfg.squares_y % 2 == 0:
            raise ValueError("charuco_board.squares_y debe ser impar (patrón idéntico en todas las versiones de OpenCV)")
        if not 0.0 < cfg.marker_length_m < cfg.square_length_m:
            raise ValueError("charuco_board.marker_length_m debe ser positivo y menor que square_length_m")
        return cfg

    @classmethod
    def load(cls, path=DEFAULT_CONFIG) -> "CharucoConfig":
        import yaml
        return cls.from_dict(yaml.safe_load(open(path))["charuco_board"])

    @property
    def size_m(self) -> Tuple[float, float]:
        return self.squares_x * self.square_length_m, self.squares_y * self.square_length_m


@dataclass
class BoardPose:
    """Pose del tablero en el marco óptico de la cámara: ``p_cam = R @ p_board + t``."""
    rotation: np.ndarray
    translation: np.ndarray
    n_corners: int
    reprojection_error_px: float

    def camera_in_board(self) -> Tuple[np.ndarray, np.ndarray]:
        """Pose de la cámara en el marco del tablero: ``(R^T, -R^T t)``."""
        return self.rotation.T, -self.rotation.T @ self.translation


def _dictionary(cfg: CharucoConfig):
    import cv2
    if not hasattr(cv2.aruco, cfg.dictionary):
        raise ValueError(f"diccionario ArUco desconocido: {cfg.dictionary}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.dictionary))


def make_board(cfg: CharucoConfig):
    import cv2
    d = _dictionary(cfg)
    if hasattr(cv2.aruco, "CharucoBoard_create"):                       # OpenCV <= 4.6
        return cv2.aruco.CharucoBoard_create(cfg.squares_x, cfg.squares_y, cfg.square_length_m,
                                             cfg.marker_length_m, d)
    return cv2.aruco.CharucoBoard((cfg.squares_x, cfg.squares_y), cfg.square_length_m, cfg.marker_length_m, d)


def board_image(board, size_px: Tuple[int, int], margin_px: int = 0) -> np.ndarray:
    """Imagen en escala de grises del tablero (``size_px`` = ancho, alto)."""
    if hasattr(board, "generateImage"):
        return board.generateImage(size_px, marginSize=margin_px, borderBits=1)
    return board.draw(size_px, marginSize=margin_px, borderBits=1)


def chessboard_corners(board) -> np.ndarray:
    """Esquinas interiores del tablero en su marco (m), ``[N, 3]``."""
    c = board.getChessboardCorners() if hasattr(board, "getChessboardCorners") else board.chessboardCorners
    return np.asarray(c, dtype=np.float64).reshape(-1, 3)


def detect_corners(gray: np.ndarray, board, cfg: CharucoConfig, camera_matrix=None, dist_coeffs=None):
    """Esquinas ChArUco interpoladas: ``(pixeles [N, 2], ids [N])`` o ``(None, None)``."""
    import cv2
    d = _dictionary(cfg)
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=cv2.aruco.DetectorParameters_create())
    if ids is None or len(ids) == 0:
        return None, None
    if hasattr(cv2.aruco, "interpolateCornersCharuco"):
        n, ch, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board, cameraMatrix=camera_matrix,
                                                            distCoeffs=dist_coeffs)
        if not n:
            return None, None
    else:                                                                # OpenCV >= 4.7
        ch, ch_ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray, markerCorners=corners, markerIds=ids)
    if ch_ids is None or len(ch_ids) == 0:
        return None, None
    return np.asarray(ch, dtype=np.float64).reshape(-1, 2), np.asarray(ch_ids).ravel().astype(int)


def estimate_pose(gray: np.ndarray, board, cfg: CharucoConfig, camera_matrix: np.ndarray,
                  dist_coeffs: Optional[np.ndarray] = None, min_corners: int = 6) -> Optional[BoardPose]:
    """Pose del tablero (PnP sobre las esquinas ChArUco) o None si hay menos de ``min_corners`` esquinas o los puntos
    son colineales."""
    import cv2
    k = np.asarray(camera_matrix, dtype=np.float64)
    dist = np.zeros(5) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)
    px, ids = detect_corners(gray, board, cfg, k, dist)
    if ids is None or len(ids) < min_corners:
        return None
    obj = chessboard_corners(board)[ids]
    if np.linalg.matrix_rank(obj[:, :2] - obj[:, :2].mean(0)) < 2:        # todas en una fila: pose indeterminada
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, px, k, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - px, axis=1).mean())
    r, _ = cv2.Rodrigues(rvec)
    return BoardPose(rotation=r, translation=tvec.ravel(), n_corners=int(len(ids)), reprojection_error_px=err)
