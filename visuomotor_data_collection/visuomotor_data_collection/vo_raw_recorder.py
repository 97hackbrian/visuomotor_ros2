"""Grabación cruda por episodio (rosbag2, MCAP) para reprocesar la odometría offline.

Con los datos crudos de cada demostración (imagen, profundidad, IMU, camera_info, odometría) se puede recalcular la
trayectoria con otro estimador o parámetros (Fase 3 del plan) sin repetir la toma. Antes, cada toma era definitiva.

Los tópicos se leen de YAML. Su tipo se resuelve en el grafo al empezar el episodio; los tópicos que no existen se
avisan y se omiten. Se suscribe en modo serializado (``raw=True``): no deserializa nada, así que el coste es copiar
bytes.

Los tópicos latched (``/tf_static``) se publican una sola vez al arrancar la cámara: con la QoS de sensores (volátil)
el bag no los recibía y el reprocesado offline quedaba sin la transformación cámara-óptica. Se graban con
TRANSIENT_LOCAL y fiable (corregido el 2026-11-24).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional

LATCHED_TOPICS = ("/tf_static",)


class RawEpisodeBag:
    def __init__(self, node, topics: List[str], storage_id: str = "mcap"):
        self.node = node
        self.topics = list(topics)
        self.storage_id = storage_id
        self.writer = None
        self.path: Optional[Path] = None
        self._subs = []
        self.counts: Dict[str, int] = {}

    @property
    def active(self) -> bool:
        return self.writer is not None

    def start(self, path: Path) -> List[str]:
        """Abre el bag y se suscribe. Devuelve los tópicos grabados."""
        import rosbag2_py
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                               qos_profile_sensor_data)
        from rosidl_runtime_py.utilities import get_message
        latched = QoSProfile(depth=100, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.stop(delete=False)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        types = {name: t[0] for name, t in self.node.get_topic_names_and_types() if t}
        writer = rosbag2_py.SequentialWriter()
        writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=self.storage_id),
                    rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
        recorded = []
        for i, topic in enumerate(self.topics):
            if topic not in types:
                self.node.get_logger().warn(f"[RAW BAG] El tópico {topic} no existe: no se graba.")
                continue
            try:
                meta = rosbag2_py.TopicMetadata(id=i, name=topic, type=types[topic], serialization_format="cdr")
            except TypeError:                                   # versiones de rosbag2 sin 'id'
                meta = rosbag2_py.TopicMetadata(name=topic, type=types[topic], serialization_format="cdr")
            writer.create_topic(meta)
            self.counts[topic] = 0
            qos = latched if topic in LATCHED_TOPICS else qos_profile_sensor_data
            self._subs.append(self.node.create_subscription(
                get_message(types[topic]), topic, lambda data, tp=topic: self._write(tp, data), qos, raw=True))
            recorded.append(topic)
        self.writer, self.path = writer, path
        return recorded

    def _write(self, topic: str, data: bytes) -> None:
        if self.writer is None:
            return
        self.writer.write(topic, data, self.node.get_clock().now().nanoseconds)
        self.counts[topic] = self.counts.get(topic, 0) + 1

    def stop(self, delete: bool) -> Optional[Path]:
        """Cierra el bag; con ``delete`` lo borra (episodio descartado). Devuelve la ruta si se conservó."""
        for s in self._subs:
            self.node.destroy_subscription(s)
        self._subs = []
        path, self.path = self.path, None
        if self.writer is not None:
            del self.writer                                     # cierra el archivo
            self.writer = None
        if path is not None and delete:
            shutil.rmtree(path, ignore_errors=True)
            return None
        return path
