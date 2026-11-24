#!/usr/bin/env python3
"""Genera el tablero ChArUco de la celda como PDF (y PNG de vista previa) a escala 1:1, listo para imprimir.

Lee la geometría de ``visuomotor_bringup/config/realsense/charuco_board.yaml`` (fuente única) y la centra en una
hoja A4 o A3 apaisada. Añade una regla de 100 mm para comprobar la escala de la impresión. Después de imprimir hay
que medir con calibre y escribir las medidas reales en el YAML (``measured: true``).

El PDF se escribe directamente, sin matplotlib:
* el tablero va en 1 bit a ``--dpi`` (600 por defecto), sin pérdida (Flate), con cada fila rellenada al byte;
* la regla es vectorial;
* el texto usa Helvetica, una fuente estándar que no se incrusta.
El backend PDF de matplotlib 3.6 guardaba la imagen de 2 colores con el relleno de filas incorrecto: los visores
mostraban rayas diagonales en vez del tablero (corregido el 2026-11-24). El tamaño impreso no depende de la
resolución: la imagen se dibuja exactamente en ``squares × square_length`` milímetros.

Uso (contenedor o host con OpenCV):
    python3 tools/hardware/make_charuco_board.py                    # A3 (tablero por defecto) -> docs/hardware/
    python3 tools/hardware/make_charuco_board.py --paper A4 --out-dir /tmp/board
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "visuomotor_bringup"))

from visuomotor_bringup.charuco import DEFAULT_CONFIG, CharucoConfig, board_image, make_board  # noqa: E402

PAPERS_MM = {"A4": (297.0, 210.0), "A3": (420.0, 297.0)}           # apaisado (ancho, alto)
PT_PER_MM = 72.0 / 25.4


def _pdf_text(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, page_mm, board_u8: np.ndarray, board_rect_mm, ruler_mm, texts) -> None:
    """PDF de una página: ``board_u8`` (gris, 0 = negro) en ``board_rect_mm = (x, y, ancho, alto)`` con origen
    abajo a la izquierda; regla ``ruler_mm = (x, y, largo)``; ``texts = [(x_mm, y_mm, tamaño_pt, texto ASCII)]``."""
    pw, ph = (v * PT_PER_MM for v in page_mm)
    h, w = board_u8.shape
    bits = np.packbits(board_u8 > 127, axis=1)          # 1 = blanco en DeviceGray; cada fila completa sus bytes
    img = zlib.compress(bits.tobytes(), 9)
    bx, by, bw, bh = (v * PT_PER_MM for v in board_rect_mm)
    rx, ry, rl = (v * PT_PER_MM for v in ruler_mm)
    tick = 1.5 * PT_PER_MM
    ops = [f"q {bw:.4f} 0 0 {bh:.4f} {bx:.4f} {by:.4f} cm /Im0 Do Q",
           "0 g 0 G 1 w",
           f"{rx:.4f} {ry:.4f} m {rx + rl:.4f} {ry:.4f} l S",
           f"{rx:.4f} {ry - tick:.4f} m {rx:.4f} {ry + tick:.4f} l S",
           f"{rx + rl:.4f} {ry - tick:.4f} m {rx + rl:.4f} {ry + tick:.4f} l S",
           "0.32 0.32 0.31 rg"]
    for x, y, size, s in texts:
        ops.append(f"BT /F1 {size:g} Tf {x * PT_PER_MM:.4f} {y * PT_PER_MM:.4f} Td ({_pdf_text(s)}) Tj ET")
    content = "\n".join(ops).encode("ascii")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.4f} {ph:.4f}] /Resources << /XObject << /Im0 5 0 R >> "
         f"/Font << /F1 6 0 R >> /ProcSet [/PDF /Text /ImageB] >> /Contents 4 0 R >>").encode("ascii"),
        f"<< /Length {len(content)} >>\nstream\n".encode("ascii") + content + b"\nendstream",
        (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceGray /BitsPerComponent 1 "
         f"/Interpolate false /Filter /FlateDecode /Length {len(img)} >>\nstream\n").encode("ascii") + img
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("ascii")
    out += b"".join(f"{o:010d} 00000 n \n".encode("ascii") for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    path.write_bytes(bytes(out))


def write_png(path: Path, page_mm, board_u8: np.ndarray, board_rect_mm, ruler_mm, texts, dpi: int = 150) -> None:
    """Vista previa (no para imprimir): la misma página rasterizada a ``dpi``."""
    import cv2
    px_mm = dpi / 25.4
    pw, ph = (int(round(v * px_mm)) for v in page_mm)
    page = np.full((ph, pw), 255, np.uint8)
    x, y, w, h = board_rect_mm
    x0, y0 = int(round(x * px_mm)), int(round((page_mm[1] - y - h) * px_mm))
    small = cv2.resize(board_u8, (int(round(w * px_mm)), int(round(h * px_mm))), interpolation=cv2.INTER_AREA)
    page[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    rx, ry, rl = ruler_mm
    yy = int(round((page_mm[1] - ry) * px_mm))
    cv2.line(page, (int(round(rx * px_mm)), yy), (int(round((rx + rl) * px_mm)), yy), 0, 2)
    for xx in (rx, rx + rl):
        cv2.line(page, (int(round(xx * px_mm)), yy - 9), (int(round(xx * px_mm)), yy + 9), 0, 2)
    for tx, ty, size, s in texts:
        cv2.putText(page, s, (int(round(tx * px_mm)), int(round((page_mm[1] - ty) * px_mm))),
                    cv2.FONT_HERSHEY_SIMPLEX, size / 22.0, 80, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), page)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--paper", choices=sorted(PAPERS_MM), default="A3")
    ap.add_argument("--dpi", type=int, default=600, help="resolución del tablero dentro del PDF")
    ap.add_argument("--out-dir", default=str(REPO / "docs" / "hardware"))
    args = ap.parse_args(argv)

    cfg = CharucoConfig.load(args.config)
    board = make_board(cfg)
    w_mm, h_mm = (1000.0 * v for v in cfg.size_m)
    pw, ph = PAPERS_MM[args.paper]
    if round(w_mm, 3) > pw - 20.0 or round(h_mm, 3) > ph - 35.0:
        print(f"El tablero ({w_mm:.0f} x {h_mm:.0f} mm) no cabe en {args.paper} con márgenes: usar --paper A3 o "
              f"reducir square_length_m.", file=sys.stderr)
        return 2
    px = (int(round(w_mm / 25.4 * args.dpi)), int(round(h_mm / 25.4 * args.dpi)))
    img = board_image(board, px, margin_px=0)
    left, bottom = (pw - w_mm) / 2.0, (ph - h_mm) / 2.0 + 8.0
    rect = (left, bottom, w_mm, h_mm)
    ruler = (left, bottom - 10.0, 100.0)
    texts = [(left + 102.0, bottom - 11.2, 8, "100 mm  (print at 100 %, actual size, no fit-to-page)"),
             (left, bottom - 18.0, 7.5,
              f"ChArUco {cfg.squares_x}x{cfg.squares_y}, {cfg.dictionary}, square {cfg.square_length_m * 1000:.1f} mm, "
              f"marker {cfg.marker_length_m * 1000:.1f} mm (nominal). Measure with calipers and write the measured "
              f"values in charuco_board.yaml (measured: true).")]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"charuco_board_{cfg.squares_x}x{cfg.squares_y}_{cfg.square_length_m * 1000:.0f}mm_{args.paper}"
    write_pdf(out / f"{stem}.pdf", (pw, ph), img, rect, ruler, texts)
    write_png(out / f"{stem}.png", (pw, ph), img, rect, ruler, texts)
    print(f"Tablero {cfg.squares_x}x{cfg.squares_y} ({w_mm:.0f} x {h_mm:.0f} mm) en {args.paper}: "
          f"{out / (stem + '.pdf')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
