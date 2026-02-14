#!/usr/bin/env python3
# TAG: YOLO_PI_LUXONIS
"""
YOLO130226_HIBRIDO_PRO
----------------------
Pipeline profesional híbrido para Raspberry Pi + Luxonis OAK:
- Luxonis: RGB + profundidad (sin NN on-device para máxima estabilidad).
- Host (Raspberry): inferencia YOLOv4-tiny con OpenCV DNN.

Objetivo: evitar congelamientos típicos del modo on-device spatial NN y mantener
etiquetas + profundidad de forma robusta.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import cv2
import depthai as dai
import numpy as np

cv2.setUseOptimized(True)
cv2.setNumThreads(2)

# ----------------------------
# Configuración de rutas
# ----------------------------
BASE_DIR = Path('/home/machina/Desktop/computer vision/yolo_pi_luxonis')
STOP_FILE = BASE_DIR / 'STOP_YOLO_PI_LUXONIS.flag'
LOG_FILE = BASE_DIR / 'YOLO_PI_LUXONIS_runtime.log'

MODELS_DIR = Path('/home/machina/.openclaw/workspace/coral/models')
CFG = MODELS_DIR / 'yolov4-tiny.cfg'
WEIGHTS = MODELS_DIR / 'yolov4-tiny.weights'
NAMES = MODELS_DIR / 'coco.names'

# ----------------------------
# Parámetros ajustables
# ----------------------------
RGB_PREVIEW_SIZE = (640, 360)
RGB_FPS = 15
DEPTH_OUT_SIZE = (640, 352)
CONF_TH = 0.35
NMS_TH = 0.40

# Botones en overlay (x1,y1,x2,y2)
# Se calculan dinámicamente en la esquina inferior izquierda para no tapar etiquetas
BTN_STOP = (0, 0, 0, 0)
BTN_EXIT = (0, 0, 0, 0)


def log(msg: str) -> None:
    """Escribe log a consola y archivo."""
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {msg}'
    print(line, flush=True)
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def load_detector() -> Tuple[cv2.dnn_DetectionModel, list[str]]:
    """Carga YOLOv4-tiny en OpenCV DNN."""
    if not CFG.exists() or not WEIGHTS.exists() or not NAMES.exists():
        raise FileNotFoundError('Faltan archivos del modelo en coral/models (cfg/weights/names).')

    with NAMES.open('r', encoding='utf-8') as f:
        classes = [x.strip() for x in f if x.strip()]

    net = cv2.dnn.readNetFromDarknet(str(CFG), str(WEIGHTS))
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255.0, swapRB=True)
    return model, classes


def build_pipeline() -> dai.Pipeline:
    """Construye pipeline DepthAI solo para RGB + depth (sin NN on-device)."""
    p = dai.Pipeline()

    # Cámara RGB principal (API estable depthai 2.x)
    cam_rgb = p.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setPreviewSize(*RGB_PREVIEW_SIZE)
    cam_rgb.setFps(RGB_FPS)

    xout_rgb = p.create(dai.node.XLinkOut)
    xout_rgb.setStreamName('rgb')
    xout_rgb.input.setBlocking(False)
    xout_rgb.input.setQueueSize(1)
    cam_rgb.preview.link(xout_rgb.input)

    # Estéreo para profundidad
    mono_l = p.create(dai.node.MonoCamera)
    mono_r = p.create(dai.node.MonoCamera)
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_l.setFps(RGB_FPS)
    mono_r.setFps(RGB_FPS)

    stereo = p.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)

    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    xout_depth = p.create(dai.node.XLinkOut)
    xout_depth.setStreamName('depth')
    xout_depth.input.setBlocking(False)
    xout_depth.input.setQueueSize(1)
    stereo.depth.link(xout_depth.input)

    return p


def point_depth_mm(depth: np.ndarray, fx: float, fy: float, frame_shape: Tuple[int, int, int]) -> int:
    """Devuelve profundidad aproximada en mm en el centro de una bbox."""
    x, y, w, h = fx, fy, 0.0, 0.0  # dummy to keep signature concise
    _ = (x, y, w, h, frame_shape)  # quiet lints
    return 0


def run_once(model: cv2.dnn_DetectionModel, classes: list[str]) -> str:
    """Ejecuta una sesión de pipeline; retorna stop/exit/restart."""
    clicked = {'stop': False, 'exit': False}
    btn_stop = [0, 0, 0, 0]
    btn_exit = [0, 0, 0, 0]

    def on_mouse(event, x, y, flags, param):
        _ = (flags, param)
        if event == cv2.EVENT_LBUTTONDOWN:
            if btn_stop[0] <= x <= btn_stop[2] and btn_stop[1] <= y <= btn_stop[3]:
                clicked['stop'] = True
            if btn_exit[0] <= x <= btn_exit[2] and btn_exit[1] <= y <= btn_exit[3]:
                clicked['exit'] = True

    pipeline = build_pipeline()

    # Forzar USB2 para estabilidad en Pi (evita saltos SS/HS en algunos setups)
    try:
        device_cm = dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH)
    except TypeError:
        device_cm = dai.Device(pipeline, usb2Mode=True)

    with device_cm as device:
        q_rgb = device.getOutputQueue(name='rgb', maxSize=1, blocking=False)
        q_depth = device.getOutputQueue(name='depth', maxSize=1, blocking=False)

        win_rgb = 'YOLO PI LUXONIS - RGB+Depth'
        win_depth = 'YOLO PI LUXONIS - Depth'
        cv2.namedWindow(win_rgb)
        cv2.setMouseCallback(win_rgb, on_mouse)
        cv2.namedWindow(win_depth)

        start_t = time.monotonic()
        fps_counter = 0
        fps = 0.0

        last_frame = None
        last_depth = None
        last_frame_ts = time.monotonic()

        # Fluidez: correr YOLO cada N frames y reutilizar última detección entre medias
        detect_every_n = 2
        frame_idx = 0
        cached_class_ids, cached_scores, cached_boxes = [], [], []

        while True:
            if STOP_FILE.exists() or clicked['stop']:
                log('STOP solicitado por botón/flag')
                return 'stop'
            if clicked['exit']:
                log('EXIT solicitado por botón')
                return 'exit'

            msg_rgb = q_rgb.tryGet()
            msg_depth = q_depth.tryGet()

            if msg_rgb is not None:
                last_frame = msg_rgb.getCvFrame()
                last_frame_ts = time.monotonic()
                fps_counter += 1

            if msg_depth is not None:
                last_depth = msg_depth.getFrame()

            now = time.monotonic()
            if now - start_t >= 1.0:
                fps = fps_counter / (now - start_t)
                fps_counter = 0
                start_t = now

            # Si no llega frame RGB por varios segundos, reiniciar pipeline
            if now - last_frame_ts > 6.0:
                raise RuntimeError('No llegan frames RGB >6s (reinicio automático)')

            if last_frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    return 'exit'
                continue

            frame = last_frame.copy()
            h, w = frame.shape[:2]

            frame_idx += 1

            # YOLO host (acelerado): inferencia cada N frames
            if frame_idx % detect_every_n == 0:
                class_ids, scores, boxes = model.detect(frame, confThreshold=CONF_TH, nmsThreshold=NMS_TH)
                if len(class_ids):
                    cached_class_ids = class_ids.flatten()
                    cached_scores = scores.flatten()
                    cached_boxes = boxes
                else:
                    cached_class_ids, cached_scores, cached_boxes = [], [], []

            # Dibujar detecciones + Z(mm) usando caché
            for cid, score, box in zip(cached_class_ids, cached_scores, cached_boxes):
                x, y, bw, bh = [int(v) for v in box]
                label = classes[int(cid)] if int(cid) < len(classes) else str(int(cid))

                z_text = ''
                if last_depth is not None and last_depth.size > 0:
                    dh, dw = last_depth.shape[:2]
                    cx = int((x + bw * 0.5) * dw / w)
                    cy = int((y + bh * 0.5) * dh / h)
                    cx = min(max(cx, 0), dw - 1)
                    cy = min(max(cy, 0), dh - 1)
                    z_mm = int(last_depth[cy, cx])
                    if z_mm > 0:
                        z_text = f' | Z:{z_mm/10:.1f}cm'

                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f'{label} {score*100:.0f}%{z_text}',
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

            # Botones overlay (abajo-izquierda para no tapar etiquetas arriba)
            margin = 12
            btn_w, btn_h = 118, 32
            gap = 10
            y1 = h - margin - btn_h
            y2 = h - margin

            btn_stop[:] = [margin, y1, margin + btn_w, y2]
            btn_exit[:] = [margin + btn_w + gap, y1, margin + btn_w + gap + btn_w, y2]

            cv2.rectangle(frame, (btn_stop[0], btn_stop[1]), (btn_stop[2], btn_stop[3]), (0, 140, 255), -1)
            cv2.putText(frame, 'STOP', (btn_stop[0] + 24, btn_stop[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
            cv2.rectangle(frame, (btn_exit[0], btn_exit[1]), (btn_exit[2], btn_exit[3]), (0, 0, 255), -1)
            cv2.putText(frame, 'EXIT', (btn_exit[0] + 28, btn_exit[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Métricas arriba-izquierda para evitar choque con botones
            cv2.putText(frame, f'detections: {len(cached_boxes)}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f'FPS host: {fps:.1f}', (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow(win_rgb, frame)

            if last_depth is not None:
                depth_vis = cv2.normalize(last_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
                cv2.imshow(win_depth, depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                log('EXIT solicitado por teclado')
                return 'exit'

            # No cerrar automáticamente por eventos de ventana:
            # solo STOP/EXIT explícito o ESC/q.


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    STOP_FILE.unlink(missing_ok=True)

    model, classes = load_detector()
    log('Iniciando YOLO PI LUXONIS')

    while True:
        if STOP_FILE.exists():
            log('STOP flag detectado antes de iniciar')
            break
        try:
            action = run_once(model, classes)
            if action in ('stop', 'exit'):
                break
        except Exception as e:
            log(f'Reinicio de pipeline por excepción: {e}')
            if STOP_FILE.exists():
                break
            time.sleep(2)

    cv2.destroyAllWindows()
    log('YOLO PI LUXONIS detenido')


if __name__ == '__main__':
    main()
