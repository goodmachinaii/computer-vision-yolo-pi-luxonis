# YOLO PI LUXONIS (Sin Coral)

Versión enfocada en estabilidad y simplicidad:
- **Luxonis OAK**: RGB + profundidad.
- **Raspberry Pi (CPU host)**: detección YOLOv4-tiny.

## Características
- Detección de objetos + etiqueta + confianza.
- Distancia aproximada **Z** usando depth de Luxonis.
- Botones en ventana:
  - **STOP**
  - **EXIT**
- Teclas: `ESC` o `q`.
- Watchdog y reinicio automático de pipeline cuando no llegan frames.

## Archivo principal
- `YOLO_PI_LUXONIS.py`

## Dependencias
- Python con `opencv-python` y `depthai`.
- Modelos YOLO en host (ruta usada por script):
  - `/home/machina/.openclaw/workspace/coral/models/yolov4-tiny.cfg`
  - `/home/machina/.openclaw/workspace/coral/models/yolov4-tiny.weights`
  - `/home/machina/.openclaw/workspace/coral/models/coco.names`

## Ejecución manual
```bash
python YOLO_PI_LUXONIS.py
```

## Notas de estabilidad
Si hay congelamientos/cierres, revisar primero reconexiones USB de la OAK:
```bash
dmesg | tail -n 100
```
