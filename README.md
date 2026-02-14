# YOLO PI LUXONIS (sin Coral)

Pipeline de visión por computador para **Raspberry Pi + Luxonis OAK**, orientado a operación estable local.

## Qué hace
- Captura **RGB** desde Luxonis OAK.
- Calcula **profundidad estéreo** en Luxonis.
- Ejecuta detección de objetos en la Pi con **YOLOv4-tiny (OpenCV DNN)**.
- Fusiona detección + profundidad para mostrar distancia aproximada **Z** por objeto.

## Método técnico (detección + profundidad)
1. **RGB stream**: `ColorCamera` en OAK → host (Pi).
2. **Depth stream**: `Mono left/right + StereoDepth` en OAK → mapa de profundidad en mm.
3. **Detección host**: OpenCV DNN (`readNetFromDarknet`) sobre frame RGB.
4. **Fusión espacial**: para cada bounding box se toma el centro y se proyecta al mapa depth para estimar `Z`.

## Funcionalidades de operación
- Botones en UI: **STOP** y **EXIT**.
- Teclas rápidas: `ESC` / `q`.
- Watchdog: reinicia pipeline si dejan de llegar frames.
- Logs en runtime para diagnóstico.

## Hardware requerido
- Raspberry Pi 4 (recomendado 8GB, también funciona con menos).
- Cámara Luxonis OAK (con sensores estéreo activos).
- Cable USB de buena calidad (datos + energía estables).

## Software requerido
- Python con:
  - `depthai`
  - `opencv-python`
  - `numpy`
- Modelos YOLO en host (rutas usadas por script):
  - `/home/machina/.openclaw/workspace/coral/models/yolov4-tiny.cfg`
  - `/home/machina/.openclaw/workspace/coral/models/yolov4-tiny.weights`
  - `/home/machina/.openclaw/workspace/coral/models/coco.names`

## Archivos principales
- `YOLO_PI_LUXONIS.py`
- `launchers/START.desktop` (doble click)

## Ejecución
### Doble click
- Abrir: `launchers/START.desktop`

### Manual
```bash
python YOLO_PI_LUXONIS.py
```

## Troubleshooting rápido
Si hay congelamientos o cierres, revisar reconexiones USB:
```bash
dmesg | tail -n 120
```
Si aparecen `USB disconnect` en la OAK, el problema suele ser de enlace/energía USB, no del modelo.
