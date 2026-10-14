"""RESAM reusable library: radar-screen geometry, annotations, metrics, inference backends,
frame sources, tracking and Jetson telemetry.

Modules are import-light on purpose: heavy dependencies (torch, tensorrt, transformers,
ultralytics) are only imported by the backend that needs them, so the dashboard and the
classical baseline run with nothing but numpy + OpenCV.
"""

__version__ = "0.1.0"
