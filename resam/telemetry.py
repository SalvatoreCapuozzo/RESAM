"""Jetson telemetry without root: jtop service (power rails, GPU load, temps) when available,
else tegrastats parsing (no power rails on Orin Nano without root), else /sys GPU load."""

import re
import subprocess
import sys
import threading
import time

JTOP_PATHS = ["/usr/local/lib/python3.8/dist-packages", "/usr/lib/python3/dist-packages"]


class Telemetry:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.latest = {}
        self.energy_j = 0.0
        self._last_t = None
        self._stop = threading.Event()
        self.backend = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def snapshot(self):
        d = dict(self.latest)
        d["age_s"] = time.time() - d["ts"] if "ts" in d else None
        d["energy_j"] = self.energy_j
        d["backend"] = self.backend
        return d

    def _integrate(self, power_mw):
        now = time.time()
        if self._last_t is not None and power_mw is not None:
            self.energy_j += power_mw / 1000.0 * (now - self._last_t)
        self._last_t = now

    def _run(self):
        # jtop's ok() returns False on a missed sample under heavy GPU load; the old loop then
        # exited and the last reading stayed frozen. Reconnect until stopped.
        failures = 0
        while not self._stop.is_set() and failures < 5:
            try:
                self._run_jtop()
                failures = 0
            except Exception:
                failures += 1
                time.sleep(1.0)
        if self._stop.is_set():
            return
        try:
            self._run_tegrastats()
        except Exception:
            self._run_sysfs()

    def _run_jtop(self):
        for p in JTOP_PATHS:
            if p not in sys.path:
                sys.path.append(p)
        from jtop import jtop
        with jtop(interval=self.interval) as j:
            self.backend = "jtop"
            misses = 0
            while not self._stop.is_set():
                if not j.ok():
                    misses += 1
                    if misses > 10:
                        return  # reconnect
                    time.sleep(self.interval / 2)
                    continue
                misses = 0
                s = j.stats
                tot = j.power.get("tot", {}) if isinstance(j.power, dict) else {}
                rails = j.power.get("rail", {}) if isinstance(j.power, dict) else {}
                cpus = [v for k, v in s.items() if re.match(r"CPU\d+$", k) and isinstance(v, (int, float))]
                ram = j.memory.get("RAM", {}) if hasattr(j, "memory") else {}
                d = {
                    "gpu_pct": s.get("GPU"),
                    "cpu_pct": sum(cpus) / len(cpus) if cpus else None,
                    "ram_used_mb": ram.get("used", 0) / 1024 if ram else None,
                    "ram_total_mb": ram.get("tot", 0) / 1024 if ram else None,
                    "temp_gpu": s.get("Temp GPU"),
                    "temp_cpu": s.get("Temp CPU"),
                    "power_mw": tot.get("power"),
                    "power_avg_mw": tot.get("avg"),
                    "rails_mw": {k: v.get("power") for k, v in rails.items()},
                    "nvp_model": s.get("nvp model"),
                    "jetson_clocks": s.get("jetson_clocks"),
                    "ts": time.time(),
                }
                self._integrate(d["power_mw"])
                self.latest = d

    def _run_tegrastats(self):
        p = subprocess.Popen(["tegrastats", "--interval", str(int(self.interval * 1000))],
                             stdout=subprocess.PIPE, text=True)
        self.backend = "tegrastats"
        for line in p.stdout:
            if self._stop.is_set():
                p.kill()
                return
            d = {}
            m = re.search(r"RAM (\d+)/(\d+)MB", line)
            if m:
                d["ram_used_mb"], d["ram_total_mb"] = float(m.group(1)), float(m.group(2))
            m = re.search(r"CPU \[([^\]]+)\]", line)
            if m:
                loads = [float(x.split("%")[0]) for x in m.group(1).split(",") if "%" in x]
                d["cpu_pct"] = sum(loads) / len(loads) if loads else None
            m = re.search(r"GR3D_FREQ (\d+)%", line)
            if m:
                d["gpu_pct"] = float(m.group(1))
            m = re.search(r"GPU@([\d.]+)C", line)
            if m:
                d["temp_gpu"] = float(m.group(1))
            m = re.search(r"CPU@([\d.]+)C", line)
            if m:
                d["temp_cpu"] = float(m.group(1))
            m = re.search(r"VDD_IN (\d+)mW", line)
            d["power_mw"] = float(m.group(1)) if m else None
            d["ts"] = time.time()
            self._integrate(d["power_mw"])
            self.latest = d

    def _run_sysfs(self):
        self.backend = "sysfs"
        while not self._stop.is_set():
            try:
                with open("/sys/devices/gpu.0/load") as f:
                    self.latest = {"gpu_pct": int(f.read().strip()) / 10.0}
            except OSError:
                pass
            time.sleep(self.interval)
