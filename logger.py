"""
logger.py  —  CSV session logger + basic analytics
"""
import csv, os, time
from datetime import datetime
from collections import deque

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

class SessionLogger:
    FIELDS = ["timestamp", "ear", "perclos", "mar", "head_pitch",
              "blink_count", "yawn_count", "tf_prob", "alert_level"]

    def __init__(self):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOG_DIR, f"session_{ts}.csv")
        self._f   = open(path, "w", newline="")
        self._w   = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader()
        self._path = path
        self._rows = 0

    def log(self, **kwargs):
        row = {f: kwargs.get(f, "") for f in self.FIELDS}
        row["timestamp"] = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._w.writerow(row)
        self._rows += 1
        if self._rows % 30 == 0:
            self._f.flush()

    def close(self):
        self._f.close()

    @property
    def path(self): return self._path
