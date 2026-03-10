"""
VIGILANCE  —  Driver Fatigue Detection System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Author  : Daksh Chauhan
Stack   : Python · OpenCV · MediaPipe · TensorFlow/Keras · PyQt5 · pygame
GitHub  : https://github.com/daksh-chauhan/vigilance

Detection pipeline:
  1. MediaPipe FaceMesh → 468 facial landmarks (real-time)
  2. Classical CV metrics → EAR, MAR, PERCLOS, head pitch, blink rate
  3. TensorFlow Bi-LSTM  → fuses 30-frame feature sequences → fatigue P(%)
  4. Multi-tier alert engine → SAFE / CAUTION / WARNING / CRITICAL
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import sys, os, time, math
import numpy as np
import cv2
import mediapipe as mp
import pygame
from scipy.spatial import distance
from collections import deque
from datetime import datetime
import keras
from keras.models import Sequential
from keras.saving import load_model

# PyQt5
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFrame,
    QGridLayout, QSlider, QSizePolicy, QTabWidget,
    QTextEdit, QScrollArea
)
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor,
    QFont, QBrush, QPainterPath, QLinearGradient
)
from PyQt5.QtCore import Qt, QTimer, QRect, QThread, pyqtSignal

# Local modules
sys.path.insert(0, os.path.dirname(__file__))
from fatigue_model import FatigueClassifier
from logger import SessionLogger

# ─── pygame ──────────────────────────────────────────────────────────────────
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

def _make_beep(freq=880, dur=0.5, vol=0.7):
    sr = 44100
    t  = np.linspace(0, dur, int(sr*dur), False)
    w  = (np.sin(2*np.pi*freq*t)*vol*32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([w, w]))

try:    SND_WARN     = pygame.mixer.Sound("assets/alert.wav")
except: SND_WARN     = _make_beep(880, 0.4)
try:    SND_CRITICAL = pygame.mixer.Sound("assets/critical.wav")
except: SND_CRITICAL = _make_beep(1200, 0.8)

# ─── MediaPipe ───────────────────────────────────────────────────────────────
mp_fm = mp.solutions.face_mesh
FACE_MESH = mp_fm.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65,
)

# Landmark indices
L_EYE  = [33, 160, 158, 133, 153, 144]
R_EYE  = [362, 385, 387, 263, 373, 380]
M_TOP, M_BOT = 13, 14
NOSE   = 1
L_TMP, R_TMP = 234, 454
L_BROW, R_BROW = 70, 300   # for brow furrow

# ─── Tunable thresholds ──────────────────────────────────────────────────────
EAR_THRESH    = 0.25
MAR_THRESH    = 0.60
PERCLOS_WIN   = 90          # ~3s at 30fps
HEAD_THRESH   = 0.18
TF_THRESH     = 0.55        # TF model fatigue threshold
BLINK_CONSEC  = 2

# ─── Global state ────────────────────────────────────────────────────────────
ear_buf     = deque(maxlen=PERCLOS_WIN)
feat_buf    = deque(maxlen=30)         # for TF model
blink_times = deque(maxlen=40)
yawn_buf    = deque(maxlen=40)

eye_closed_ctr = 0
yawn_ctr       = 0
blink_count    = 0
yawn_count     = 0
last_alert_t   = 0.0
ALERT_CD       = 2.5

# ─── Metric helpers ──────────────────────────────────────────────────────────
def calc_ear(pts):
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A+B)/(2.0*C+1e-6)

def calc_mar(top, bot, eye_dist):
    return distance.euclidean(top, bot) / (eye_dist+1e-6)

def calc_perclos():
    if not ear_buf: return 0.0
    return sum(1 for e in ear_buf if e < EAR_THRESH)/len(ear_buf)

def calc_head_pitch(lm):
    nose_y = lm[NOSE].y
    mid_y  = (lm[L_TMP].y + lm[R_TMP].y)/2
    return nose_y - mid_y

def blink_rate_per_min():
    now = time.time()
    recent = [t for t in blink_times if now-t < 60]
    return len(recent)

# ─── Alert level ─────────────────────────────────────────────────────────────
def compute_alert(ear, perclos, tf_prob, mar, pitch):
    score = 0
    if ear      < EAR_THRESH:   score += 2
    if perclos  > 0.35:         score += 3
    if tf_prob  > TF_THRESH:    score += 3
    if mar      > MAR_THRESH:   score += 1
    if pitch    > HEAD_THRESH:  score += 2
    if score == 0: return 0, "SAFE",     "#3fb950"
    if score <= 2: return 1, "CAUTION",  "#e3b341"
    if score <= 5: return 2, "WARNING",  "#f0883e"
    return             3,    "CRITICAL", "#f85149"

# ═══════════════════════ CUSTOM WIDGETS ═════════════════════════════════════

class ArcGauge(QWidget):
    """Animated arc gauge — label / value / unit."""
    def __init__(self, label, c_lo="#00c853", c_hi="#d50000", parent=None):
        super().__init__(parent)
        self.setFixedSize(150, 165)
        self.label  = label
        self.c_lo   = QColor(c_lo); self.c_hi = QColor(c_hi)
        self._tgt   = 0.0;  self._cur = 0.0
        t = QTimer(self); t.timeout.connect(self._tick); t.start(16)
    def set_value(self, v):  self._tgt = max(0.0, min(1.0, v))
    def _tick(self):
        self._cur += (self._tgt - self._cur)*0.10
        self.update()
    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        cx,cy,r = 75,78,54
        # track
        pen = QPen(QColor(35,40,52),9); pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen); p.drawArc(cx-r,cy-r,r*2,r*2, 225*16,-270*16)
        # fill
        t = self._cur
        rc = int(self.c_lo.red()*(1-t)   + self.c_hi.red()*t)
        gc = int(self.c_lo.green()*(1-t) + self.c_hi.green()*t)
        bc = int(self.c_lo.blue()*(1-t)  + self.c_hi.blue()*t)
        pen = QPen(QColor(rc,gc,bc),9); pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(cx-r,cy-r,r*2,r*2, 225*16, int(-270*self._cur)*16)
        # pct text
        p.setPen(QColor(225,225,235))
        p.setFont(QFont("Consolas",16,QFont.Bold))
        p.drawText(QRect(cx-38,cy-18,76,32),Qt.AlignCenter,f"{int(self._cur*100)}%")
        # label
        p.setPen(QColor(140,148,164)); p.setFont(QFont("Consolas",8))
        p.drawText(QRect(0,cy+r+2,150,18),Qt.AlignCenter,self.label)
        p.end()

class TFProbBar(QWidget):
    """Horizontal TF probability bar with gradient."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(34)
        self._val = 0.0; self._cur = 0.0
        t = QTimer(self); t.timeout.connect(self._tick); t.start(16)
    def set_value(self, v): self._val = max(0.0,min(1.0,v))
    def _tick(self):
        self._cur += (self._val-self._cur)*0.08; self.update()
    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w,h = self.width(), self.height()
        path = QPainterPath(); path.addRoundedRect(0,6,w,18,9,9)
        p.fillPath(path, QColor(28,32,42))
        fw = int(self._cur*(w-4))
        if fw > 4:
            grad = QLinearGradient(2,0,w-2,0)
            grad.setColorAt(0.0, QColor(0,200,100))
            grad.setColorAt(0.5, QColor(240,160,0))
            grad.setColorAt(1.0, QColor(220,40,40))
            fill = QPainterPath(); fill.addRoundedRect(2,8,fw,14,7,7)
            p.fillPath(fill, QBrush(grad))
        p.setPen(QColor(180,188,210)); p.setFont(QFont("Consolas",9,QFont.Bold))
        pct = int(self._cur*100)
        label = f"TF MODEL  —  Fatigue Probability: {pct}%"
        p.drawText(QRect(0,0,w,h), Qt.AlignCenter, label)
        p.end()

class StatCard(QWidget):
    def __init__(self, title, unit="", parent=None):
        super().__init__(parent); self.setFixedSize(130,68)
        self.title=title; self.unit=unit; self._val="—"
    def set_value(self,v): self._val=str(v); self.update()
    def paintEvent(self,_):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        path=QPainterPath(); path.addRoundedRect(0,0,130,68,8,8)
        p.fillPath(path,QColor(22,26,35))
        p.setPen(QColor(80,90,110)); p.setFont(QFont("Consolas",7))
        p.drawText(QRect(0,7,130,14),Qt.AlignCenter,self.title.upper())
        p.setPen(QColor(210,218,235)); p.setFont(QFont("Consolas",18,QFont.Bold))
        p.drawText(QRect(0,22,108,30),Qt.AlignRight|Qt.AlignVCenter,self._val)
        p.setPen(QColor(88,130,200)); p.setFont(QFont("Consolas",8))
        p.drawText(QRect(110,32,18,18),Qt.AlignLeft|Qt.AlignVCenter,self.unit)
        p.end()

class StatusBanner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setFixedHeight(46)
        self._txt="#00c853"; self._msg="INITIALISING …"
        self._alpha=255; self._dir=-1
        self._pulse=QTimer(self); self._pulse.timeout.connect(self._tick)
    def set_status(self, msg, color):
        self._msg=msg; self._txt=color
        if msg not in ("SAFE","INITIALISING …"):
            self._pulse.start(25)
        else:
            self._pulse.stop(); self._alpha=255
        self.update()
    def _tick(self):
        self._alpha+=self._dir*9
        if self._alpha<=70: self._dir=1
        if self._alpha>=255: self._dir=-1
        self.update()
    def paintEvent(self,_):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        bg=QColor(self._txt); bg.setAlpha(25); p.fillRect(self.rect(),bg)
        dot=QColor(self._txt); dot.setAlpha(self._alpha)
        p.setBrush(dot); p.setPen(Qt.NoPen)
        p.drawEllipse(14,15,14,14)
        p.setPen(QColor(self._txt))
        p.setFont(QFont("Consolas",12,QFont.Bold))
        p.drawText(QRect(38,0,self.width()-48,46),Qt.AlignVCenter,self._msg)
        p.end()

# ═══════════════════════ TF TRAINING THREAD ══════════════════════════════════
class TrainThread(QThread):
    done = pyqtSignal(object)
    def run(self):
        clf = FatigueClassifier()
        self.done.emit(clf)

# ═══════════════════════ MAIN WINDOW ════════════════════════════════════════
class Vigilance(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VIGILANCE  |  Driver Fatigue Detection System")
        self.resize(1340, 800)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#0d1117;color:#c9d1d9;font-family:'Consolas';}
            QPushButton{border:1px solid #30363d;border-radius:7px;padding:7px 14px;
                        font-family:'Consolas';font-size:11px;font-weight:bold;letter-spacing:1px;}
            QPushButton:hover{border-color:#58a6ff;background:#161b22;}
            QSlider::groove:horizontal{height:4px;background:#21262d;border-radius:2px;}
            QSlider::handle:horizontal{width:14px;height:14px;margin:-5px 0;
                                        background:#58a6ff;border-radius:7px;}
            QSlider::sub-page:horizontal{background:#58a6ff;border-radius:2px;}
            QTabWidget::pane{border:1px solid #21262d;border-radius:6px;}
            QTabBar::tab{background:#161b22;color:#8b949e;padding:6px 16px;
                         border-top-left-radius:6px;border-top-right-radius:6px;
                         font-family:'Consolas';font-size:10px;}
            QTabBar::tab:selected{background:#21262d;color:#c9d1d9;}
            QTextEdit{background:#0d1117;color:#8b949e;font-family:'Consolas';
                      font-size:10px;border:1px solid #21262d;border-radius:4px;}
        """)

        self.clf        = None   # set after TF training thread
        self.logger     = SessionLogger()
        self.running    = False
        self.tf_ready   = False
        self.tf_prob    = 0.0

        self._build_ui()
        self._init_camera()
        self._start_tf_training()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        rl   = QHBoxLayout(root); rl.setContentsMargins(10,10,10,10); rl.setSpacing(10)

        # ── LEFT ─────────────────────────────────────────────────────────────
        left = QFrame(); left.setStyleSheet("background:#161b22;border-radius:10px;")
        lv   = QVBoxLayout(left); lv.setContentsMargins(10,10,10,10); lv.setSpacing(8)

        # Brand row
        br = QHBoxLayout()
        brand = QLabel("◈  VIGILANCE"); brand.setStyleSheet(
            "color:#58a6ff;font-size:14px;font-weight:bold;letter-spacing:4px;")
        sub   = QLabel("Driver Fatigue Detection  ·  MediaPipe + TensorFlow Bi-LSTM")
        sub.setStyleSheet("color:#30363d;font-size:9px;")
        br.addWidget(brand); br.addStretch(); br.addWidget(sub)
        lv.addLayout(br)

        self.status_banner = StatusBanner()
        lv.addWidget(self.status_banner)

        # Video
        self.video_lbl = QLabel()
        self.video_lbl.setFixedSize(900,500)
        self.video_lbl.setStyleSheet("background:#000;border-radius:8px;border:1px solid #21262d;")
        self.video_lbl.setAlignment(Qt.AlignCenter)
        lv.addWidget(self.video_lbl, alignment=Qt.AlignCenter)

        # TF probability bar
        self.tf_bar = TFProbBar(); lv.addWidget(self.tf_bar)

        # Threshold row
        tr = QHBoxLayout()
        tl = QLabel("EAR THRESHOLD"); tl.setStyleSheet("color:#484f58;font-size:9px;letter-spacing:2px;")
        self.thr_lbl = QLabel("0.25"); self.thr_lbl.setStyleSheet("color:#58a6ff;font-size:10px;font-weight:bold;")
        self.thr_sld = QSlider(Qt.Horizontal); self.thr_sld.setRange(15,40); self.thr_sld.setValue(25)
        self.thr_sld.valueChanged.connect(self._on_thr)
        tr.addWidget(tl); tr.addWidget(self.thr_sld); tr.addWidget(self.thr_lbl)
        lv.addLayout(tr)

        # Buttons
        brow = QHBoxLayout()
        self.btn_start  = QPushButton("▶  START");  self.btn_start.clicked.connect(self.start)
        self.btn_pause  = QPushButton("⏸  PAUSE");  self.btn_pause.clicked.connect(self.pause)
        self.btn_reset  = QPushButton("↺  RESET");  self.btn_reset.clicked.connect(self.reset)
        self.btn_export = QPushButton("⬇  EXPORT"); self.btn_export.clicked.connect(self._export)
        self.btn_exit   = QPushButton("✕  EXIT");   self.btn_exit.clicked.connect(self.close)
        styles = [
            "background:#0d4429;color:#3fb950;border-color:#238636;",
            "background:#1c2128;color:#8b949e;",
            "background:#1c2128;color:#8b949e;",
            "background:#1a2233;color:#79c0ff;border-color:#1f3a5f;",
            "background:#2d1515;color:#f85149;border-color:#b91c1c;",
        ]
        for btn, sty in zip([self.btn_start,self.btn_pause,self.btn_reset,
                              self.btn_export,self.btn_exit], styles):
            btn.setFixedHeight(34); btn.setStyleSheet(sty); brow.addWidget(btn)
        lv.addLayout(brow)
        rl.addWidget(left, stretch=3)

        # ── RIGHT ────────────────────────────────────────────────────────────
        right = QFrame(); right.setFixedWidth(330)
        right.setStyleSheet("background:#161b22;border-radius:10px;")
        rv = QVBoxLayout(right); rv.setContentsMargins(12,12,12,12); rv.setSpacing(12)

        tabs = QTabWidget()
        # ── Tab 1: Live Metrics ───────────────────────────────────────────
        t1 = QWidget(); t1v = QVBoxLayout(t1); t1v.setContentsMargins(8,8,8,8); t1v.setSpacing(10)

        glbl = QLabel("FATIGUE GAUGES"); glbl.setStyleSheet("color:#484f58;font-size:9px;letter-spacing:3px;")
        t1v.addWidget(glbl)
        grow = QHBoxLayout()
        self.g_drowsy  = ArcGauge("DROWSINESS",  "#00c853","#d50000")
        self.g_perclos = ArcGauge("PERCLOS",      "#00b0ff","#ff6d00")
        self.g_tf      = ArcGauge("TF FATIGUE",   "#a371f7","#d50000")
        grow.addWidget(self.g_drowsy,  alignment=Qt.AlignCenter)
        grow.addWidget(self.g_perclos, alignment=Qt.AlignCenter)
        t1v.addLayout(grow)
        t1v.addWidget(self.g_tf, alignment=Qt.AlignCenter)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setStyleSheet("color:#21262d;")
        t1v.addWidget(sep)

        slbl = QLabel("LIVE STATS"); slbl.setStyleSheet("color:#484f58;font-size:9px;letter-spacing:3px;")
        t1v.addWidget(slbl)
        sg = QGridLayout(); sg.setSpacing(6)
        self.c_blinks  = StatCard("Blinks")
        self.c_yawns   = StatCard("Yawns")
        self.c_ear     = StatCard("EAR","")
        self.c_pitch   = StatCard("Head Pitch","")
        self.c_bpm     = StatCard("Blinks/min","")
        self.c_sess    = StatCard("Session","s")
        for i,c in enumerate([self.c_blinks,self.c_yawns,self.c_ear,
                               self.c_pitch,self.c_bpm,self.c_sess]):
            sg.addWidget(c, i//2, i%2)
        t1v.addLayout(sg)
        t1v.addStretch()
        tabs.addTab(t1, "METRICS")

        # ── Tab 2: Alert Log ──────────────────────────────────────────────
        t2 = QWidget(); t2v = QVBoxLayout(t2); t2v.setContentsMargins(8,8,8,8)
        albl = QLabel("ALERT LOG"); albl.setStyleSheet("color:#484f58;font-size:9px;letter-spacing:3px;")
        t2v.addWidget(albl)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        t2v.addWidget(self.log_box)
        tabs.addTab(t2, "LOG")

        # ── Tab 3: Model Info ─────────────────────────────────────────────
        t3 = QWidget(); t3v = QVBoxLayout(t3); t3v.setContentsMargins(8,8,8,8)
        self.model_info = QTextEdit(); self.model_info.setReadOnly(True)
        self.model_info.setPlainText(
            "TF MODEL STATUS\n"
            "━━━━━━━━━━━━━━━\n"
            "Architecture : Bidirectional LSTM\n"
            "Input shape  : (30 frames × 6 features)\n"
            "Features     : EAR_L, EAR_R, MAR,\n"
            "               PERCLOS, Head Pitch,\n"
            "               Blink Rate/min\n"
            "Output       : Fatigue probability\n"
            "Training     : Synthetic + augmented\n"
            "               data (4000 sequences)\n\n"
            "STATUS       : Training in background…\n"
            "               (will activate when ready)"
        )
        t3v.addWidget(self.model_info)
        tabs.addTab(t3, "MODEL")

        rv.addWidget(tabs)

        # Camera indicator
        camr = QHBoxLayout()
        self.cam_dot = QLabel("●"); self.cam_dot.setStyleSheet("color:#3fb950;font-size:13px;")
        self.cam_lbl = QLabel("CAM 0  |  OFFLINE")
        self.cam_lbl.setStyleSheet("color:#484f58;font-size:9px;")
        camr.addWidget(self.cam_dot); camr.addWidget(self.cam_lbl); camr.addStretch()
        rv.addLayout(camr)
        rl.addWidget(right, stretch=1)

    # ── Camera ────────────────────────────────────────────────────────────────
    def _init_camera(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cam_lbl.setText("CAM 0  |  READY")
        else:
            self.cam_dot.setStyleSheet("color:#f85149;font-size:13px;")
            self.cam_lbl.setText("CAM 0  |  NOT FOUND")

        self.timer = QTimer(); self.timer.timeout.connect(self._frame)
        self.clock = QTimer(); self.clock.timeout.connect(self._tick_clock)
        self.clock.start(1000)
        self.sess_start = time.time()
        self.status_banner.set_status("READY — Press START", "#58a6ff")

    # ── TF Training thread ────────────────────────────────────────────────────
    def _start_tf_training(self):
        self._train_thread = TrainThread()
        self._train_thread.done.connect(self._on_tf_ready)
        self._train_thread.start()

    def _on_tf_ready(self, clf):
        self.clf = clf; self.tf_ready = True
        self.model_info.setPlainText(
            "TF MODEL STATUS\n"
            "━━━━━━━━━━━━━━━\n"
            "Architecture : Bidirectional LSTM\n"
            "Input shape  : (30 frames × 6 features)\n"
            "Features     : EAR_L, EAR_R, MAR,\n"
            "               PERCLOS, Head Pitch,\n"
            "               Blink Rate/min\n"
            "Output       : Fatigue probability\n"
            "Training     : Synthetic + augmented\n"
            "               data (4000 sequences)\n\n"
            f"STATUS       : ✓ ACTIVE\n"
            f"Saved path   : models/fatigue_lstm.h5"
        )
        self._log("TF Bi-LSTM model ready ✓")

    # ── Controls ──────────────────────────────────────────────────────────────
    def start(self):
        if not self.cap.isOpened(): return
        self.running = True
        self.sess_start = time.time()
        self.timer.start(30)
        self.cam_lbl.setText("CAM 0  |  ACTIVE")
        self.cam_dot.setStyleSheet("color:#3fb950;font-size:13px;")
        self.status_banner.set_status("SAFE", "#3fb950")
        self._log("Session started")

    def pause(self):
        self.running = False; self.timer.stop()
        pygame.mixer.stop()
        self.status_banner.set_status("PAUSED", "#e3b341")
        self._log("Session paused")

    def reset(self):
        global eye_closed_ctr, yawn_ctr, blink_count, yawn_count
        eye_closed_ctr = yawn_ctr = blink_count = yawn_count = 0
        ear_buf.clear(); feat_buf.clear(); blink_times.clear(); yawn_buf.clear()
        self.tf_prob = 0.0
        self.g_drowsy.set_value(0); self.g_perclos.set_value(0); self.g_tf.set_value(0)
        self.tf_bar.set_value(0)
        for c in [self.c_blinks,self.c_yawns,self.c_ear,self.c_pitch,self.c_bpm,self.c_sess]:
            c.set_value("—")
        self.sess_start = time.time()
        self.status_banner.set_status("SAFE", "#3fb950")
        self._log("Session reset")

    def _on_thr(self, v):
        global EAR_THRESH
        EAR_THRESH = v/100.0; self.thr_lbl.setText(f"{EAR_THRESH:.2f}")

    def _tick_clock(self):
        self.c_sess.set_value(str(int(time.time()-self.sess_start)))

    def _export(self):
        self._log(f"CSV saved → {self.logger.path}")

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{ts}]  {msg}")

    # ── Frame processing ──────────────────────────────────────────────────────
    def _frame(self):
        global eye_closed_ctr, yawn_ctr, blink_count, yawn_count, last_alert_t

        ret, frame = self.cap.read()
        if not ret: return

        frame = cv2.resize(frame, (900,500))
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = FACE_MESH.process(rgb)

        avg_ear=0.28; mar_val=0.0; pitch=0.0; perclos_v=0.0
        face_ok = False

        if res.multi_face_landmarks:
            face_ok = True
            lm = res.multi_face_landmarks[0].landmark
            H, W = frame.shape[:2]
            def pt(i): return (int(lm[i].x*W), int(lm[i].y*H))

            lp = [pt(i) for i in L_EYE]
            rp = [pt(i) for i in R_EYE]
            l_ear = calc_ear(lp); r_ear = calc_ear(rp)
            avg_ear = (l_ear+r_ear)/2.0
            ear_buf.append(avg_ear)
            perclos_v = calc_perclos()

            # Eye closed counter / blink detection
            if avg_ear < EAR_THRESH:
                eye_closed_ctr += 1
            else:
                if eye_closed_ctr >= BLINK_CONSEC:
                    blink_count += 1
                    blink_times.append(time.time())
                eye_closed_ctr = 0

            # MAR / yawn
            lec = np.mean(lp, axis=0); rec = np.mean(rp, axis=0)
            eye_d = distance.euclidean(lec, rec)
            mar_val = calc_mar(pt(M_TOP), pt(M_BOT), eye_d)
            yawn_buf.append(mar_val)
            if mar_val > MAR_THRESH:
                yawn_ctr += 1
            elif yawn_ctr >= 15:
                yawn_count += 1; yawn_ctr = 0
                self._log(f"Yawn #{yawn_count} detected")
            else:
                yawn_ctr = max(0, yawn_ctr-1)

            # Head pitch
            pitch = calc_head_pitch(lm)

            # TF model update
            bpm = blink_rate_per_min()
            if self.clf:
                self.clf.update(l_ear, r_ear, mar_val, perclos_v, pitch, bpm/60.0)
                if len(feat_buf) == 0 or True:   # predict every frame
                    self.tf_prob = self.clf.predict()

            # ── Draw overlays ─────────────────────────────────────────────
            eye_c = (50,220,100) if avg_ear >= EAR_THRESH else (60,60,240)
            for pts in [lp, rp]:
                hull = cv2.convexHull(np.array(pts))
                cv2.polylines(frame,[hull],True,eye_c,1)

            # Mouth outline
            m_pts = [pt(i) for i in [61,185,40,39,37,0,267,269,270,409,
                                      291,375,321,405,314,17,84,181,91,146]]
            cv2.polylines(frame,[np.array(m_pts)],True,(160,160,80),1)

            # Head pitch indicator
            ph = int(min(abs(pitch)/HEAD_THRESH, 1.0)*60)
            pc = (50,220,100) if pitch < HEAD_THRESH else (60,60,240)
            cv2.rectangle(frame,(16,20),(30,80),(28,33,46),-1)
            cv2.rectangle(frame,(16,80-ph),(30,80),pc,-1)
            cv2.putText(frame,"HD",(10,16),cv2.FONT_HERSHEY_PLAIN,0.7,(100,110,130),1)

            # HUD text
            cv2.putText(frame,f"EAR {avg_ear:.3f}",(W-130,22),cv2.FONT_HERSHEY_PLAIN,0.9,(170,180,200),1)
            cv2.putText(frame,f"MAR {mar_val:.3f}",(W-130,38),cv2.FONT_HERSHEY_PLAIN,0.9,(170,180,200),1)
            cv2.putText(frame,f"PC  {perclos_v:.2f}",(W-130,54),cv2.FONT_HERSHEY_PLAIN,0.9,(170,180,200),1)
            cv2.putText(frame,f"TF  {self.tf_prob:.2f}",(W-130,70),cv2.FONT_HERSHEY_PLAIN,0.9,(180,140,230),1)
            cv2.putText(frame,f"BLK {blink_count}",(W-130,86),cv2.FONT_HERSHEY_PLAIN,0.9,(170,180,200),1)

            # Alert overlay
            lvl,msg,col_hex = compute_alert(avg_ear, perclos_v, self.tf_prob, mar_val, pitch)
            bgr = tuple(int(col_hex.lstrip("#")[i:i+2],16) for i in (4,2,0))

            if lvl >= 2:
                overlay = frame.copy()
                cv2.rectangle(overlay,(0,0),(900,500),bgr,6)
                cv2.addWeighted(overlay,0.4,frame,0.6,0,frame)
                cv2.putText(frame,msg,(int(900/2)-100,50),
                            cv2.FONT_HERSHEY_SIMPLEX,1.4,bgr,2)

            # Update widgets
            now = time.time()
            self.status_banner.set_status(msg, col_hex)
            drowsy_pct = max(0.0, min(1.0,(1-avg_ear/EAR_THRESH)))
            self.g_drowsy.set_value(drowsy_pct)
            self.g_perclos.set_value(perclos_v)
            self.g_tf.set_value(self.tf_prob)
            self.tf_bar.set_value(self.tf_prob)
            self.c_blinks.set_value(str(blink_count))
            self.c_yawns.set_value(str(yawn_count))
            self.c_ear.set_value(f"{avg_ear:.3f}")
            self.c_pitch.set_value(f"{pitch:.3f}")
            self.c_bpm.set_value(str(int(blink_rate_per_min())))

            if lvl >= 1 and (now-last_alert_t) > ALERT_CD:
                last_alert_t = now
                if lvl >= 3: SND_CRITICAL.play()
                else:        SND_WARN.play()
                self._log(f"ALERT [{msg}]  EAR={avg_ear:.3f}  TF={self.tf_prob:.2f}")

            # Log to CSV
            self.logger.log(
                ear=round(avg_ear,4), perclos=round(perclos_v,3),
                mar=round(mar_val,3), head_pitch=round(pitch,3),
                blink_count=blink_count, yawn_count=yawn_count,
                tf_prob=round(self.tf_prob,3), alert_level=lvl
            )

        else:
            cv2.putText(frame,"NO FACE DETECTED",(300,250),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,(80,80,100),2)
            self.status_banner.set_status("NO FACE DETECTED","#484f58")

        # Display
        rgb2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb2.shape
        qi = QImage(rgb2.data,w,h,ch*w,QImage.Format_RGB888)
        self.video_lbl.setPixmap(QPixmap.fromImage(qi))

    def closeEvent(self, e):
        self.timer.stop()
        self.logger.close()
        if self.cap.isOpened(): self.cap.release()
        pygame.mixer.quit()
        e.accept()

# ─── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = Vigilance()
    win.show()
    sys.exit(app.exec_())
