"""
fatigue_model.py  —  TensorFlow/Keras LSTM Fatigue Classifier
"""
import os, numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# ── Keras imports (compatible with TF 2.15 intel backend) ────────────────────
Sequential      = tf.keras.Sequential
load_model      = tf.keras.models.load_model
LSTM            = tf.keras.layers.LSTM
Dense           = tf.keras.layers.Dense
Dropout         = tf.keras.layers.Dropout
BatchNormalization = tf.keras.layers.BatchNormalization
Bidirectional   = tf.keras.layers.Bidirectional
Input           = tf.keras.layers.Input
EarlyStopping   = tf.keras.callbacks.EarlyStopping
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
Adam            = tf.keras.optimizers.Adam

SEQ_LEN    = 30
N_FEATURES = 6
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fatigue_lstm.h5")

def build_model():
    model = Sequential([
        Input(shape=(SEQ_LEN, N_FEATURES)),
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1,  activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model

def _generate_data(n=4000):
    rng = np.random.default_rng(42)
    X, y = [], []
    for _ in range(n // 2):
        ear  = rng.normal(0.30, 0.03, SEQ_LEN).clip(0.15, 0.45)
        mar  = rng.normal(0.20, 0.05, SEQ_LEN).clip(0.05, 0.40)
        pc   = rng.normal(0.05, 0.02, SEQ_LEN).clip(0.0,  0.20)
        pit  = rng.normal(0.05, 0.02, SEQ_LEN).clip(-0.1, 0.15)
        br   = rng.normal(0.25, 0.05, SEQ_LEN).clip(0.05, 0.60)
        X.append(np.stack([ear, ear+rng.normal(0,.01,SEQ_LEN), mar, pc, pit, br], 1)); y.append(0)
    for _ in range(n // 2):
        ear  = rng.normal(0.18, 0.04, SEQ_LEN).clip(0.08, 0.30)
        mar  = rng.normal(0.50, 0.10, SEQ_LEN).clip(0.20, 0.90)
        pc   = rng.normal(0.40, 0.10, SEQ_LEN).clip(0.15, 0.80)
        pit  = rng.normal(0.20, 0.05, SEQ_LEN).clip(0.05, 0.40)
        br   = rng.normal(0.10, 0.04, SEQ_LEN).clip(0.01, 0.30)
        X.append(np.stack([ear, ear+rng.normal(0,.01,SEQ_LEN), mar, pc, pit, br], 1)); y.append(1)
    X,y = np.array(X,np.float32), np.array(y,np.float32)
    idx = rng.permutation(len(X)); return X[idx], y[idx]

def train_and_save(epochs=25, batch=64):
    print("[TF] Generating synthetic training data …")
    X, y = _generate_data()
    s = int(0.85*len(X))
    model = build_model(); model.summary()
    model.fit(X[:s], y[:s], validation_data=(X[s:],y[s:]),
              epochs=epochs, batch_size=batch, verbose=1,
              callbacks=[EarlyStopping(monitor="val_auc", patience=5,
                                       restore_best_weights=True, mode="max"),
                         ModelCheckpoint(MODEL_PATH, monitor="val_auc",
                                         save_best_only=True, mode="max", verbose=0)])
    print(f"[TF] Saved → {MODEL_PATH}"); return model

def get_model():
    if os.path.exists(MODEL_PATH):
        print(f"[TF] Loading {MODEL_PATH}"); return load_model(MODEL_PATH)
    return train_and_save()

class FatigueClassifier:
    def __init__(self):
        self._model = get_model()
        self._buf   = []
        self.prob   = 0.0
    def update(self, ear_l, ear_r, mar, perclos, pitch, blink_rate):
        self._buf.append([ear_l, ear_r, mar, perclos, pitch, blink_rate])
        if len(self._buf) > SEQ_LEN: self._buf.pop(0)
    def predict(self):
        if len(self._buf) < SEQ_LEN: return 0.0
        seq = np.array(self._buf, np.float32)[np.newaxis]
        self.prob = float(self._model.predict(seq, verbose=0)[0][0])
        return self.prob