# VIGILANCE — Driver Fatigue Detection System

A real-time drowsiness detection app that watches your face through a webcam and alerts you when it thinks you're getting too tired to drive safely.

---

## What It Does

VIGILANCE watches your face in real time and tracks signs of fatigue — drooping eyes, yawning, head nodding. When it detects danger signs, it triggers a visual and audio alert to wake you up before an accident happens.

It runs at ~30 frames per second and gives alerts at 4 levels — Safe, Caution, Warning, and Critical.

---

## How It Works

**Step 1 — Face Tracking**
Your webcam feed is analyzed using MediaPipe, which places 468 invisible dots on your face every frame to track your eyes, mouth, and head position.

**Step 2 — Fatigue Metrics**
From those dots, the app calculates:
- **EAR** — how open your eyes are (low = drowsy)
- **PERCLOS** — what % of the last 3 seconds your eyes were closed
- **MAR** — how wide your mouth is open (yawn detection)
- **Head Pitch** — whether your head is drooping forward
- **Blink Rate** — blinks per minute (too low = fatigue)

**Step 3 — AI Prediction**
All 6 metrics are fed into a Bidirectional LSTM neural network that looks at the last 30 frames together and outputs a fatigue probability score between 0% and 100%.

**Step 4 — Alert**
Based on the score, it shows a coloured alert on screen and plays an audio alarm.

---

## Alert Levels

| Level | What It Means |
|---|---|
| 🟢 SAFE | You're fine |
| 🟡 CAUTION | Early signs of tiredness |
| 🟠 WARNING | Clearly drowsy — take a break |
| 🔴 CRITICAL | Dangerous — loud alarm triggers |

---

## Tech Used

- **Python 3.11**
- **OpenCV** — webcam input and drawing on screen
- **MediaPipe** — face landmark detection
- **TensorFlow / Keras** — LSTM fatigue prediction model
- **PyQt5** — desktop app interface with live gauges
- **pygame** — audio alerts

---

## How to Run

```bash
# Install dependencies
py -3.11 -m pip install -r requirements.txt

# Run the app
py -3.11 main.py
```

On first launch, the AI model trains itself automatically in the background (takes about 30 seconds). After that it saves and loads instantly every time.

---

## Project Structure

```
vigilance/
├── main.py              # Main app
├── fatigue_model.py     # AI model
├── logger.py            # Saves session data to CSV
├── requirements.txt
└── assets/
    ├── alert.wav        # Warning sound (optional)
    └── critical.wav     # Critical alarm (optional)
```

---

## License

MIT — free to use for academic and personal projects.