# Five-Cat-PawSture

A webcam-and-Arduino posture-awareness system that turns body pose into clear visual, audio, and physical feedback.

## Team

Five-Cat-PawSture, also called **PawSture**, was created as a group project by **Conan, Amshanaa, Yifan, and Kevin**. The project combines computer vision, a real-time software interface, and a physical Arduino display.

## Purpose

PawSture helps users notice prolonged slouching or poor sitting posture without requiring a wearable sensor. It observes a user through a webcam, compares their current pose with a personal baseline, and reports one of three simple states:

- **HAPPY** — posture is close to the calibrated baseline.
- **NEUTRAL** — posture is beginning to drift.
- **SAD** — posture has moved far enough from the baseline to trigger corrective feedback.

The states appear in the desktop or browser interface and can also be sent to a physical **Stats & Emotion Cube**, where an OLED screen displays an animated face.

## Features

- Real-time pose tracking with an Ultralytics YOLO pose model and OpenCV.
- Automatic switching between front-view and side-view posture analysis.
- Separate personal baselines for front and side views, captured automatically after several stable frames.
- Manual recalibration from the interface whenever the user's position changes.
- Front-view analysis using head, eye, shoulder, hip, and neck-compression measurements.
- Side-view analysis using neck and trunk angles.
- Persistent person tracking to stay focused on one user when multiple people appear.
- Exponential moving-average smoothing to reduce noisy frame-to-frame readings.
- Live posture overlays, measurements, state, and connection status.
- Optional audio alerts after poor posture continues for several seconds.
- Two interfaces: a standard OpenCV window and a local browser dashboard.
- Automatic serial-port detection, with manual camera, port, and baud-rate options.
- Arduino-driven animated **HAPPY**, **NEUTRAL**, and **SAD** faces on a 128×64 OLED display.

## How It Works

1. A webcam supplies live video frames.
2. The pose model identifies body keypoints such as the eyes, ears, shoulders, and hips.
3. PawSture selects the tracked person and determines whether the front or side view is more reliable.
4. Measurements are smoothed and compared with the calibrated baseline for that view.
5. The result is classified as `HAPPY`, `NEUTRAL`, or `SAD`.
6. The software updates the live interface, plays an alert when needed, and sends the state over USB serial.
7. The Arduino receives the state and animates the matching face on the OLED screen.

## Physical Arduino Component

The physical portion of the project uses:

- An Arduino-compatible microcontroller.
- A 128×64 SSD1306 I2C OLED display at address `0x3C`.
- A USB data cable for serial communication with the computer.
- A breadboard and jumper wires, depending on the final enclosure and wiring.

The firmware is located at:

```text
OLED Screen/OLED SCREEN OFFICIAL.cpp
```

It requires the Arduino libraries `Adafruit GFX Library` and `Adafruit SSD1306`. Connect the display to the board's power, ground, SDA, and SCL pins, upload the firmware, and keep the board connected by USB while PawSture is running.

The current Arduino firmware uses **9600 baud**, while the Python program defaults to 115200 baud. Start the program with `--baud 9600` unless you change the baud rate in the firmware:

```bash
python3 run.py --baud 9600
```

If automatic port detection does not find the board, specify it directly:

```bash
python3 run.py --port /dev/cu.usbmodem1101 --baud 9600
```

On Windows, the port will usually look like `COM3` or `COM4`. PawSture can also run without the Arduino; the software interface and posture analysis will still work.

## Requirements

- Python 3
- *A webcam*
- The packages listed in `requirements.txt`
- Optional: the Arduino and OLED hardware described above

Install the Python dependencies from the project directory:

```bash
python3 -m pip install -r requirements.txt
```

On systems where the Python command is `python`, use:

```bash
python -m pip install -r requirements.txt
```

## Running PawSture

### Standard desktop view

```bash
python3 run.py
```

### Browser dashboard

```bash
python3 run.py --web
```

The dashboard opens automatically at [http://127.0.0.1:8000](http://127.0.0.1:8000). Camera processing and the dashboard stay on the local computer.

On macOS, you can also double-click:

```text
Start PawSture Web.command
```

### Useful options

| Option | Purpose |
| --- | --- |
| `--camera 1` | Use a different camera index. |
| `--port PORT` | Select the Arduino serial port manually. |
| `--baud 9600` | Match the baud rate used by the included Arduino firmware. |
| `--manual-calibration` | Disable automatic baseline capture and calibrate manually. |
| `--web-port 8001` | Run the browser dashboard on a different port. |
| `--no-open` | Start web mode without opening the browser automatically. |

Options can be combined:

```bash
python3 run.py --web --camera 1 --port /dev/cu.usbmodem1101 --baud 9600
```

## Controls

| Control | Action |
| --- | --- |
| `C` or **Calibrate/Recalibrate** | Set a new baseline for the active front or side view. |
| `M` or **Mute Alerts** | Toggle audio feedback. |
| `Q` or **Stop Session** | End the monitoring session. |

For the most useful baseline, begin the session while sitting or standing in the upright posture you want PawSture to treat as normal. Hold that position for several clear frames while the automatic calibration completes.

## Project Structure

| Path | Description |
| --- | --- |
| `run.py` | Friendly launcher from the repository root. |
| `Main Thing/main.py` | Pose tracking, posture scoring, calibration, interfaces, alerts, and serial communication. |
| `OLED Screen/OLED SCREEN OFFICIAL.cpp` | Arduino firmware for the animated SSD1306 OLED faces. |
| `alert.wav` | Audio cue used for sustained poor-posture alerts. |
| `requirements.txt` | Python dependencies. |
| `Start PawSture Web.command` | macOS launcher for the browser dashboard. |

## Project Scope

PawSture is an educational prototype, not a medical device or diagnostic tool. Its readings depend on camera placement, lighting, visibility of body keypoints, and the quality of the user's calibration.
