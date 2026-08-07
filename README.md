# Five-Cat-PawSture

Start the program with:

```bash
python3 run.py
```

On macOS, you can start the browser version without opening VS Code or typing a
command by double-clicking:

```text
Start PawSture Web.command
```

That starts the local PawSture server and opens the web console automatically.

To use the browser console instead of the OpenCV window:

```bash
python3 run.py --web
```

The browser should open automatically. If it does not, open:

```text
http://127.0.0.1:8000
```

If Python says a package is missing, install the project requirements manually:

```bash
python3 -m pip install -r requirements.txt
```

You can still pass monitor options after the launcher option, for example:

```bash
python3 run.py --camera 1 --port /dev/cu.usbmodem1101
```

The web console supports the same monitor options:

```bash
python3 run.py --web --camera 1 --port /dev/cu.usbmodem1101
```

Front and side baselines are captured automatically after a few good tracking
frames in each view. Use the Calibrate/Recalibrate button, or press `C`, only
when you want to override the automatic baseline.

To use the older manual calibration flow:

```bash
python3 run.py --manual-calibration
```
