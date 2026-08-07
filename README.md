# Five-Cat-PawSture

Start the program with:

```bash
python3 run.py
```

The launcher checks for the required packages and offers to install anything
missing automatically. It also uses the local `.venv` environment automatically,
so you do not need to activate it first.

To install without a prompt:

```bash
python3 run.py --install
```

To skip the local environment and use your current Python instead:

```bash
python3 run.py --no-venv
```

You can still pass monitor options after the launcher option, for example:

```bash
python3 run.py --camera 1 --port /dev/cu.usbmodem1101
```
