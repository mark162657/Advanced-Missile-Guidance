# Web Control Terminal

A browser-based mission-control frontend for the missile-guidance simulator. It
provides Planning, Mission Control, and Final Report workspaces with a tactical
map, lightweight 3D viewer, telemetry instruments, replay, and live simulation.

The browser UI is plain ES-module JavaScript and CSS. Node.js and npm are not
required.

## Prerequisites

- 64-bit Python 3.10 or newer. Use the same interpreter for installation,
  compiling the pathfinder, and launching the frontend.
- CMake 3.15 or newer.
- A C++14 compiler:
  - Windows: Visual Studio Build Tools with **Desktop development with C++** and
    a Windows SDK.
  - macOS: Xcode Command Line Tools (`xcode-select --install`).
  - Linux: GCC or Clang and Python development headers.
- A GeoTIFF DEM in `data/dem/` and a missile profile in `data/missiles/`.

## Python modules

Installing `frontend/requirements.txt` installs every direct module needed by
the web terminal and live simulator:

- `fastapi` - HTTP and WebSocket application.
- `pydantic` - API request validation.
- `uvicorn[standard]` - ASGI server, reload watcher, and WebSocket support.
- `numpy` - simulation, navigation, and terrain arrays.
- `rasterio` - GeoTIFF DEM reading and coordinate transforms.
- `scipy` - B-spline trajectory generation.
- `matplotlib` - terrain colour and hillshade utilities imported by the DEM
  loader.
- `pybind11` - Python bindings and CMake integration for the C++ pathfinder.

Their transitive packages are installed automatically. Node.js and npm are not
part of the frontend toolchain.

## Create an isolated Python environment

Run these commands from the project root.

### Windows PowerShell

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r frontend\requirements.txt
```

If PowerShell blocks activation, activation is optional; use
`.\.venv\Scripts\python.exe` in place of `python` for every command below.

### macOS or Linux

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r frontend/requirements.txt
```

## Build the native pathfinder

The generated module is interpreter- and platform-specific (`.pyd` on Windows,
`.so` on macOS/Linux). Do not copy it between operating systems or Python minor
versions; rebuild it locally.

### Windows PowerShell

Make sure `cmake` is on `PATH`. A normal PowerShell terminal works when CMake
and Visual Studio Build Tools are installed correctly; otherwise use the
**x64 Native Tools Command Prompt for Visual Studio**.

```powershell
$pythonExe = python -c "import sys; print(sys.executable)"
cmake -S src\missile\planning\cpp -B src\missile\planning\cpp\build "-DPython3_EXECUTABLE=$pythonExe"
cmake --build src\missile\planning\cpp\build --config Release
```

### macOS or Linux

```bash
python_exe="$(python -c 'import sys; print(sys.executable)')"
cmake -S src/missile/planning/cpp -B src/missile/planning/cpp/build -DPython3_EXECUTABLE="$python_exe"
cmake --build src/missile/planning/cpp/build --config Release
```

CMake copies the compiled module into `src/missile/planning/`, where Python can
import it. Confirm the import with:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from missile.planning import missile_backend; print('pathfinder ready')"
```

## Run the frontend

```bash
python frontend/run.py             # http://127.0.0.1:8000
python frontend/run.py --reload    # development auto-reload
```

On Windows without environment activation:

```powershell
.\.venv\Scripts\python.exe frontend\run.py
```

Everything is served from `http://127.0.0.1:8000`.

## Troubleshooting

### `ModuleNotFoundError`

Install `frontend/requirements.txt` with the exact Python interpreter used to
launch `frontend/run.py`. IDEs such as PyCharm may select a different interpreter
from the terminal.

### C++ pathfinding engine is unavailable

Rebuild the extension with the same active Python environment. On Windows,
include `--config Release`. Check that a `missile_backend*.pyd` or
`missile_backend*.so` exists under `src/missile/planning/`.

### CMake or compiler not found

- Windows: install CMake and Visual Studio Build Tools, then open a new terminal.
- macOS: install CMake and Xcode Command Line Tools.
- Linux: install CMake, a compiler, and your distribution's Python development
  package.

### Planning hangs or the process runs out of memory

Merged DEMs are very large. The simulator currently loads full terrain arrays
more than once during live operation. Start with the smallest available DEM to
validate setup, then use large tiles only on a machine with sufficient RAM.

The Planning screen selects the smallest available tile initially and warns when
a selected terrain exceeds 500 million pixels. This behaviour is identical on
Windows, macOS, and Linux.

## Architecture

```text
frontend/
|-- run.py                 FastAPI/uvicorn launcher
|-- requirements.txt      Complete direct Python dependencies
|-- backend/
|   |-- app.py             REST API and live WebSocket
|   |-- bootstrap.py       Adds ../src to Python's import path
|   |-- native_backend.py  Cross-platform native-module preflight
|   |-- dem_service.py     Downsampled DEM grids and elevation queries
|   |-- catalog.py         Profiles, recorded flights, and results
|   |-- frames.py          Shared replay/live telemetry frame
|   |-- planning.py        C++ A* and B-spline route planning
|   `-- live_runner.py     Non-interactive live simulation adapter
`-- web/
    |-- index.html
    |-- css/
    `-- js/
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/dems` | DEM tiles and bounds |
| GET | `/api/dems/{name}/grid` | Downsampled elevation grid |
| GET | `/api/dems/{name}/elevation?lat=&lon=` | Point elevation |
| GET | `/api/profiles` | Missile profiles |
| GET | `/api/missions` | Recorded-flight summaries |
| GET | `/api/missions/{id}` | Full telemetry and verdict |
| GET | `/api/results` | Saved verdicts |
| POST | `/api/plan` | A* and spline route planning |
| WS | `/ws/live` | Live simulation telemetry |

## Notes and limits

- Recorded-flight replay does not require the native pathfinder.
- Live planning and live simulation require the compiled native pathfinder.
- The map hillshade is derived from the local DEM, so it works offline.
- Recorded telemetry includes kinematic state; detailed autopilot/PID signals
  are available only on the live stream.
