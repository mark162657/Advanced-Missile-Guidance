# Missile Guidance

A Python simulation and mission-planning project that models cruise-missile
guidance, navigation, and control over digital elevation model (DEM) terrain. It
combines terrain-aware route planning with a 3-DoF flight model, noisy sensors,
navigation estimation, guidance, control, telemetry, and a browser-based
mission-control terminal.

> **Status: working, under active development.** This is an experimentation and
> learning project, not production or hardware-fidelity software.

## Highlights

- GeoTIFF terrain loading, elevation queries, and coordinate conversion.
- C++ A* terrain pathfinding with B-spline trajectory smoothing.
- 3-DoF point-mass physics with RK4 integration, propulsion, wind, and
  turbulence.
- INS, GPS, IMU, TERCOM, barometric/radar altitude, and Kalman estimation.
- Path following, terminal guidance, PID autopilot, and flight sequencing.
- Interactive CLI simulation with telemetry and mission-result output.
- Browser planning, live monitoring, replay, and final reports.

## Supported platforms

- Windows 10/11, 64-bit.
- macOS, Intel or Apple Silicon.
- Linux should work with equivalent Python and compiler packages, although the
  primary setup instructions below focus on Windows and macOS.

Compiled pathfinder modules are platform- and Python-version-specific. Windows
produces a `.pyd`; macOS and Linux produce a `.so`. Never copy a compiled module
between operating systems or Python minor versions.

## Required software

### All platforms

- 64-bit Python 3.10 or newer.
- CMake 3.15 or newer.
- A C++14-compatible compiler.
- A GeoTIFF DEM under `data/dem/`.
- A missile profile under `data/missiles/`.

### Windows

- Python from python.org, Anaconda, or another 64-bit distribution.
- Visual Studio Build Tools with:
  - **Desktop development with C++**
  - MSVC C++ build tools
  - Windows 10 or 11 SDK
- CMake added to `PATH`, or the CMake bundled with Visual Studio.

### macOS

- Xcode Command Line Tools:

  ```bash
  xcode-select --install
  ```

- CMake, for example through Homebrew:

  ```bash
  brew install cmake
  ```

## Python modules

Install `frontend/requirements.txt` to get every direct module required by the
simulation and web terminal:

| Module | Purpose |
|---|---|
| `fastapi` | HTTP and WebSocket backend |
| `pydantic` | API request validation |
| `uvicorn[standard]` | ASGI server, reload, and WebSockets |
| `numpy` | Simulation, navigation, and terrain arrays |
| `rasterio` | GeoTIFF DEM loading and geographic transforms |
| `scipy` | B-spline trajectory generation |
| `matplotlib` | Terrain colour and hillshade utilities |
| `pybind11` | C++ pathfinder Python bindings and CMake integration |

Additional development or optional modules:

| Module | When it is needed |
|---|---|
| `pytest` | Automated tests |
| `cryptography` | `src/launcher/terminal.py` encryption helper |
| `fastplotlib` | Manual DEM/pathfinding visualizer scripts under `tests/` |

Transitive packages such as Starlette, AnyIO, WebSockets, GDAL support bundled
by Rasterio wheels, and Uvicorn's standard extras are installed automatically.
Node.js and npm are not required.

## Windows setup

Run PowerShell from the project root.

### Recommended: isolated environment

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r frontend\requirements.txt
```

If activation is blocked, use `.\.venv\Scripts\python.exe` instead of `python`
in every subsequent command.

### Existing Python or PyCharm interpreter

Install with the exact interpreter used by the run configuration:

```powershell
& "C:\path\to\python.exe" -m pip install -r "frontend\requirements.txt"
```

Do not install packages with one Python and launch with another. Check the
active interpreter with:

```powershell
python -c "import sys; print(sys.executable)"
```

### Build the Windows pathfinder

```powershell
$pythonExe = python -c "import sys; print(sys.executable)"
cmake -S src\missile\planning\cpp -B src\missile\planning\cpp\build "-DPython3_EXECUTABLE=$pythonExe"
cmake --build src\missile\planning\cpp\build --config Release
```

If `cmake` or MSVC is not found, use the **x64 Native Tools Command Prompt for
Visual Studio** or correct the CMake/Build Tools installation.

Confirm the binary imports with the same Python:

```powershell
python -c "import sys; sys.path.insert(0, 'src'); from missile.planning import missile_backend; print('pathfinder ready')"
```

### Run on Windows

Web terminal:

```powershell
python frontend\run.py
```

Interactive CLI simulation:

```powershell
$env:PYTHONPATH = "src"
python src\main.py
```

## macOS setup

Run Terminal from the project root.

### Create the environment

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r frontend/requirements.txt
```

### Build the macOS pathfinder

```bash
python_exe="$(python -c 'import sys; print(sys.executable)')"
cmake -S src/missile/planning/cpp -B src/missile/planning/cpp/build -DPython3_EXECUTABLE="$python_exe"
cmake --build src/missile/planning/cpp/build --config Release
```

Confirm the binary imports:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from missile.planning import missile_backend; print('pathfinder ready')"
```

### Run on macOS

Web terminal:

```bash
python frontend/run.py
```

Interactive CLI simulation:

```bash
PYTHONPATH=src python src/main.py
```

## Web terminal

Open `http://127.0.0.1:8000` after starting `frontend/run.py`. Development mode
with automatic reload is available on both platforms:

```bash
python frontend/run.py --reload
```

The browser application has no Node.js build step. See
[frontend/README.md](frontend/README.md) for its architecture and API.

## Large DEM warning

Merged DEMs can contain hundreds of millions or billions of pixels. Planning
and live simulation currently load full terrain arrays more than once, so large
tiles can require 10+ GiB RAM and may appear to hang on any operating system.

Start with the smallest available DEM (`srtm_43_02.tif` in the current data
set). The Planning screen selects the smallest tile initially and displays a
warning when a selected DEM exceeds 500 million pixels.

## Testing

Install the test dependency:

```bash
python -m pip install pytest
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests\navigation tests\test_controls_guidance.py -v
```

macOS/Linux:

```bash
PYTHONPATH=src python -m pytest tests/navigation tests/test_controls_guidance.py -v
```

Some manual scripts require `fastplotlib`, local DEM files, or the compiled C++
pathfinder. The test suite is under active development and may contain tests
that target older simulator APIs.

## Project layout

```text
src/
|-- main.py                 Interactive simulation runner
|-- terrain/                DEM loading, queries, and coordinates
|-- missile/
|   |-- navigation/         INS, GPS, TERCOM, and Kalman estimation
|   |-- planning/           C++ pathfinder and trajectory generation
|   |-- guidance/           Path following and terminal guidance
|   |-- controls/           PID control, autopilot, and flight computer
|   `-- datalink/           Datalink scaffolding
|-- simulation/
|   |-- physics/            Dynamics, atmosphere, propulsion, and weather
|   |-- sensors/            IMU, GPS, radar, and barometric sensors
|   `-- result/             Flight logs and mission results
`-- launcher/               Supporting launcher utilities

frontend/                   FastAPI backend and browser application
data/
|-- dem/                    GeoTIFF terrain files
|-- missiles/               Missile profile JSON files
|-- logs/                   Generated telemetry CSV files
`-- results/                Generated mission-result JSON files
tests/                      Tests, benchmarks, and manual visualizers
```

## How it works

- The physics model advances the true position and velocity.
- Simulated sensors observe truth with noise and uncertainty.
- The navigation stack estimates position from those measurements.
- Guidance converts the planned route into flight setpoints.
- Control converts setpoints into plant inputs.

The main coordinate systems are geographic latitude/longitude/altitude, DEM
pixel coordinates, and local ENU metres. Positive ENU `up` is vertical.

## Scope and limitations

The simulator is an algorithm, navigation, guidance, and pathfinding showcase.
The vehicle model is a 3-DoF point mass; it does not model airframe control
surfaces or full attitude dynamics. Some modules and integrations remain
incomplete or need additional validation.

## Credits

The frontend was created with Fable 5 and Claude Opus 4.8 High. The project also
uses open-source Python packages, CMake, pybind11, and GeoTIFF/SRTM-style terrain
data.

## License

No license has been specified for this repository yet.
