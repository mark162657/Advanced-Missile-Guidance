import sys
import time
import math
import re
import numpy as np
from pathlib import Path
import fastplotlib as fpl
import matplotlib.pyplot as plt

from src.missile.state import MissileState
from src.missile.profile import MissileProfile
# from src.navigation.system import NavigationSystem
from src.guidance.pathfinding_backend import Pathfinding
from src.guidance.trajectory import TrajectoryGenerator
from src.terrain.dem_loader import DEMLoader
from src.visualization.plotter import MissionPlotter


# --- Data Constants (missile profile) ---

TOMAHAWK_BLOCK_V = {
    # km/h is used instead of m/s and deg instead of radian for better understanding,
    # will be converted back into m/s in data handling
    "cruise_speed": 800,         # km/h
    "min_speed": 400,            # km/h
    "max_speed": 920,            # m/s
    "max_acceleration": 9.8,     # m/s^2
    "min_altitude": 30,          # m (AGL)
    "max_altitude": 1200,        # m (AGL)
    "max_g_force": 6.89,         # g-force
    "sustained_turn_rate": 8.0,  # deg/s
    "sustained_g_force": 2.0,    # g-force
    "evasive_turn_rate": 25.0,   # deg/s
}

# ----------------------------------------

# --- Helper Functions (for simulation) ---
def parse_gps_input(gps_input: str) -> tuple[float, float]:
    """
    Parses a GPS string into (latitude, longitude) floats.
    Supported formats:
        - "22.5, -120.3      (Comma separated, +/-)"
        - "22.5 -120.3       (Space separated, +/-)"
        - "22.5 N, 120.3 W   (Directional, NSWE)"

    We then check if the user chose to input with NSWE for directions, if so, then we turn into +/- format for
    rasterio to recognize our format.

    Args:
        - gps_input: GPS string to parse
    """

    clean_str = gps_input.strip().upper().replace(",", " ") # unifying format, remove space, comma, and turn upper case

    # Extract float-like numbers
    numbers = [float(x) for x in re.findall(r'-?\d+\.?\d*', clean_str)]

    if len(numbers) != 2:
        raise ValueError(f"Could not find exactly 2 coordinates for the input{gps_input}. ")

    lat, lon = numbers[0], numbers[1]

    # Check for 'S' (South) -> Negate Latitude
    if 'S' in clean_str:
        lat = -abs(lat)
    # Check for 'N' (North) -> Positive Latitude
    if 'N' in clean_str:
        lat = abs(lat)

    # Check for 'W' (West) -> Negate Longitude
    if 'W' in clean_str:
        lon = -abs(lon)
    # Check for 'E' (East) -> Positive Longitude
    if 'E' in clean_str:
        lon = abs(lon)

    return lat, lon

def get_path_distance(trajectory: list[float, float, float]) -> float:
    """
    We cannot calculate flight distance by just counting trajectory points or multiplies total pixels by x meter
    grid resolution (30 for our sample dem file). The B-Spline smoothing algorithm creates a variable density of
    non-equidistant waypoints. It clusters them on curves and spreads them on straight sections. This separates the array
    length from physical metric distance.

    Instead, we calculate haversine distance for every neighbouring points on the trajectory
    then sum up.

    Return:
        - distance in meter
    """

    if not trajectory or len(trajectory) < 2:
        return 0.0

    total_dist = 0.0
    r = 637100 # Earth radius in meter

    # Main loop for iterating and accumulating total distance
    for i in range(len(trajectory) - 1):

        lat1, lon1 = trajectory[i][0], trajectory[i][1]
        lat2, lon2 = trajectory[i+1][0], trajectory[i+1][1]

        # Convert to radians
        lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
        lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

        dlon = lon2_r - lon1_r
        dlat = lat2_r - lat1_r

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        total_dist += r * c

        return total_dist
# -----------------------------------------

# --- Launchpad Menu Display / Data Handling ---
def clear_screen():
    """ Clear console view through new lines"""
    print("\n" * 5)

def print_header(title):
    """ Print a header to the console """
    print("\n" + "=" * 65)
    print(f"   {title.upper()}   ")
    print("=" * 65 + "\n")

def prompt_gps_loop(prompt_text):
    """
    UI Wrapper that loops until valid input is received using the
    existing 'parse_gps_input' function.
    """
    while True:
        raw = input(f"   >> {prompt_text} (or 'b' to back): ").strip()

        if raw.lower() == 'b':
            return None

        try:
            # USE EXISTING FUNCTION HERE
            return parse_gps_input(raw)
        except ValueError:
            print("      [!] Format Error. Accepted formats:")
            print("          - 22.5, 120.3")
            print("          - 22.5 N, 120.3 E")

def create_missile_profile(data: dict) -> MissileProfile:
    """
    Receive the dictionary data (for example, like the TOMAHAWK_BlOCK_V you can see from top.
    Convert the unit km/h to standard m/s, degree to radian, then return as MissileProfile object.

    Arg:
        - data: a dictionary that contains each parameters for MissileProfile object.
    """

    KMH_TO_MS = 1.0 / 3.6
    DEG_TO_RAD = math.pi / 180.0

    return MissileProfile(
        cruise_speed = data["cruise_speed"] * KMH_TO_MS,
        min_speed = data["min_speed"] * KMH_TO_MS,
        max_speed = data["max_speed"] * KMH_TO_MS,
        max_acceleration = data["max_acceleration"],
        min_altitude = data["min_altitude"],
        max_altitude = data["max_altitude"],
        max_g_force = data["max_g_force"],

        sustained_turn_rate = data["sustained_turn_rate"] * DEG_TO_RAD,
        sustained_g_force = data["sustained_g_force"],

        evasive_turn_rate = data["evasive_turn_rate"] * DEG_TO_RAD,
    )

def get_custom_profile（
# ----------------------------------------------

class MissileSimulation:
    def __init__(self, start_gps: tuple, target_gps: tuple, dem_name: str):
        """
        Subsystem initialisation
        """
        self.start_gps = start_gps
        self.target_gps = target_gps
        self.dem_name = dem_name
        
        # Construct full path to DEM file
        from pathlib import Path
        dem_path = Path(__file__).parent / "data" / "dem" / self.dem_name

        self.dem_loader = DEMLoader(dem_path)

        init_z = self.dem_loader.get_elevation(*self.start_gps)

        # Initiate pathfinding through Pyhton backend (as an intermediate between CPP and simulation)
        self.pf = Pathfinding(self.dem_name)
        self.trajectory = TrajectoryGenerator(self.pf.engine, self.pf.dem_loader)

        # Define initial state
        self.missile_state = MissileState(x=0, y=0, z=init_z,
                                    vx=0, vy=0, vz=0,
                                    pitch=0, roll=0, heading=0,
                                    lon=self.start_gps[0], lat=self.start_gps[1], altitude=init_z,
                                    time=0, gps_valid=True, tercom_active=False, ins_calibrated=False,
                                    distance_traveled=0, distance_to_target=get_path_distance(self.trajectory))


    def run(self):
        pass




# --- MAIN ENTRY POINT ---
if __name__ == "__main__":

    # 1. System Configuration State
    config = {
        "dem_name": "merged_dem_sib_N54_N59_E090_E100.tif",
        "start_gps": None,
        "target_gps": None,
        "missile_type": TOMAHAWK_BLOCK_V.copy()
    }

    while True:
        clear_screen()
        print_header("GNC Simulation Console - Main Menu")

        # Format status strings for display
        s_gps_str = f"{config['start_gps']}" if config['start_gps'] else "[ NOT SET ]"
        t_gps_str = f"{config['target_gps']}" if config['target_gps'] else "[ NOT SET ]"

        # Main Menu Options
        print(f"   1. Set Terrain File      [{config['dem_name']}]")
        print(f"   2. Set Coordinates       [Start: {s_gps_str}]")
        print(f"                            [Target: {t_gps_str}]")
        print(f"   3. Missile Configuration [{config['missile_type']}]")
        print("   -----------------------------------------------------------------")
        print(f"   4. INITIALIZE & LAUNCH")
        print(f"   5. Exit Console")
        print("\n" + "=" * 65)

        choice = input("\n   >> Select Option [1-5]: ").strip()

        # --- OPTION 1: SET DEM ---
        if choice == '1':
            print_header("Terrain Configuration")
            print(f"   Current File: {config['dem_name']}")
            print("   Enter new filename (must be located in src/data/dem/).")

            new_name = input("\n   >> Filename (ENTER to keep current): ").strip()
            if new_name:
                config['dem_name'] = new_name
                print("      [OK] Terrain filename updated.")
            else:
                print("      [INFO] No change made.")
            time.sleep(1)

        # --- OPTION 2: SET COORDINATES ---
        elif choice == '2':
            while True:
                clear_screen()
                print_header("Coordinate Configuration")

                curr_start = config['start_gps'] if config['start_gps'] else "NOT SET"
                curr_target = config['target_gps'] if config['target_gps'] else "NOT SET"

                print(f"   1. Set Start Location   [{curr_start}]")
                print(f"   2. Set Target Location  [{curr_target}]")
                print(f"   3. Return to Main Menu")
                print("\n" + "=" * 65)

                sub_choice = input("\n   >> Select Option [1-3]: ").strip()

                if sub_choice == '1':
                    print("\n   --- Input Launch Site ---")
                    # Loops until valid using existing parse function
                    res = prompt_gps_loop("Enter GPS")
                    if res: config['start_gps'] = res

                elif sub_choice == '2':
                    print("\n   --- Input Target Site ---")
                    res = prompt_gps_loop("Enter GPS")
                    if res: config['target_gps'] = res

                elif sub_choice == '3':
                    break

        # --- OPTION 3: MISSILE PROFILE ---
        elif choice == '3':
            while True:
                clear_screen()
                print_header("Missile Profile Selection")
                print(f"   Current Profile: {config['missile_type']}\n")

                print("   1. Load Custom Profile (Function Pending)")
                print("   2. Load Preset: Tomahawk Block IV")
                print("   3. Return to Main Menu")
                print("\n" + "=" * 65)

                sub_choice = input("\n   >> Select Option [1-3]: ").strip()

                if sub_choice == '1':
                    print("\n      [INFO] Custom profile loader is under development.")
                    time.sleep(1.5)
                elif sub_choice == '2':
                    config['missile_type'] = "Tomahawk Block IV (Preset)"
                    print("\n      [OK] Preset Loaded.")
                    time.sleep(1)
                    break
                elif sub_choice == '3':
                    break

        # --- OPTION 4: LAUNCH ---
        elif choice == '4':
            # Pre-flight checks
            if not config['start_gps'] or not config['target_gps']:
                print("\n      [ERROR] Aborted: Start and Target coordinates are required.")
                input("      Press ENTER to continue...")
                continue

            print_header("Mission Pre-Flight Check")
            print(f"   Terrain: {config['dem_name']}")
            print(f"   Profile: {config['missile_type']}")
            print(f"   Launch:  {config['start_gps']}")
            print(f"   Target:  {config['target_gps']}")
            print("-" * 65)

            confirm = input("\n   >> Confirm Launch Sequence? (y/n): ").strip().lower()
            if confirm == 'y':
                try:
                    print("\n   [INFO] Initializing Simulation Environment...")

                    # Instantiate the Main Simulation Class
                    sim = MissileSimulation(
                        config['start_gps'],
                        config['target_gps'],
                        config['dem_name']
                    )

                    print("   [INFO] Running Mission...")
                    sim.run()

                    input("\n   [COMPLETE] Simulation Finished. Press ENTER for Menu...")
                except Exception as e:
                    print(f"\n   [FATAL ERROR] Simulation Failed: {e}")
                    # Optional: Print full traceback for debugging
                    import traceback

                    traceback.print_exc()
                    input("   Press ENTER to return to menu...")
            else:
                print("      [INFO] Mission Aborted.")
                time.sleep(1)

        # --- OPTION 5: QUIT ---
        elif choice == '5':
            print("\n   [INFO] Terminating Session. Goodbye.")
            sys.exit()
