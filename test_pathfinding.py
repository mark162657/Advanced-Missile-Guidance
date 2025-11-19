"""Test script for pathfinding 3D distance calculation."""

from src.guidance.pathfinding import Pathfinding
from src.terrain.dem_loader import DEMLoader

if __name__ == "__main__":
    # Test 3D Distance calculation
    pathfinder = Pathfinding()

    # # Example 1
    # loc1_1 = (55.5, 92.3)
    # loc2_1 = (58.2, 97.7)
    
    # dist = pathfinder.get_surfcae_distance(loc1_1, loc2_1)

    # print(f"\nDistance from {loc1_1} to {loc2_1}: {dist:.2f} meters\n")

    
    # # Example 2
    # loc1_2 = (54.219, 91.405)
    # loc2_2 = (58.843, 97.712)
    
    # dist2 = pathfinder.get_surfcae_distance(loc1_2, loc2_2)

    # print(f"\nDistance from {loc1_2} to {loc2_2}: {dist2:.2f} meters\n")