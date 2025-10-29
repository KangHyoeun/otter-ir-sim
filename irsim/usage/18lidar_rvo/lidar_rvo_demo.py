import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import irsim
from irsim.lib.algorithm.lidar_processing import LidarNeighborDetector, extract_rvo_neighbors_simple


def create_lidar_rvo_world():
    """Create a test world with multiple robots and obstacles for LiDAR RVO testing."""
    world_config = """
# LiDAR RVO Test World
world:
  size: [20, 20]
  collision_mode: "reactive"

# Main robot with LiDAR
robot:
  - name: "robot_main"
    shape:
      name: "circle"
      radius: 0.3
    kinematics:
      name: "omni"
    state: [2, 2, 0]
    velocity: [0, 0]
    goal: [15, 15, 0]
    behavior:
      name: "dash"
    sensors:
      - name: "lidar2d"
        range_max: 12
        number: 200
        angle_range: 6.28
        noise: false
    color: "blue"

# No static obstacles - only dynamic ones for velocity testing

# Additional robots (moving obstacles)
  - name: "robot2"
    shape:
      name: "circle"
      radius: 0.4
    kinematics:
      name: "omni"
    state: [5, 3, 0]
    velocity: [0.3, 0.2]
    goal: [8, 8, 0]
    behavior:
      name: "dash"
    color: "purple"
    
  - name: "robot3"
    shape:
      name: "circle"
      radius: 0.35
    kinematics:
      name: "omni"
    state: [3, 5, 0]
    velocity: [-0.2, 0.4]
    goal: [6, 9, 0]
    behavior:
      name: "dash"
    color: "brown"
"""
    
    # Write config to file
    with open('/home/hyo/ir-sim/irsim/usage/18lidar_rvo/lidar_rvo_world.yaml', 'w') as f:
        f.write(world_config)
    
    return '/home/hyo/ir-sim/irsim/usage/18lidar_rvo/lidar_rvo_world.yaml'


def demo_simple_lidar_rvo():
    """Demonstrate simple LiDAR to RVO neighbor extraction."""
    print("=== Simple LiDAR to RVO Demo ===\n")
    
    # Create test environment
    print("1. Creating test environment...")
    world_file = create_lidar_rvo_world()
    print(f"   Created world file: {world_file}")
    
    # Create IR-SIM environment
    print("\n2. Initializing IR-SIM environment...")
    try:
        env = irsim.make(world_file)
        print("   Environment created successfully")
    except Exception as e:
        print(f"   Error creating environment: {e}")
        return
    
    # Get the main robot with LiDAR
    robot = None
    for obj in env.objects:
        if hasattr(obj, 'lidar') and obj.lidar is not None and obj.role == "robot":
            robot = obj
            break
    
    if robot is None:
        print("   Error: No robot with LiDAR found")
        return
    
    print(f"   Found robot with LiDAR: {robot.role}")
    
    # Create advanced detector for velocity estimation
    print("\n3. Creating advanced LiDAR detector with velocity estimation...")
    detector = LidarNeighborDetector(
        eps=0.6,        # Larger clusters to avoid splits
        min_samples=5,   # More points per cluster
        min_radius=0.3, # Filter very small detections
        max_radius=1.0, # Filter very large detections
        velocity_history_length=5
    )
    
    # Run simulation and extract neighbors with velocity estimation
    print("\n4. Running simulation with velocity estimation...")
    
    for step in range(15):
        env.step()
        current_time = step * 0.1  # Assume 10Hz simulation
        
        # Extract neighbors with velocity estimation
        neighbors = detector.extract_neighbors(
            robot.lidar.range_data, 
            robot.lidar.angle_list, 
            current_time
        )
        
        print(f"   Step {step+1}: Detected {len(neighbors)} neighbors")
        
        if len(neighbors) > 0:
            for i, neighbor in enumerate(neighbors):
                x, y = neighbor['position']
                vx, vy = neighbor['velocity']
                radius = neighbor['radius']
                confidence = neighbor['confidence']
                print(f"     Neighbor {i+1}: pos=({x:.2f}, {y:.2f}), "
                      f"vel=({vx:.2f}, {vy:.2f}), radius={radius:.2f}, "
                      f"confidence={confidence:.2f}")
        
        # Convert to RVO format
        rvo_neighbors = detector.neighbors_to_rvo_format(neighbors)
        if len(rvo_neighbors) > 0:
            print(f"     RVO format: {len(rvo_neighbors)} neighbors ready for collision avoidance")
    
    print(f"\n=== Simple Demo Complete ===")
    print(f"Successfully demonstrated LiDAR to RVO neighbor extraction!")


def demo_advanced_lidar_rvo():
    """Demonstrate advanced LiDAR to RVO with velocity estimation."""
    print("=== Advanced LiDAR to RVO Demo ===\n")
    
    # Create test environment
    print("1. Creating test environment...")
    world_file = create_lidar_rvo_world()
    
    # Create IR-SIM environment
    print("\n2. Initializing IR-SIM environment...")
    try:
        env = irsim.make(world_file)
        print("   Environment created successfully")
    except Exception as e:
        print(f"   Error creating environment: {e}")
        return
    
    # Get the main robot with LiDAR
    robot = None
    for obj in env.objects:
        if hasattr(obj, 'lidar') and obj.lidar is not None and obj.role == "robot":
            robot = obj
            break
    
    if robot is None:
        print("   Error: No robot with LiDAR found")
        return
    
    print(f"   Found robot with LiDAR: {robot.role}")
    
    # Create advanced detector
    print("\n3. Creating advanced LiDAR neighbor detector...")
    detector = LidarNeighborDetector(
        eps=0.3,
        min_samples=3,
        min_radius=0.2,
        max_radius=1.5,
        velocity_history_length=5
    )
    
    # Run simulation with velocity estimation
    print("\n4. Running simulation with velocity estimation...")
    
    for step in range(15):
        env.step()
        current_time = step * 0.1  # Assume 10Hz simulation
        
        # Extract neighbors with velocity estimation
        neighbors = detector.extract_neighbors(
            robot.lidar.range_data, 
            robot.lidar.angle_list, 
            current_time
        )
        
        print(f"   Step {step+1}: Detected {len(neighbors)} neighbors")
        
        if len(neighbors) > 0:
            for i, neighbor in enumerate(neighbors):
                x, y = neighbor['position']
                vx, vy = neighbor['velocity']
                radius = neighbor['radius']
                confidence = neighbor['confidence']
                print(f"     Neighbor {i+1}: pos=({x:.2f}, {y:.2f}), "
                      f"vel=({vx:.2f}, {vy:.2f}), radius={radius:.2f}, "
                      f"confidence={confidence:.2f}")
        
        # Convert to RVO format
        rvo_neighbors = detector.neighbors_to_rvo_format(neighbors)
        
        if len(rvo_neighbors) > 0:
            print(f"     RVO format: {len(rvo_neighbors)} neighbors ready for collision avoidance")
    
    print(f"\n=== Advanced Demo Complete ===")
    print(f"Successfully demonstrated advanced LiDAR to RVO with velocity estimation!")


def demo_integration_with_rvo():
    """Demonstrate full integration with RVO collision avoidance."""
    print("=== Full RVO Integration Demo ===\n")
    
    # Create test environment
    print("1. Creating test environment...")
    world_file = create_lidar_rvo_world()
    
    # Create IR-SIM environment
    print("\n2. Initializing IR-SIM environment...")
    try:
        env = irsim.make(world_file)
        print("   Environment created successfully")
    except Exception as e:
        print(f"   Error creating environment: {e}")
        return
    
    # Get the main robot with LiDAR
    robot = None
    for obj in env.objects:
        if hasattr(obj, 'lidar') and obj.lidar is not None and obj.role == "robot":
            robot = obj
            break
    
    if robot is None:
        print("   Error: No robot with LiDAR found")
        return
    
    print(f"   Found robot with LiDAR: {robot.role}")
    
    # Import RVO algorithm
    from irsim.lib.algorithm.rvo import reciprocal_vel_obs
    
    # Run simulation with RVO collision avoidance
    print("\n3. Running simulation with RVO collision avoidance...")
    
    for step in range(20):
        env.step()
        
        # Extract neighbors from LiDAR
        rvo_neighbors = robot.lidar.extract_rvo_neighbors()
        
        if len(rvo_neighbors) > 0:
            print(f"   Step {step+1}: Using {len(rvo_neighbors)} LiDAR-detected neighbors for RVO")
            
            # Create RVO behavior with LiDAR neighbors
            rvo_behavior = reciprocal_vel_obs(
                state=robot.rvo_state,
                obs_state_list=rvo_neighbors,
                vxmax=1.5,
                vymax=1.5,
                acce=0.5,
                factor=1.0
            )
            
            # Calculate collision-free velocity
            safe_velocity = rvo_behavior.cal_vel("rvo")
            print(f"     Safe velocity from RVO: {safe_velocity}")
        else:
            print(f"   Step {step+1}: No neighbors detected, robot can move freely")
    
    print(f"\n=== Full Integration Demo Complete ===")
    print(f"Successfully demonstrated LiDAR-based RVO collision avoidance!")


if __name__ == "__main__":
    print("LiDAR to RVO Neighbors Demo\n")
    print("Running simple demo automatically...")
    demo_simple_lidar_rvo()
