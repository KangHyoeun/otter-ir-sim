#!/usr/bin/env python3
"""
Demo script to test absolute coordinate transformation from LiDAR data.

This script demonstrates how to extract obstacle positions and velocities
in absolute world coordinates using the robot's position and velocity.
"""

import os
import sys
import numpy as np
import irsim

def create_absolute_coordinates_world():
    """Create a world configuration for testing absolute coordinates."""
    world_content = """world:
  size: [20, 20]

robot:
  - name: "robot_main"
    shape:
      radius: 0.3
    kinematics:
      name: "omni"
    state: [2, 2, 0]
    velocity: [1, 0.5]
    goal: [15, 15]
    behavior:
      name: "dash"
    sensors:
      - name: "lidar2d"
        range_max: 8
        number: 180
        angle_range: 6.28
        noise: true
        std: 0.1

obstacle:
  - name: "static_obstacle_1"
    shape:
      radius: 0.5
    state: [8, 5]
    static: true
    color: "blue"
    
  - name: "dynamic_obstacle_1"
    shape:
      radius: 0.4
    kinematics:
      name: "omni"
    state: [12, 8]
    velocity: [-0.5, 0.3]
    goal: [5, 12]
    behavior:
      name: "dash"
    color: "red"
    
  - name: "dynamic_obstacle_2"
    shape:
      radius: 0.3
    kinematics:
      name: "omni"
    state: [6, 12]
    velocity: [0.8, -0.2]
    goal: [14, 6]
    behavior:
      name: "dash"
    color: "green"
"""
    
    world_path = os.path.join(os.path.dirname(__file__), 'absolute_coordinates_world.yaml')
    with open(world_path, 'w') as f:
        f.write(world_content)
    
    return world_path

def test_absolute_coordinates():
    """Test absolute coordinate transformation."""
    print("=== Absolute Coordinate Transformation Demo ===")
    print("Testing LiDAR-based obstacle detection with absolute coordinates")
    print()
    
    # Create test world
    world_path = create_absolute_coordinates_world()
    print(f"Created test world: {world_path}")
    
    # Initialize environment
    env = irsim.make(world_path)
    robot = env.robot
    
    print(f"Robot initial state: {robot.state}")
    print(f"Robot initial velocity: {robot.velocity_xy}")
    print(f"World size: {env._world.width} x {env._world.height}")
    print()
    
    # Get LiDAR sensor
    lidar = robot.lidar
    if lidar is None:
        print("Error: No LiDAR sensor found on robot")
        return
    
    print("LiDAR sensor found!")
    print(f"LiDAR range_max: {lidar.range_max}")
    print(f"LiDAR number of beams: {lidar.number}")
    print()
    
    # Run simulation and test coordinate transformation
    print("Running simulation with coordinate transformation...")
    print("=" * 60)
    
    for step in range(10):
        # Step simulation
        env.step()
        
        # Get robot's current state and velocity
        robot_pos = robot.state[:2].flatten()  # [x, y]
        robot_vel = robot.velocity_xy.flatten()[:2]  # [vx, vy]
        robot_orientation = robot.state[2].item()  # theta
        
        print(f"Step {step + 1}:")
        print(f"  Robot position: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})")
        print(f"  Robot velocity: ({robot_vel[0]:.2f}, {robot_vel[1]:.2f})")
        print(f"  Robot orientation: {robot_orientation:.2f} rad")
        
        # Extract neighbors with absolute coordinates
        neighbors = lidar.extract_neighbors_detailed_with_robot_velocity(
            robot_velocity=(robot_vel[0], robot_vel[1]),
            current_time=step * 0.1,  # Simulate time progression
            world_bounds=(0, 0, env._world.width, env._world.height)
        )
        
        print(f"  Detected {len(neighbors)} neighbors:")
        
        for i, neighbor in enumerate(neighbors):
            pos = neighbor['position']
            vel = neighbor['velocity']
            radius = neighbor['radius']
            obj_type = neighbor['type']
            confidence = neighbor['confidence']
            
            print(f"    Neighbor {i+1}:")
            print(f"      Position (absolute): ({pos[0]:.2f}, {pos[1]:.2f})")
            print(f"      Velocity (absolute): ({vel[0]:.2f}, {vel[1]:.2f})")
            print(f"      Radius: {radius:.2f}")
            print(f"      Type: {obj_type}")
            print(f"      Confidence: {confidence:.2f}")
        
        print()
        
        # Check if robot reached goal
        if env.done():
            print("Robot reached goal!")
            break
    
    print("=" * 60)
    print("Demo completed successfully!")
    print()
    print("Key observations:")
    print("• Positions are now in absolute world coordinates")
    print("• Velocities are absolute velocities in world frame")
    print("• Robot's own velocity is properly accounted for")
    print("• Obstacles are classified as walls, static, or dynamic")
    print("• Coordinate transformation handles robot orientation")

def main():
    """Main function."""
    try:
        test_absolute_coordinates()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
