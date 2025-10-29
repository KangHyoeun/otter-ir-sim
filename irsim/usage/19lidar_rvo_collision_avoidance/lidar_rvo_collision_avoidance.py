#!/usr/bin/env python3
"""
LiDAR-Based RVO Collision Avoidance Demo

This demo shows how to use LiDAR sensor data to implement RVO collision avoidance
without any prior knowledge of obstacle positions or velocities.

Author: AI Assistant
Date: 2024
"""

import irsim
import numpy as np
import matplotlib.pyplot as plt
from irsim.lib.algorithm.rvo import reciprocal_vel_obs


def create_lidar_rvo_world():
    """Create a YAML configuration for LiDAR-based RVO collision avoidance."""
    yaml_content = """
# LiDAR-Based RVO Collision Avoidance World
world:
  size: [20, 20]
  collision_mode: "reactive"

# Main robot with LiDAR and RVO behavior
robot:
  - name: "robot_main"
    shape:
      name: "circle"
      radius: 0.3
    kinematics:
      name: "omni"
    state: [2, 2, 0]
    velocity: [0, 0]
    goal: [18, 18, 0]
    behavior:
      name: "dash"
    sensors:
      - name: "lidar2d"
        range_max: 8
        number: 180
        angle_range: 6.28
        noise: false
    color: "blue"

# Moving obstacles (unknown to the robot)
  - name: "robot2"
    shape:
      name: "circle"
      radius: 0.4
    kinematics:
      name: "omni"
    state: [8, 4, 0]
    velocity: [0.2, 0.3]
    goal: [4, 12, 0]
    behavior:
      name: "dash"
    color: "red"
    
  - name: "robot3"
    shape:
      name: "circle"
      radius: 0.35
    kinematics:
      name: "omni"
    state: [6, 8, 0]
    velocity: [-0.3, 0.2]
    goal: [12, 6, 0]
    behavior:
      name: "dash"
    color: "green"

# Static obstacles
obstacle:
  - name: "obs1"
    shape:
      name: "circle"
      radius: 0.5
    state: [10, 10, 0]
    color: "orange"
    static: true
    
  - name: "obs2"
    shape:
      name: "circle"
      radius: 0.4
    state: [14, 6, 0]
    color: "purple"
    static: true
"""
    
    import os
    yaml_path = os.path.join(os.path.dirname(__file__), 'lidar_rvo_world.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print("Created LiDAR RVO collision avoidance world configuration")


def demo_lidar_rvo_collision_avoidance():
    """Demonstrate LiDAR-based RVO collision avoidance."""
    print("=== LiDAR-Based RVO Collision Avoidance Demo ===")
    
    # Create world configuration
    print("\n1. Creating world configuration...")
    create_lidar_rvo_world()
    
    # Initialize environment
    print("\n2. Initializing IR-SIM environment...")
    import os
    yaml_path = os.path.join(os.path.dirname(__file__), 'lidar_rvo_world.yaml')
    env = irsim.make(yaml_path)
    
    # Find the main robot with LiDAR
    robot_main = None
    for obj in env.objects:
        if obj.role == 'robot' and hasattr(obj, 'lidar') and obj.lidar is not None:
            robot_main = obj
            break
    
    if not robot_main:
        print("Error: No robot with LiDAR found")
        return
    
    print(f"Found robot with LiDAR: {robot_main.role}")
    
    # Initialize RVO algorithm
    print("\n3. Initializing RVO collision avoidance...")
    
    # RVO parameters
    vxmax = 2.0  # Maximum velocity in x direction
    vymax = 2.0  # Maximum velocity in y direction
    acce = 1.0   # Acceleration limit
    factor = 1.0 # RVO factor
    
    # Run simulation with LiDAR-based RVO
    print("\n4. Running simulation with LiDAR-based RVO collision avoidance...")
    
    for step in range(50):
        # Get current robot state
        robot_state = robot_main.state
        robot_velocity = robot_main.velocity_xy
        
        # Extract neighbors from LiDAR
        current_time = step * 0.1  # Assume 10Hz simulation
        rvo_neighbors = robot_main.lidar.extract_rvo_neighbors_with_time(current_time)
        
        if step % 10 == 0:  # Print every 10 steps
            print(f"\nStep {step+1}:")
            print(f"  Robot position: ({robot_state[0,0]:.2f}, {robot_state[1,0]:.2f})")
            print(f"  Robot velocity: ({robot_velocity[0,0]:.2f}, {robot_velocity[1,0]:.2f})")
            print(f"  LiDAR detected {len(rvo_neighbors)} neighbors")
            
            for i, neighbor in enumerate(rvo_neighbors):
                print(f"    Neighbor {i+1}: pos=({neighbor[0]:.2f}, {neighbor[1]:.2f}), "
                      f"vel=({neighbor[2]:.2f}, {neighbor[3]:.2f}), radius={neighbor[4]:.2f}")
        
        # Prepare RVO state
        goal = robot_main.goal
        desired_vx = (goal[0,0] - robot_state[0,0]) * 0.5  # Simple goal-seeking
        desired_vy = (goal[1,0] - robot_state[1,0]) * 0.5
        
        # Limit desired velocity
        desired_vx = np.clip(desired_vx, -vxmax, vxmax)
        desired_vy = np.clip(desired_vy, -vymax, vymax)
        
        rvo_state = [
            robot_state[0,0],  # x
            robot_state[1,0],  # y
            robot_velocity[0,0],  # vx
            robot_velocity[1,0],  # vy
            robot_main.radius_extend,  # radius
            desired_vx,  # desired vx
            desired_vy,  # desired vy
        ]
        
        # Run RVO algorithm with LiDAR-detected neighbors
        if len(rvo_neighbors) > 0:
            rvo_algorithm = reciprocal_vel_obs(
                state=rvo_state,
                obs_state_list=rvo_neighbors,
                vxmax=vxmax,
                vymax=vymax,
                acce=acce,
                factor=factor
            )
            
            # Calculate collision-free velocity
            collision_free_vel = rvo_algorithm.cal_vel(mode="rvo")
            
            # Apply the collision-free velocity
            robot_main.velocity_xy[0,0] = collision_free_vel[0]
            robot_main.velocity_xy[1,0] = collision_free_vel[1]
            
            if step % 10 == 0:
                print(f"  RVO output: ({collision_free_vel[0]:.2f}, {collision_free_vel[1]:.2f})")
        else:
            # No obstacles detected, use desired velocity
            robot_main.velocity_xy[0,0] = desired_vx
            robot_main.velocity_xy[1,0] = desired_vy
            
            if step % 10 == 0:
                print(f"  No obstacles, using desired velocity: ({desired_vx:.2f}, {desired_vy:.2f})")
        
        # Step the simulation
        env.step()
        
        # Check if robot reached goal
        distance_to_goal = np.linalg.norm(robot_state[:2,0] - goal[:2,0])
        if distance_to_goal < 0.5:
            print(f"\n🎉 Robot reached goal at step {step+1}!")
            break
    
    print(f"\n=== Demo Complete ===")
    print(f"Successfully demonstrated LiDAR-based RVO collision avoidance!")


if __name__ == "__main__":
    demo_lidar_rvo_collision_avoidance()
