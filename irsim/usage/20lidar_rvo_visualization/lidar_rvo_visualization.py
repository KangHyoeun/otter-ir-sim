#!/usr/bin/env python3
"""
LiDAR-Based RVO Collision Avoidance with Visualization

This demo shows how to use LiDAR sensor data to implement RVO collision avoidance
with real-time plotting to visualize the process.

Author: AI Assistant
Date: 2024
"""

import irsim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from irsim.lib.algorithm.rvo import reciprocal_vel_obs


def create_lidar_rvo_world():
    """Create a YAML configuration for LiDAR-based RVO collision avoidance."""
    yaml_content = """# LiDAR-Based RVO Collision Avoidance World with Visualization
world:
  size: [15, 15]
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
    goal: [12, 12, 0]
    behavior:
      name: "dash"
    sensors:
      - name: "lidar2d"
        range_max: 6
        number: 180
        angle_range: 6.28
        noise: false
    color: "blue"

# Moving obstacles
  - name: "robot2"
    shape:
      name: "circle"
      radius: 0.4
    kinematics:
      name: "omni"
    state: [6, 4, 0]
    velocity: [0.2, 0.3]
    goal: [4, 10, 0]
    behavior:
      name: "dash"
    color: "red"
    
  - name: "robot3"
    shape:
      name: "circle"
      radius: 0.35
    kinematics:
      name: "omni"
    state: [4, 6, 0]
    velocity: [-0.3, 0.2]
    goal: [10, 4, 0]
    behavior:
      name: "dash"
    color: "green"

# Static obstacles
obstacle:
  - name: "obs1"
    shape:
      name: "circle"
      radius: 0.5
    state: [8, 8, 0]
    color: "orange"
    static: true
"""
    
    import os
    yaml_path = os.path.join(os.path.dirname(__file__), 'lidar_rvo_visualization_world.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print("Created LiDAR RVO visualization world configuration")


def demo_lidar_rvo_with_plotting():
    """Demonstrate LiDAR-based RVO collision avoidance with real-time plotting."""
    print("=== LiDAR-Based RVO Collision Avoidance with Visualization ===")
    
    # Create world configuration
    print("\n1. Creating world configuration...")
    create_lidar_rvo_world()
    
    # Initialize environment
    print("\n2. Initializing IR-SIM environment...")
    import os
    yaml_path = os.path.join(os.path.dirname(__file__), 'lidar_rvo_visualization_world.yaml')
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
    
    # Initialize plotting
    print("\n3. Setting up visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Environment and robot trajectories
    ax1.set_xlim(0, 15)
    ax1.set_ylim(0, 15)
    ax1.set_aspect('equal')
    ax1.set_title('LiDAR-Based RVO Collision Avoidance')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: LiDAR scan visualization
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-6, 6)
    ax2.set_aspect('equal')
    ax2.set_title('LiDAR Scan Data')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True, alpha=0.3)
    
    # Initialize RVO algorithm
    print("\n4. Initializing RVO collision avoidance...")
    
    # RVO parameters
    vxmax = 1.5
    vymax = 1.5
    acce = 0.8
    factor = 1.0
    
    # Storage for plotting
    robot_trajectory = []
    detected_neighbors_history = []
    lidar_points_history = []
    
    # Run simulation with plotting
    print("\n5. Running simulation with real-time visualization...")
    
    for step in range(50):
        # Get current robot state
        robot_state = robot_main.state
        robot_velocity = robot_main.velocity_xy
        
        # Store robot position for trajectory
        robot_trajectory.append([robot_state[0,0], robot_state[1,0]])
        
        # Extract neighbors from LiDAR
        current_time = step * 0.1
        rvo_neighbors = robot_main.lidar.extract_rvo_neighbors_with_time(current_time)
        
        # Store LiDAR data for visualization
        lidar_points = []
        for i in range(len(robot_main.lidar.range_data)):
            angle = robot_main.lidar.angle_list[i]
            range_val = robot_main.lidar.range_data[i]
            if range_val < robot_main.lidar.range_max - 0.02:
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                lidar_points.append([x, y])
        
        lidar_points_history.append(lidar_points)
        detected_neighbors_history.append(rvo_neighbors)
        
        # Prepare RVO state
        goal = robot_main.goal
        desired_vx = (goal[0,0] - robot_state[0,0]) * 0.3
        desired_vy = (goal[1,0] - robot_state[1,0]) * 0.3
        
        desired_vx = np.clip(desired_vx, -vxmax, vxmax)
        desired_vy = np.clip(desired_vy, -vymax, vymax)
        
        rvo_state = [
            robot_state[0,0], robot_state[1,0],
            robot_velocity[0,0], robot_velocity[1,0],
            robot_main.radius_extend,
            desired_vx, desired_vy
        ]
        
        # Run RVO algorithm
        if len(rvo_neighbors) > 0:
            rvo_algorithm = reciprocal_vel_obs(
                state=rvo_state,
                obs_state_list=rvo_neighbors,
                vxmax=vxmax, vymax=vymax, acce=acce, factor=factor
            )
            collision_free_vel = rvo_algorithm.cal_vel(mode="rvo")
            robot_main.velocity_xy[0,0] = collision_free_vel[0]
            robot_main.velocity_xy[1,0] = collision_free_vel[1]
        else:
            robot_main.velocity_xy[0,0] = desired_vx
            robot_main.velocity_xy[1,0] = desired_vy
        
        # Step the simulation
        env.step()
        
        # Check if robot reached goal
        distance_to_goal = np.linalg.norm(robot_state[:2,0] - goal[:2,0])
        if distance_to_goal < 0.5:
            print(f"\n🎉 Robot reached goal at step {step+1}!")
            break
    
    # Create final visualization
    print("\n6. Creating final visualization...")
    
    # Clear plots
    ax1.clear()
    ax2.clear()
    
    # Setup plots
    ax1.set_xlim(0, 15)
    ax1.set_ylim(0, 15)
    ax1.set_aspect('equal')
    ax1.set_title('LiDAR-Based RVO Collision Avoidance - Final Result')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-6, 6)
    ax2.set_aspect('equal')
    ax2.set_title('LiDAR Detection Results')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True, alpha=0.3)
    
    # Plot robot trajectory
    if len(robot_trajectory) > 1:
        trajectory = np.array(robot_trajectory)
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Robot Path')
        ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='Start')
        ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=8, label='End')
    
    # Plot goal
    ax1.plot(robot_main.goal[0,0], robot_main.goal[1,0], 'r*', markersize=15, label='Goal')
    
    # Plot all objects
    for obj in env.objects:
        if obj.role == 'robot':
            if obj == robot_main:
                circle = plt.Circle((obj.state[0,0], obj.state[1,0]), obj.radius_extend, 
                                  color='blue', alpha=0.7, label='Main Robot')
            else:
                circle = plt.Circle((obj.state[0,0], obj.state[1,0]), obj.radius_extend, 
                                  color='red', alpha=0.7, label='Other Robots')
            ax1.add_patch(circle)
        elif obj.role == 'obstacle':
            circle = plt.Circle((obj.state[0,0], obj.state[1,0]), obj.radius_extend, 
                              color='orange', alpha=0.7, label='Static Obstacles')
            ax1.add_patch(circle)
    
    # Plot LiDAR detections
    if len(detected_neighbors_history) > 0:
        # Get the last detection
        last_neighbors = detected_neighbors_history[-1]
        last_lidar_points = lidar_points_history[-1]
        
        # Plot LiDAR points
        if last_lidar_points:
            lidar_array = np.array(last_lidar_points)
            ax2.scatter(lidar_array[:, 0], lidar_array[:, 1], c='lightblue', s=10, alpha=0.6, label='LiDAR Points')
        
        # Plot detected neighbors
        for i, neighbor in enumerate(last_neighbors):
            x, y, vx, vy, radius = neighbor
            circle = plt.Circle((x, y), radius, color='red', alpha=0.5, label=f'Detected Obstacle {i+1}')
            ax2.add_patch(circle)
            ax2.arrow(x, y, vx*2, vy*2, head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
    
    # Add legends
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add statistics
    stats_text = f"""
Statistics:
• Total steps: {len(robot_trajectory)}
• Final position: ({robot_trajectory[-1][0]:.2f}, {robot_trajectory[-1][1]:.2f})
• Goal position: ({robot_main.goal[0,0]:.2f}, {robot_main.goal[1,0]:.2f})
• Distance to goal: {distance_to_goal:.2f}m
• LiDAR detections: {len(detected_neighbors_history[-1]) if detected_neighbors_history else 0}
"""
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== Visualization Complete ===")
    print(f"✅ Robot trajectory plotted")
    print(f"✅ LiDAR detections visualized")
    print(f"✅ Collision avoidance demonstrated")
    print(f"✅ Final statistics displayed")


if __name__ == "__main__":
    demo_lidar_rvo_with_plotting()
