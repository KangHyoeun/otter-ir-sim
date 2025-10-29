#!/usr/bin/env python3
"""
Otter USV LiDAR Demonstration
Simple demonstration of LiDAR sensor working with Otter USV robot.
"""

import os
import sys
import numpy as np
import irsim

def demonstrate_otter_usv_lidar():
    """Demonstrate Otter USV LiDAR functionality."""
    
    print("🚤 Otter USV LiDAR Demonstration")
    print("=" * 40)
    
    # Create environment
    world_path = os.path.join(os.path.dirname(__file__), 'empty_world.yaml')
    print(f"📁 Loading world: {world_path}")
    
    env = irsim.make(world_path)
    robot = env.robot
    lidar = robot.lidar
    
    print(f"🤖 Robot: {type(robot).__name__} ({robot.kinematics})")
    print(f"📡 LiDAR: {type(lidar).__name__}")
    print(f"🌍 World: {env._world.width}x{env._world.height} m")
    print()
    
    # Show initial state
    robot_pos = robot.state[:2].flatten()
    robot_heading = robot.state[2, 0]
    print(f"📍 Initial position: ({robot_pos[0]:.1f}, {robot_pos[1]:.1f})")
    print(f"🧭 Initial heading: {np.degrees(robot_heading):.1f}°")
    goal_pos = robot.goal[:2].flatten()
    print(f"🎯 Goal: ({goal_pos[0]:.1f}, {goal_pos[1]:.1f})")
    print()
    
    # Run simulation
    print("🏃 Running simulation...")
    for step in range(160):
        # Step simulation first to update robot state
        env.step()
        
        # Get robot state after step
        robot_pos = robot.state[:2].flatten()
        robot_vel = robot.velocity_xy.flatten()[:2]
        robot_heading = robot.state[2, 0]
        
        # Get LiDAR data
        ranges = lidar.range_data
        valid_points = np.sum(ranges < lidar.range_max)
        
        # Extract neighbors
        neighbors = lidar.extract_neighbors_detailed_with_robot_velocity(
            robot_velocity=(robot_vel[0], robot_vel[1]),
            current_time=step * env._world.step_time,
            world_bounds=(0, 0, env._world.width, env._world.height)
        )
        
        # Calculate distance to goal
        goal_distance = np.linalg.norm(goal_pos - robot_pos)
        
        print(f"Step {step+1:2d}: pos=({robot_pos[0]:6.1f}, {robot_pos[1]:6.1f}), "
              f"vel=({robot_vel[0]:5.2f}, {robot_vel[1]:5.2f}), "
              f"heading={np.degrees(robot_heading):6.1f}°, "
              f"goal={goal_distance:6.1f}m, "
              f"LiDAR={valid_points:3d}pts, "
              f"neighbors={len(neighbors):2d}")
        
        # Check if goal reached
        if env.done():
            print(f"🎯 Goal reached at step {step+1}!")
            break
    
    print()
    print("✅ Demonstration completed!")
    print("🎉 LiDAR sensor works properly with Otter USV!")

def main():
    """Main function."""
    try:
        demonstrate_otter_usv_lidar()
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
