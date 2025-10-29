#!/usr/bin/env python3
"""
Parameterized LiDAR Processing Demo

This demo shows how to use the fully parameterized LiDAR processing system
with different configurations for various scenarios.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import irsim
from irsim.lib.algorithm.lidar_processing import LidarProcessingConfig, LidarNeighborDetector


def create_parameterized_world():
    """Create a world configuration for testing different parameters."""
    world_config = """
# Parameterized LiDAR Processing Test World
world:
  size: [30, 30]  # Larger world for testing different scales
  collision_mode: "reactive"

robot:
  - name: "robot_main"
    shape:
      name: "circle"
      radius: 0.3
    kinematics:
      name: "omni"
    state: [5, 5, 0]
    velocity: [0, 0]
    goal: [25, 25, 0]
    behavior:
      name: "dash"
    sensors:
      - name: "lidar2d"
        range_max: 15
        number: 360
        angle_range: 6.28
        noise: false
    color: "blue"

# Small obstacles (for small robot config)
obstacle:
  - name: "small_obstacle_1"
    shape:
      name: "circle"
      radius: 0.2
    state: [8, 8, 0]
    color: "red"
    static: true
    
  - name: "small_obstacle_2"
    shape:
      name: "circle"
      radius: 0.15
    state: [12, 12, 0]
    color: "orange"
    static: true

# Large obstacles (for large robot config)
  - name: "large_obstacle_1"
    shape:
      name: "circle"
      radius: 1.0
    state: [18, 8, 0]
    color: "green"
    static: true
    
  - name: "large_obstacle_2"
    shape:
      name: "rectangle"
      width: 2
      length: 3
    state: [22, 18, 0]
    color: "purple"
    static: true

# Dynamic obstacles
  - name: "dynamic_obstacle_1"
    shape:
      name: "circle"
      radius: 0.4
    kinematics:
      name: "omni"
    state: [10, 20, 0]
    velocity: [0.5, -0.3]
    goal: [20, 10, 0]
    behavior:
      name: "dash"
    color: "brown"
    
  - name: "dynamic_obstacle_2"
    shape:
      name: "circle"
      radius: 0.6
    kinematics:
      name: "omni"
    state: [20, 5, 0]
    velocity: [-0.2, 0.4]
    goal: [5, 20, 0]
    behavior:
      name: "dash"
    color: "pink"
"""
    
    # Write the configuration to a file
    config_path = os.path.join(os.path.dirname(__file__), 'parameterized_world.yaml')
    with open(config_path, 'w') as f:
        f.write(world_config)
    
    return config_path


def test_configuration(config_name: str, config: LidarProcessingConfig, env, robot, steps: int = 10):
    """Test a specific configuration and return results."""
    print(f"\n=== Testing {config_name} Configuration ===")
    print(f"Parameters: eps={config.eps}, min_samples={config.min_samples}")
    print(f"Radius range: [{config.min_radius}, {config.max_radius}]")
    print(f"Wall detection: boundary_threshold={config.wall_boundary_threshold}")
    print(f"Velocity threshold: {config.velocity_threshold} m/s")
    
    # Create detector with this configuration
    detector = LidarNeighborDetector.from_config(config)
    
    results = {
        'config_name': config_name,
        'total_detections': 0,
        'wall_detections': 0,
        'static_detections': 0,
        'dynamic_detections': 0,
        'rvo_neighbors': 0
    }
    
    for step in range(steps):
        env.step()
        
        # Get LiDAR data
        lidar = robot.sensors[0]
        ranges = lidar.range_data
        angles = lidar.angle_list
        
        # Extract neighbors with this configuration
        neighbors = detector.extract_neighbors(
            ranges, angles, 
            current_time=env.time, 
            world_bounds=(0, 0, 30, 30)
        )
        
        # Count different types
        for neighbor in neighbors:
            results['total_detections'] += 1
            if neighbor['type'] == 'wall':
                results['wall_detections'] += 1
            elif neighbor['type'] == 'static_obstacle':
                results['static_detections'] += 1
            elif neighbor['type'] == 'dynamic_obstacle':
                results['dynamic_detections'] += 1
        
        # Get RVO neighbors
        rvo_neighbors = detector.neighbors_to_rvo_format(neighbors, include_static=False)
        results['rvo_neighbors'] += len(rvo_neighbors)
        
        if step == steps - 1:  # Print last step details
            print(f"  Step {step + 1}: Detected {len(neighbors)} objects")
            for i, neighbor in enumerate(neighbors):
                print(f"    {neighbor['type']}: pos={neighbor['position']}, "
                      f"vel={neighbor['velocity']}, radius={neighbor['radius']:.2f}")
            print(f"  RVO neighbors (dynamic only): {len(rvo_neighbors)}")
    
    # Calculate averages
    results['avg_total'] = results['total_detections'] / steps
    results['avg_walls'] = results['wall_detections'] / steps
    results['avg_static'] = results['static_detections'] / steps
    results['avg_dynamic'] = results['dynamic_detections'] / steps
    results['avg_rvo'] = results['rvo_neighbors'] / steps
    
    return results


def main():
    """Main demo function."""
    print("=== Parameterized LiDAR Processing Demo ===")
    print("Testing different configurations for various scenarios")
    
    # Create test world
    world_path = create_parameterized_world()
    print(f"Created test world: {world_path}")
    
    # Initialize environment
    env = irsim.make(world_path)
    robot = env.robot  # env.robot returns the first robot directly
    
    print(f"Environment initialized with robot at {robot.state}")
    print(f"World size: {env._world.width} x {env._world.height}")
    
    # Test different configurations
    configurations = {
        'Default': LidarProcessingConfig.default(),
        'Small Robots': LidarProcessingConfig.small_robots(robot_radius=0.2),
        'Large Robots': LidarProcessingConfig.large_robots(robot_radius=0.5),
        'High Precision': LidarProcessingConfig.high_precision(),
        'Indoor Environment': LidarProcessingConfig.indoor_environment(),
        'Outdoor Environment': LidarProcessingConfig.outdoor_environment(),
        'Dynamic Environment': LidarProcessingConfig.dynamic_environment()
    }
    
    results = []
    
    for config_name, config in configurations.items():
        # Reset environment for each test
        env.reset()
        robot = env.robot
        
        # Test this configuration
        result = test_configuration(config_name, config, env, robot, steps=5)
        results.append(result)
    
    # Print summary
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Configuration':<20} {'Total':<8} {'Walls':<8} {'Static':<8} {'Dynamic':<8} {'RVO':<8}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['config_name']:<20} "
              f"{result['avg_total']:<8.1f} "
              f"{result['avg_walls']:<8.1f} "
              f"{result['avg_static']:<8.1f} "
              f"{result['avg_dynamic']:<8.1f} "
              f"{result['avg_rvo']:<8.1f}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("• Small Robots: Better at detecting small obstacles, tighter clustering")
    print("• Large Robots: Better at detecting large obstacles, more robust to noise")
    print("• High Precision: Very sensitive, detects more objects with higher accuracy")
    print("• Indoor Environment: Optimized for walls and furniture detection")
    print("• Outdoor Environment: Handles large distances and obstacles")
    print("• Dynamic Environment: Better velocity estimation for moving objects")
    print("• RVO neighbors: Only dynamic obstacles (walls and static excluded)")
    
    print("\n" + "="*80)
    print("USAGE RECOMMENDATIONS:")
    print("="*80)
    print("1. Choose configuration based on your robot size and environment")
    print("2. Adjust parameters for specific scenarios:")
    print("   - eps: Clustering sensitivity (smaller = more clusters)")
    print("   - min_samples: Minimum points per cluster")
    print("   - velocity_threshold: Dynamic vs static classification")
    print("   - wall_boundary_threshold: Wall detection sensitivity")
    print("3. Use LidarProcessingConfig.from_config() for easy parameter management")
    print("4. Test different configurations to find optimal settings for your use case")
    
    # Clean up
    os.remove(world_path)
    print(f"\nCleaned up test file: {world_path}")


if __name__ == "__main__":
    main()
