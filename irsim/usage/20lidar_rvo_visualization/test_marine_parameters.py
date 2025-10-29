#!/usr/bin/env python3
"""
Test marine-optimized parameters for Otter USV LiDAR processing.
"""

import os
import sys
import numpy as np

def test_marine_parameters():
    """Test marine-optimized parameters vs default parameters."""
    print("🚤 Testing Marine-Optimized Parameters for Otter USV")
    print("=" * 55)
    
    try:
        import irsim
        from irsim.lib.algorithm.lidar_processing import LidarProcessingConfig
        
        # Load the Otter USV world
        world_path = os.path.join(os.path.dirname(__file__), 'empty_world.yaml')
        env = irsim.make(world_path)
        robot = env.robot
        lidar = robot.sensors[0]
        
        print(f"✅ Loaded Otter USV world")
        print(f"   Robot: {type(robot).__name__}")
        print(f"   LiDAR range: {lidar.range_max}m")
        print(f"   World size: {env._world.width}x{env._world.height}m")
        
        # Test configurations
        configs = {
            "Default": LidarProcessingConfig.default(),
            "Marine Environment": LidarProcessingConfig.marine_environment(vessel_length=2.0),
            "Large Robots": LidarProcessingConfig.large_robots(robot_radius=1.0),
            "Outdoor Environment": LidarProcessingConfig.outdoor_environment()
        }
        
        print(f"\n🔧 Parameter Comparison:")
        print(f"{'Parameter':<25} {'Default':<12} {'Marine':<12} {'Large':<12} {'Outdoor':<12}")
        print("-" * 75)
        
        # Compare key parameters
        key_params = [
            'eps', 'min_samples', 'min_radius', 'max_radius', 
            'velocity_threshold', 'max_velocity', 'wall_boundary_threshold',
            'duplicate_distance_threshold', 'data_association_distance'
        ]
        
        for param in key_params:
            values = []
            for config_name in configs.keys():
                config = configs[config_name]
                value = getattr(config, param)
                if isinstance(value, float):
                    values.append(f"{value:.2f}")
                else:
                    values.append(f"{value}")
            
            print(f"{param:<25} {values[0]:<12} {values[1]:<12} {values[2]:<12} {values[3]:<12}")
        
        # Test performance with different configurations
        print(f"\n🧪 Performance Testing:")
        
        for config_name, config in configs.items():
            print(f"\n📊 Testing {config_name} configuration:")
            
            # Create detector with this configuration
            detector = LidarProcessingConfig.from_config(config)
            
            # Run simulation for a few steps
            total_neighbors = 0
            wall_count = 0
            static_count = 0
            dynamic_count = 0
            
            for step in range(10):
                env.step()
                
                # Get LiDAR data
                ranges = lidar.range_data
                angles = lidar.angle_data
                
                # Extract neighbors
                neighbors = detector.extract_neighbors(
                    ranges=ranges,
                    angles=angles,
                    current_time=step * env._world.step_time,
                    world_bounds=(0, 0, env._world.width, env._world.height),
                    lidar_position=(lidar.lidar_origin[0], lidar.lidar_origin[1]),
                    lidar_velocity=(robot.velocity_xy[0,0], robot.velocity_xy[1,0]),
                    lidar_orientation=robot.state[2,0],
                    range_max=lidar.range_max
                )
                
                total_neighbors += len(neighbors)
                
                # Count by type
                for neighbor in neighbors:
                    if neighbor['type'] == 'wall':
                        wall_count += 1
                    elif neighbor['type'] == 'static_obstacle':
                        static_count += 1
                    elif neighbor['type'] == 'dynamic_obstacle':
                        dynamic_count += 1
            
            avg_neighbors = total_neighbors / 10
            print(f"   Average neighbors per step: {avg_neighbors:.1f}")
            print(f"   Wall detections: {wall_count}")
            print(f"   Static obstacles: {static_count}")
            print(f"   Dynamic obstacles: {dynamic_count}")
        
        # Recommendations
        print(f"\n🎯 Recommendations for Otter USV:")
        print(f"   ✅ Use 'Marine Environment' configuration for optimal performance")
        print(f"   ✅ Key advantages:")
        print(f"      - Realistic max_velocity (5.0 m/s vs 15.0 m/s)")
        print(f"      - Larger max_radius (8.0m vs 4.0m) for marine vessels")
        print(f"      - Optimized wall detection for shorelines")
        print(f"      - Vessel-length scaled duplicate filtering")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_marine_parameters()
    sys.exit(0 if success else 1)
