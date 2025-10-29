#!/usr/bin/env python3
"""
LiDAR Processing Algorithms for IR-SIM

This module provides algorithms for processing LiDAR sensor data to extract
neighbor information for collision avoidance algorithms.

Author: AI Assistant
Date: 2024
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import warnings
from dataclasses import dataclass

# Try to import scipy for optimization, fall back to simple methods if not available
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available, using simple optimization methods")


@dataclass
class LidarProcessingConfig:
    """
    Configuration class for LiDAR processing parameters.
    
    This class provides predefined configurations for different scenarios
    and makes it easy to customize parameters for specific use cases.
    """
    
    # Clustering parameters
    eps: float = 0.2  # distance threshold for clustering (more sensitive for better detection)
    min_samples: int = 1  # minimum points per cluster (more sensitive for better detection)
    
    # Radius constraints
    min_radius: float = 0.05  # minimum expected radius (more sensitive for better detection)
    max_radius: float = 4.0  # maximum expected radius (increased for walls)
    
    # Velocity estimation parameters
    velocity_history_length: int = 8  # increased for better velocity estimation
    velocity_threshold: float = 0.05  # reduced threshold for better sensitivity
    max_velocity: float = 15.0  # increased max velocity limit
    
    # Wall detection parameters
    wall_boundary_threshold: float = 0.5  # optimized for better wall detection
    wall_aspect_ratio_threshold: float = 2.5  # optimized for better wall detection
    wall_min_radius: float = 0.6  # optimized minimum radius for walls
    wall_max_radius: float = 3.0  # optimized maximum radius for walls
    wall_point_spread_factor: float = 1.3  # optimized for better wall detection
    
    # Data association parameters
    data_association_distance: float = 1.5  # reduced for more precise association
    min_time_difference: float = 0.005  # reduced for better responsiveness
    
    # Duplicate filtering parameters
    duplicate_distance_threshold: float = 0.4  # optimized for better duplicate filtering
    
    # Circle fitting parameters
    circle_fit_bounds_margin: float = 1.5  # reduced for tighter bounds
    
    # Confidence calculation parameters
    confidence_points_factor: float = 8.0  # reduced for higher confidence requirements
    
    @classmethod
    def default(cls) -> 'LidarProcessingConfig':
        """Default configuration suitable for most scenarios."""
        return cls()
    
    @classmethod
    def small_robots(cls, robot_radius: float = 0.2) -> 'LidarProcessingConfig':
        """Configuration optimized for small robots (e.g., TurtleBot, small drones)."""
        return cls(
            eps=0.2,
            min_samples=3,
            min_radius=robot_radius * 0.5,
            max_radius=robot_radius * 3.0,
            wall_boundary_threshold=0.5,
            wall_min_radius=1.0,
            wall_max_radius=2.0,
            duplicate_distance_threshold=robot_radius * 2.0,
            data_association_distance=robot_radius * 5.0
        )
    
    @classmethod
    def large_robots(cls, robot_radius: float = 0.5) -> 'LidarProcessingConfig':
        """Configuration optimized for large robots (e.g., cars, trucks, large drones)."""
        return cls(
            eps=0.5,
            min_samples=5,
            min_radius=robot_radius * 0.5,
            max_radius=robot_radius * 4.0,
            wall_boundary_threshold=1.5,
            wall_min_radius=2.0,
            wall_max_radius=5.0,
            duplicate_distance_threshold=robot_radius * 2.0,
            data_association_distance=robot_radius * 4.0
        )
    
    @classmethod
    def high_precision(cls) -> 'LidarProcessingConfig':
        """Configuration for high-precision applications (e.g., surgical robots, precision manufacturing)."""
        return cls(
            eps=0.1,
            min_samples=5,
            min_radius=0.05,
            max_radius=1.0,
            velocity_threshold=0.05,
            wall_boundary_threshold=0.2,
            wall_min_radius=0.5,
            wall_max_radius=1.5,
            duplicate_distance_threshold=0.2,
            data_association_distance=1.0,
            min_time_difference=0.005,
            confidence_points_factor=20.0
        )
    
    @classmethod
    def outdoor_environment(cls) -> 'LidarProcessingConfig':
        """Configuration for outdoor environments with large obstacles and distances."""
        return cls(
            eps=1.0,
            min_samples=8,
            min_radius=0.3,
            max_radius=10.0,
            velocity_threshold=0.2,
            wall_boundary_threshold=2.0,
            wall_min_radius=3.0,
            wall_max_radius=15.0,
            duplicate_distance_threshold=1.0,
            data_association_distance=5.0,
            max_velocity=20.0,
            circle_fit_bounds_margin=5.0
        )
    
    @classmethod
    def indoor_environment(cls) -> 'LidarProcessingConfig':
        """Configuration for indoor environments with walls and furniture."""
        return cls(
            eps=0.4,
            min_samples=4,
            min_radius=0.2,
            max_radius=3.0,
            velocity_threshold=0.1,
            wall_boundary_threshold=0.8,
            wall_min_radius=1.2,
            wall_max_radius=2.5,
            wall_aspect_ratio_threshold=2.5,
            duplicate_distance_threshold=0.4,
            data_association_distance=2.5,
            max_velocity=5.0
        )
    
    @classmethod
    def dynamic_environment(cls) -> 'LidarProcessingConfig':
        """Configuration for environments with many moving objects."""
        return cls(
            eps=0.3,
            min_samples=3,
            min_radius=0.2,
            max_radius=2.0,
            velocity_threshold=0.05,  # Lower threshold for better dynamic detection
            velocity_history_length=8,  # More history for better velocity estimation
            wall_boundary_threshold=1.0,
            wall_min_radius=1.5,
            wall_max_radius=3.0,
            duplicate_distance_threshold=0.3,
            data_association_distance=1.5,
            min_time_difference=0.005,
            confidence_points_factor=8.0
        )
    
    @classmethod
    def marine_environment(cls, vessel_length: float = 2.0) -> 'LidarProcessingConfig':
        """Configuration optimized for marine environments (USVs, boats, marine vehicles).
        
        Args:
            vessel_length: Length of the marine vessel in meters (default: 2.0m for Otter USV)
        """
        return cls(
            # Clustering parameters - optimized for marine obstacles
            eps=0.3,  # Slightly larger for marine environment noise
            min_samples=2,  # More sensitive for sparse marine obstacles
            
            # Radius constraints - marine-specific
            min_radius=0.1,  # Small buoys, navigation markers
            max_radius=8.0,   # Large vessels, docks
            
            # Velocity estimation - marine vehicle speeds
            velocity_history_length=10,  # Longer history for smooth marine motion
            velocity_threshold=0.05,    # Low threshold for slow marine vehicles
            max_velocity=5.0,           # Realistic max speed for marine vehicles
            
            # Wall detection - shorelines, docks, breakwaters
            wall_boundary_threshold=1.0,  # Larger threshold for shorelines
            wall_aspect_ratio_threshold=3.0,  # Higher for long shorelines
            wall_min_radius=1.0,         # Minimum size for marine structures
            wall_max_radius=6.0,         # Large marine structures
            wall_point_spread_factor=1.5,  # Account for irregular shorelines
            
            # Data association - marine environment considerations
            data_association_distance=2.0,  # Larger for marine environment
            min_time_difference=0.01,       # Standard marine update rate
            
            # Duplicate filtering - vessel-specific
            duplicate_distance_threshold=vessel_length * 0.5,  # Scale with vessel size
            
            # Circle fitting - marine obstacles
            circle_fit_bounds_margin=2.0,  # Larger margin for marine environment
            
            # Confidence calculation - marine reliability
            confidence_points_factor=6.0   # Moderate confidence for marine environment
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert configuration to dictionary for easy parameter passing."""
        return {
            'eps': self.eps,
            'min_samples': self.min_samples,
            'min_radius': self.min_radius,
            'max_radius': self.max_radius,
            'velocity_history_length': self.velocity_history_length,
            'velocity_threshold': self.velocity_threshold,
            'max_velocity': self.max_velocity,
            'wall_boundary_threshold': self.wall_boundary_threshold,
            'wall_aspect_ratio_threshold': self.wall_aspect_ratio_threshold,
            'wall_min_radius': self.wall_min_radius,
            'wall_max_radius': self.wall_max_radius,
            'wall_point_spread_factor': self.wall_point_spread_factor,
            'data_association_distance': self.data_association_distance,
            'min_time_difference': self.min_time_difference,
            'duplicate_distance_threshold': self.duplicate_distance_threshold,
            'circle_fit_bounds_margin': self.circle_fit_bounds_margin,
            'confidence_points_factor': self.confidence_points_factor
        }


class LidarNeighborDetector:
    """
    Advanced LiDAR neighbor detection with velocity estimation.
    
    This class provides comprehensive neighbor detection from LiDAR data,
    including position, velocity, and radius estimation with fully configurable parameters.
    """
    
    def __init__(self, 
                 # Clustering parameters
                 eps: float = 0.3,
                 min_samples: int = 3,
                 
                 # Radius constraints
                 min_radius: float = 0.1,
                 max_radius: float = 2.0,
                 
                 # Velocity estimation parameters
                 velocity_history_length: int = 5,
                 velocity_threshold: float = 0.1,  # m/s threshold for dynamic vs static
                 max_velocity: float = 10.0,  # m/s maximum realistic velocity
                 
                 # Wall detection parameters
                 wall_boundary_threshold: float = 1.0,  # meters from world boundary
                 wall_aspect_ratio_threshold: float = 3.0,  # aspect ratio for wall detection
                 wall_min_radius: float = 1.5,  # minimum radius for wall detection
                 wall_max_radius: float = 3.0,  # maximum radius for wall detection
                 wall_point_spread_factor: float = 2.0,  # factor for point spread detection
                 
                 # Data association parameters
                 data_association_distance: float = 2.0,  # max distance for data association
                 min_time_difference: float = 0.01,  # minimum time difference for velocity
                 
                 # Duplicate filtering parameters
                 duplicate_distance_threshold: float = 0.5,  # distance threshold for duplicates
                 
                 # Circle fitting parameters
                 circle_fit_bounds_margin: float = 2.0,  # margin for circle fitting bounds
                 
                 # Confidence calculation parameters
                 confidence_points_factor: float = 10.0):  # points per confidence unit
        """
        Initialize the LiDAR neighbor detector with fully configurable parameters.
        
        Args:
            # Clustering parameters
            eps: Maximum distance between points in the same cluster
            min_samples: Minimum number of points required to form a cluster
            
            # Radius constraints
            min_radius: Minimum expected neighbor radius
            max_radius: Maximum expected neighbor radius
            
            # Velocity estimation parameters
            velocity_history_length: Number of previous scans for velocity estimation
            velocity_threshold: Speed threshold (m/s) for dynamic vs static classification
            max_velocity: Maximum realistic velocity (m/s) for filtering
            
            # Wall detection parameters
            wall_boundary_threshold: Distance from world boundary to classify as wall (m)
            wall_aspect_ratio_threshold: Aspect ratio threshold for wall detection
            wall_min_radius: Minimum radius for wall classification (m)
            wall_max_radius: Maximum radius for wall classification (m)
            wall_point_spread_factor: Factor for detecting spread-out points (wall-like)
            
            # Data association parameters
            data_association_distance: Maximum distance for associating objects across frames (m)
            min_time_difference: Minimum time difference for velocity calculation (s)
            
            # Duplicate filtering parameters
            duplicate_distance_threshold: Distance threshold for duplicate detection (m)
            
            # Circle fitting parameters
            circle_fit_bounds_margin: Margin for circle fitting bounds (m)
            
            # Confidence calculation parameters
            confidence_points_factor: Points per confidence unit
        """
        # Clustering parameters
        self.eps = eps
        self.min_samples = min_samples
        
        # Radius constraints
        self.min_radius = min_radius
        self.max_radius = max_radius
        
        # Velocity estimation parameters
        self.velocity_history_length = velocity_history_length
        self.velocity_threshold = velocity_threshold
        self.max_velocity = max_velocity
        
        # Wall detection parameters
        self.wall_boundary_threshold = wall_boundary_threshold
        self.wall_aspect_ratio_threshold = wall_aspect_ratio_threshold
        self.wall_min_radius = wall_min_radius
        self.wall_max_radius = wall_max_radius
        self.wall_point_spread_factor = wall_point_spread_factor
        
        # Data association parameters
        self.data_association_distance = data_association_distance
        self.min_time_difference = min_time_difference
        
        # Duplicate filtering parameters
        self.duplicate_distance_threshold = duplicate_distance_threshold
        
        # Circle fitting parameters
        self.circle_fit_bounds_margin = circle_fit_bounds_margin
        
        # Confidence calculation parameters
        self.confidence_points_factor = confidence_points_factor
        
        # History for velocity estimation
        self.position_history = []
        self.timestamp_history = []
    
    @classmethod
    def from_config(cls, config: LidarProcessingConfig) -> 'LidarNeighborDetector':
        """
        Create a LidarNeighborDetector from a configuration object.
        
        Args:
            config: LidarProcessingConfig object with all parameters
            
        Returns:
            LidarNeighborDetector: Configured detector instance
        """
        return cls(**config.to_dict())
    
    def scan_to_pointcloud(self, ranges: np.ndarray, angles: np.ndarray, range_max: float = None) -> np.ndarray:
        """Convert LiDAR range and angle data to Cartesian point cloud."""
        # Filter out invalid ranges: must be finite, positive, and less than range_max
        valid_mask = np.isfinite(ranges) & (ranges > 0)
        
        # If range_max is provided, exclude range_max values (these indicate no detection)
        if range_max is not None:
            valid_mask = valid_mask & (ranges < range_max)
        
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        
        return np.column_stack([x, y])
    
    def cluster_points(self, pointcloud: np.ndarray) -> List[np.ndarray]:
        """Cluster point cloud into separate objects."""
        if len(pointcloud) < self.min_samples:
            return []
        
        # Use simple but effective clustering
        return self._simple_cluster_points(pointcloud)
    
    def _simple_cluster_points(self, pointcloud: np.ndarray) -> List[np.ndarray]:
        """Improved distance-based clustering with better point ordering."""
        clusters = []
        used_points = set()
        
        # Sort points by distance from origin for more consistent clustering
        distances_from_origin = np.linalg.norm(pointcloud, axis=1)
        sorted_indices = np.argsort(distances_from_origin)
        
        for idx in sorted_indices:
            if idx in used_points:
                continue
                
            point = pointcloud[idx]
            cluster = [point]
            used_points.add(idx)
            
            # Find all points within eps distance
            for j, other_point in enumerate(pointcloud):
                if j in used_points:
                    continue
                    
                distance = np.linalg.norm(point - other_point)
                if distance <= self.eps:
                    cluster.append(other_point)
                    used_points.add(j)
            
            # Only keep clusters with sufficient points
            if len(cluster) >= self.min_samples:
                clusters.append(np.array(cluster))
        
        return clusters
    
    def estimate_circle_parameters(self, points: np.ndarray) -> Tuple[float, float, float]:
        """Estimate circle parameters from a cluster of points."""
        if len(points) < 3:
            center = np.mean(points, axis=0)
            return center[0], center[1], self.min_radius
        
        # Use weighted centroid for better center estimation
        # Weight points by their distance from origin (closer points are more reliable)
        distances_from_origin = np.linalg.norm(points, axis=1)
        weights = 1.0 / (distances_from_origin + 0.1)  # Avoid division by zero
        weights = weights / np.sum(weights)  # Normalize weights
        
        center_guess = np.average(points, axis=0, weights=weights)
        
        # Use median distance for more robust radius estimation
        distances_from_center = np.linalg.norm(points - center_guess, axis=1)
        radius_guess = np.median(distances_from_center)
        
        if SCIPY_AVAILABLE:
            try:
                def circle_error(params):
                    cx, cy, r = params
                    predicted_radii = np.linalg.norm(points - np.array([cx, cy]), axis=1)
                    return np.sum((predicted_radii - r) ** 2)
                
                bounds = [
                    (center_guess[0] - self.circle_fit_bounds_margin, center_guess[0] + self.circle_fit_bounds_margin),
                    (center_guess[1] - self.circle_fit_bounds_margin, center_guess[1] + self.circle_fit_bounds_margin),
                    (self.min_radius, self.max_radius)
                ]
                
                result = minimize(circle_error, [center_guess[0], center_guess[1], radius_guess], 
                                bounds=bounds, method='L-BFGS-B')
                
                if result.success:
                    return result.x[0], result.x[1], result.x[2]
            except Exception:
                pass
        
        # Fallback to improved estimation
        center = np.mean(points, axis=0)
        distances_from_center = np.linalg.norm(points - center, axis=1)
        radius = np.max(distances_from_center)  # Use max distance for LiDAR data
        return center[0], center[1], max(radius, self.min_radius)
    
    def estimate_velocities(self, current_positions: List[Tuple[float, float]], 
                          current_time: float) -> List[Tuple[float, float]]:
        """Estimate velocities using position history with improved data association."""
        velocities = []
        
        if len(self.position_history) < 2:
            return [(0.0, 0.0)] * len(current_positions)
        
        for curr_pos in current_positions:
            best_velocity = (0.0, 0.0)
            velocities_list = []
            
            # Use the most recent history for better velocity estimation
            # Look at the last few frames to get a more stable velocity estimate
            for i in range(max(0, len(self.position_history) - self.velocity_history_length), 
                          len(self.position_history)):
                hist_positions = self.position_history[i]
                hist_time = self.timestamp_history[i]
                
                if len(hist_positions) == 0:
                    continue
                
                # Find closest neighbor in history using improved distance calculation
                distances = [np.linalg.norm(np.array(curr_pos) - np.array(hist_pos)) 
                           for hist_pos in hist_positions]
                closest_idx = np.argmin(distances)
                closest_distance = distances[closest_idx]
                
                # More lenient distance threshold for data association
                if closest_distance < self.data_association_distance:
                    hist_pos = hist_positions[closest_idx]
                    dt = current_time - hist_time
                    
                    # Use smaller minimum time difference for better responsiveness
                    if dt > self.min_time_difference:
                        vx = (curr_pos[0] - hist_pos[0]) / dt
                        vy = (curr_pos[1] - hist_pos[1]) / dt
                        
                        # More lenient velocity filtering
                        speed = np.sqrt(vx**2 + vy**2)
                        if speed < self.max_velocity:
                            velocities_list.append((vx, vy))
            
            # Use weighted average instead of median for better velocity estimation
            if len(velocities_list) > 0:
                # Weight recent velocities more heavily
                weights = np.linspace(0.5, 1.0, len(velocities_list))
                vx_values = [v[0] for v in velocities_list]
                vy_values = [v[1] for v in velocities_list]
                
                # Weighted average
                weighted_vx = np.average(vx_values, weights=weights)
                weighted_vy = np.average(vy_values, weights=weights)
                best_velocity = (weighted_vx, weighted_vy)
            
            velocities.append(best_velocity)
        
        return velocities
    
    def _transform_to_absolute_coordinates(self, relative_positions: List[Tuple[float, float]], 
                                         relative_velocities: List[Tuple[float, float]],
                                         lidar_position: Optional[Tuple[float, float]] = None,
                                         lidar_velocity: Optional[Tuple[float, float]] = None,
                                         lidar_orientation: Optional[float] = None) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Transform relative coordinates to absolute world coordinates.
        
        Args:
            relative_positions: List of (x, y) positions relative to LiDAR sensor
            relative_velocities: List of (vx, vy) velocities relative to LiDAR sensor
            lidar_position: Optional (x, y) absolute position of LiDAR sensor
            lidar_velocity: Optional (vx, vy) absolute velocity of LiDAR sensor
            lidar_orientation: Optional orientation angle of LiDAR sensor (radians)
        
        Returns:
            Tuple of (absolute_positions, absolute_velocities)
        """
        if lidar_position is None:
            # No transformation available, return relative coordinates
            return relative_positions, relative_velocities
        
        lidar_x, lidar_y = lidar_position
        lidar_vx = lidar_velocity[0] if lidar_velocity is not None else 0.0
        lidar_vy = lidar_velocity[1] if lidar_velocity is not None else 0.0
        lidar_theta = lidar_orientation if lidar_orientation is not None else 0.0
        
        absolute_positions = []
        absolute_velocities = []
        
        for rel_pos, rel_vel in zip(relative_positions, relative_velocities):
            rel_x, rel_y = rel_pos
            rel_vx, rel_vy = rel_vel
            
            # Transform position: rotate by LiDAR orientation, then translate by LiDAR position
            cos_theta = np.cos(lidar_theta)
            sin_theta = np.sin(lidar_theta)
            
            # Rotate relative position to world frame
            abs_x = lidar_x + rel_x * cos_theta - rel_y * sin_theta
            abs_y = lidar_y + rel_x * sin_theta + rel_y * cos_theta
            
            # Transform velocity: rotate relative velocity, then add LiDAR velocity
            abs_vx = lidar_vx + rel_vx * cos_theta - rel_vy * sin_theta
            abs_vy = lidar_vy + rel_vx * sin_theta + rel_vy * cos_theta
            
            absolute_positions.append((abs_x, abs_y))
            absolute_velocities.append((abs_vx, abs_vy))
        
        return absolute_positions, absolute_velocities
    
    def extract_neighbors(self, ranges: np.ndarray, angles: np.ndarray, 
                         current_time: float = 0.0, world_bounds: Optional[tuple] = None,
                         lidar_position: Optional[Tuple[float, float]] = None,
                         lidar_velocity: Optional[Tuple[float, float]] = None,
                         lidar_orientation: Optional[float] = None,
                         range_max: Optional[float] = None) -> List[Dict]:
        """
        Extract neighbor information from LiDAR scan with wall/obstacle classification.
        
        Args:
            ranges: LiDAR range data
            angles: LiDAR angle data
            current_time: Current simulation time
            world_bounds: Optional (x_min, y_min, x_max, y_max) for wall detection
            lidar_position: Optional (x, y) absolute position of LiDAR sensor in world coordinates
            lidar_velocity: Optional (vx, vy) absolute velocity of LiDAR sensor in world coordinates
            lidar_orientation: Optional orientation angle of LiDAR sensor in world coordinates (radians)
            range_max: Optional maximum range value - ranges >= range_max are treated as "no detection"
        
        Returns:
            List[Dict]: List of neighbor dictionaries with keys:
                - 'position': (x, y) tuple - ABSOLUTE coordinates in world frame
                - 'velocity': (vx, vy) tuple - ABSOLUTE velocity in world frame
                - 'radius': float
                - 'confidence': float (0-1)
                - 'type': 'wall', 'static_obstacle', 'dynamic_obstacle', or 'unknown_obstacle'
        """
        # Convert scan to point cloud
        pointcloud = self.scan_to_pointcloud(ranges, angles, range_max)
        
        if len(pointcloud) == 0:
            return []
        
        # Cluster points into objects
        clusters = self.cluster_points(pointcloud)
        
        if len(clusters) == 0:
            return []
        
        # Estimate circle parameters for each cluster
        neighbors = []
        current_positions = []
        
        for cluster in clusters:
            if len(cluster) < self.min_samples:
                continue
                
            center_x, center_y, radius = self.estimate_circle_parameters(cluster)
            
            if self.min_radius <= radius <= self.max_radius:
                # Classify as wall or obstacle
                neighbor_type = self._classify_object(center_x, center_y, radius, cluster, world_bounds)
                
                current_positions.append((center_x, center_y))
                neighbors.append({
                    'position': (center_x, center_y),
                    'velocity': (0.0, 0.0),  # Will be updated below
                    'radius': radius,
                    'confidence': min(1.0, len(cluster) / self.confidence_points_factor),
                    'type': neighbor_type
                })
        
        # Estimate velocities using history and classify static vs dynamic obstacles
        if len(current_positions) > 0:
            # Estimate velocities in relative coordinates first
            relative_velocities = self.estimate_velocities(current_positions, current_time)
            
            # Transform to absolute coordinates if LiDAR position is provided
            absolute_positions, absolute_velocities = self._transform_to_absolute_coordinates(
                current_positions, relative_velocities,
                lidar_position, lidar_velocity, lidar_orientation
            )
            
            # Update neighbors with absolute coordinates and velocities
            for i, neighbor in enumerate(neighbors):
                if i < len(absolute_velocities):
                    neighbor['position'] = absolute_positions[i]
                    neighbor['velocity'] = absolute_velocities[i]
                    
                    # Classify as static or dynamic based on absolute velocity
                    # Only reclassify if it's not already classified as a wall
                    if neighbor['type'] in ['unknown_obstacle', 'wall']:
                        # Check if object is moving (has significant velocity)
                        speed = np.sqrt(absolute_velocities[i][0]**2 + absolute_velocities[i][1]**2)
                        if speed > self.velocity_threshold:
                            neighbor['type'] = 'dynamic_obstacle'
                        else:
                            # Only classify as static if it's not near boundaries (walls)
                            if world_bounds is not None:
                                x_min, y_min, x_max, y_max = world_bounds
                                abs_pos = absolute_positions[i]
                                near_boundary = (abs_pos[0] <= x_min + self.wall_boundary_threshold or 
                                               abs_pos[0] >= x_max - self.wall_boundary_threshold or
                                               abs_pos[1] <= y_min + self.wall_boundary_threshold or 
                                               abs_pos[1] >= y_max - self.wall_boundary_threshold)
                                if near_boundary:
                                    neighbor['type'] = 'wall'
                                else:
                                    neighbor['type'] = 'static_obstacle'
                            else:
                                neighbor['type'] = 'static_obstacle'
        else:
            # No positions to process, but still need to transform if LiDAR position provided
            if lidar_position is not None:
                for neighbor in neighbors:
                    # Transform position only (velocity is already 0.0)
                    rel_pos = neighbor['position']
                    rel_vel = neighbor['velocity']
                    abs_positions, abs_velocities = self._transform_to_absolute_coordinates(
                        [rel_pos], [rel_vel], lidar_position, lidar_velocity, lidar_orientation
                    )
                    neighbor['position'] = abs_positions[0]
                    neighbor['velocity'] = abs_velocities[0]
        
        # Filter duplicates
        neighbors = self._filter_duplicates(neighbors)
        
        # Store current positions for future velocity estimation
        self.position_history.append(current_positions)
        self.timestamp_history.append(current_time)
        
        # Keep only recent history
        if len(self.position_history) > self.velocity_history_length:
            self.position_history.pop(0)
            self.timestamp_history.pop(0)
        
        return neighbors
    
    def _classify_object(self, center_x: float, center_y: float, radius: float, 
                        cluster: np.ndarray, world_bounds: Optional[tuple] = None) -> str:
        """
        Classify detected object as 'wall', 'static_obstacle', or 'dynamic_obstacle'.
        
        Args:
            center_x, center_y: Object center position
            radius: Estimated radius
            cluster: Point cloud cluster
            world_bounds: Optional (x_min, y_min, x_max, y_max) for wall detection
        
        Returns:
            str: 'wall', 'static_obstacle', or 'dynamic_obstacle'
        """
        # Method 1: Check if object is near world boundaries (WALLS)
        if world_bounds is not None:
            x_min, y_min, x_max, y_max = world_bounds
            
            # Check if the detected points are near any world boundary
            # This is more accurate than checking just the center
            cluster_x, cluster_y = cluster[:, 0], cluster[:, 1]
            
            # Check if any points are near boundaries
            near_north = np.any(cluster_y >= y_max - self.wall_boundary_threshold)
            near_south = np.any(cluster_y <= y_min + self.wall_boundary_threshold)
            near_east = np.any(cluster_x >= x_max - self.wall_boundary_threshold)
            near_west = np.any(cluster_x <= x_min + self.wall_boundary_threshold)
            
            near_boundary = near_north or near_south or near_east or near_west
            
            if near_boundary:
                # Additional checks to confirm it's actually a wall
                if len(cluster) >= 3:  # Reduced requirement
                    # Calculate bounding box
                    min_x, min_y = np.min(cluster, axis=0)
                    max_x, max_y = np.max(cluster, axis=0)
                    
                    width = max_x - min_x
                    height = max_y - min_y
                    
                    # Calculate aspect ratio
                    aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1.0
                    
                    # Walls near boundaries with reasonable aspect ratio
                    if aspect_ratio > self.wall_aspect_ratio_threshold:
                        return 'wall'
                    # Very large objects near boundaries are likely walls
                    elif radius > self.wall_min_radius:
                        return 'wall'
                    # Even smaller objects near boundaries could be walls
                    elif len(cluster) >= 5:  # Sufficient points for wall detection
                        return 'wall'
        
        # Method 2: Check cluster shape characteristics for walls
        if len(cluster) >= 4:  # Reduced requirement
            # Calculate bounding box
            min_x, min_y = np.min(cluster, axis=0)
            max_x, max_y = np.max(cluster, axis=0)
            
            width = max_x - min_x
            height = max_y - min_y
            
            # Calculate aspect ratio
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1.0
            
            # Walls must have high aspect ratio AND be reasonably large
            if aspect_ratio > self.wall_aspect_ratio_threshold and radius > self.wall_min_radius:
                # Additional check: points should be spread out
                distances_from_center = np.linalg.norm(cluster - np.array([center_x, center_y]), axis=1)
                max_distance = np.max(distances_from_center)
                
                if max_distance > radius * self.wall_point_spread_factor:
                    return 'wall'
        
        # Method 3: Very large objects (likely walls)
        if radius > self.wall_max_radius and len(cluster) >= 5:
            return 'wall'
        
        # Method 4: Check for wall-like characteristics in point distribution
        if len(cluster) >= 6:
            # Calculate how spread out the points are
            distances_from_center = np.linalg.norm(cluster - np.array([center_x, center_y]), axis=1)
            max_distance = np.max(distances_from_center)
            mean_distance = np.mean(distances_from_center)
            
            # If points are very spread out relative to radius, likely a wall
            if max_distance > radius * self.wall_point_spread_factor and mean_distance > radius * 0.8:
                return 'wall'
        
        # Default: classify as unknown obstacle (will be determined by velocity later)
        return 'unknown_obstacle'
    
    def _filter_duplicates(self, neighbors: List[Dict]) -> List[Dict]:
        """Filter out duplicate detections of the same object."""
        if len(neighbors) <= 1:
            return neighbors
        
        filtered = []
        for neighbor in neighbors:
            is_duplicate = False
            for existing in filtered:
                # Check if positions are too close (same object)
                distance = np.linalg.norm(
                    np.array(neighbor['position']) - np.array(existing['position'])
                )
                if distance < self.duplicate_distance_threshold:
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if neighbor['confidence'] > existing['confidence']:
                        filtered.remove(existing)
                        filtered.append(neighbor)
                    break
            
            if not is_duplicate:
                filtered.append(neighbor)
        
        return filtered
    
    def neighbors_to_rvo_format(self, neighbors: List[Dict], include_static: bool = False) -> List[List[float]]:
        """
        Convert neighbor dictionaries to RVO format.
        
        Args:
            neighbors: List of neighbor dictionaries
            include_static: Whether to include static obstacles in RVO neighbors (default: False)
        
        Returns:
            List[List[float]]: List of RVO neighbor states [x, y, vx, vy, radius]
        """
        rvo_neighbors = []
        
        for neighbor in neighbors:
            neighbor_type = neighbor.get('type', 'unknown_obstacle')
            
            # Only include dynamic obstacles by default
            # Walls are never included (they're environment boundaries)
            # Static obstacles can be optionally included
            if neighbor_type == 'dynamic_obstacle':
                rvo_neighbors.append([neighbor['position'][0], neighbor['position'][1], 
                                    neighbor['velocity'][0], neighbor['velocity'][1], 
                                    neighbor['radius']])
            elif neighbor_type == 'static_obstacle' and include_static:
                # Static obstacles have zero velocity
                rvo_neighbors.append([neighbor['position'][0], neighbor['position'][1], 
                                    0.0, 0.0, neighbor['radius']])
        
        return rvo_neighbors


def extract_rvo_neighbors_simple(ranges: np.ndarray, 
                               angles: np.ndarray,
                               eps: float = 0.3,
                               min_samples: int = 3,
                               min_radius: float = 0.1,
                               max_radius: float = 2.0,
                               range_max: float = None) -> List[List[float]]:
    """
    Simple LiDAR neighbor extraction without velocity estimation.
    
    Args:
        ranges: Array of range measurements from LiDAR
        angles: Array of corresponding angle measurements
        eps: Clustering distance threshold
        min_samples: Minimum points per cluster
        min_radius: Minimum expected radius
        max_radius: Maximum expected radius
        range_max: Optional maximum range value - ranges >= range_max are treated as "no detection"
        
    Returns:
        List[List[float]]: List of RVO neighbor states [x, y, vx, vy, radius]
    """
    # Convert LiDAR scan to point cloud
    valid_mask = np.isfinite(ranges) & (ranges > 0)
    
    # If range_max is provided, exclude range_max values (these indicate no detection)
    if range_max is not None:
        valid_mask = valid_mask & (ranges < range_max)
    
    valid_ranges = ranges[valid_mask]
    valid_angles = angles[valid_mask]
    
    if len(valid_ranges) == 0:
        return []
    
    x = valid_ranges * np.cos(valid_angles)
    y = valid_ranges * np.sin(valid_angles)
    pointcloud = np.column_stack([x, y])
    
    if len(pointcloud) < min_samples:
        return []
    
    # Simple clustering
    clusters = []
    used_points = set()
    
    for i, point in enumerate(pointcloud):
        if i in used_points:
            continue
            
        cluster = [point]
        used_points.add(i)
        
        for j, other_point in enumerate(pointcloud):
            if j in used_points:
                continue
                
            distance = np.linalg.norm(point - other_point)
            if distance <= eps:
                cluster.append(other_point)
                used_points.add(j)
        
        if len(cluster) >= min_samples:
            clusters.append(np.array(cluster))
    
    # Estimate circle parameters for each cluster
    rvo_neighbors = []
    
    for cluster in clusters:
        if len(cluster) < min_samples:
            continue
            
        # Improved circle estimation for LiDAR data
        center = np.mean(cluster, axis=0)
        
        # For LiDAR data, points are on the surface of the obstacle
        # The radius should be the distance from center to the furthest point
        distances_from_center = np.linalg.norm(cluster - center, axis=1)
        radius = np.max(distances_from_center)  # Use max distance, not mean
        
        # Alternative: Use 90th percentile to be more robust to outliers
        # radius = np.percentile(distances_from_center, 90)
        
        if min_radius <= radius <= max_radius:
            rvo_neighbors.append([center[0], center[1], 0.0, 0.0, radius])
    
    return rvo_neighbors
