"""
Wander Behavior Extension for IR-SIM

This module adds wander behavior functionality, allowing objects to move
to random goals within a specified range.
"""

import numpy as np
from irsim.lib import register_behavior
from irsim.global_param import world_param


def generate_random_goal(range_low, range_high):
    """
    Generate a random goal position within the specified range.
    
    Args:
        range_low (list): Lower bounds [x, y, theta]
        range_high (list): Upper bounds [x, y, theta]
    
    Returns:
        np.array: Random goal [x, y, theta] (3x1)
    """
    x = np.random.uniform(range_low[0], range_high[0])
    y = np.random.uniform(range_low[1], range_high[1])
    theta = np.random.uniform(range_low[2], range_high[2])
    
    return np.array([[x], [y], [theta]])


def check_and_update_wander_goal(ego_object, range_low, range_high, goal_threshold):
    """
    Check if object reached its goal and generate a new random goal.
    
    Args:
        ego_object: The ego robot object
        range_low (list): Lower bounds for random goal
        range_high (list): Upper bounds for random goal
        goal_threshold (float): Distance threshold to consider goal reached
    
    Returns:
        bool: True if new goal was generated
    """
    if ego_object._goal is None:  # Use _goal private variable
        # Generate initial random goal
        new_goal = generate_random_goal(range_low, range_high)
        ego_object.set_goal(new_goal.flatten().tolist())  # Use set_goal method
        return True
    
    # Check if goal is reached
    state_pos = ego_object.state[0:2, 0]
    goal_pos = ego_object.goal[0:2, 0]
    distance = np.linalg.norm(state_pos - goal_pos)
    
    if distance < goal_threshold:
        # Generate new random goal
        new_goal = generate_random_goal(range_low, range_high)
        ego_object.set_goal(new_goal.flatten().tolist())  # Use set_goal method
        return True
    
    return False


@register_behavior("diff", "wander")
def beh_diff_wander(ego_object, external_objects, **kwargs):
    """
    Wander behavior for differential drive robots.
    
    The robot moves to random goals within the specified range.
    When it reaches a goal, a new random goal is generated.
    
    Args:
        ego_object: The ego robot object
        external_objects (list): List of external objects
        **kwargs: Additional keyword arguments:
            - range_low (list): Lower bounds [x, y, theta], default [0, 0, -3.14]
            - range_high (list): Upper bounds [x, y, theta], default [100, 100, 3.14]
            - angle_tolerance (float): Allowable angular deviation, default 0.1
            - wander_goal_threshold (float): Goal threshold for wander, default is 2.0
    
    Returns:
        np.array: Velocity [linear, angular] (2x1)
    """
    from irsim.lib.behavior.behavior_methods import DiffDash
    
    range_low = kwargs.get("range_low", [0, 0, -np.pi])
    range_high = kwargs.get("range_high", [100, 100, np.pi])
    angle_tolerance = kwargs.get("angle_tolerance", 0.1)
    wander_goal_threshold = kwargs.get("wander_goal_threshold", 2.0)
    
    # Update goal if needed
    check_and_update_wander_goal(ego_object, range_low, range_high, wander_goal_threshold)
    
    # Use DiffDash to move to current goal
    state = ego_object.state
    goal = ego_object.goal
    _, max_vel = ego_object.get_vel_range()
    
    behavior_vel = DiffDash(state, goal, max_vel, wander_goal_threshold, angle_tolerance)
    
    return behavior_vel


@register_behavior("omni", "wander")
def beh_omni_wander(ego_object, external_objects, **kwargs):
    """
    Wander behavior for omnidirectional robots.
    
    Args:
        ego_object: The ego robot object
        external_objects (list): List of external objects
        **kwargs: Additional keyword arguments:
            - range_low (list): Lower bounds [x, y, theta]
            - range_high (list): Upper bounds [x, y, theta]
            - wander_goal_threshold (float): Goal threshold, default 2.0
    
    Returns:
        np.array: Velocity [vx, vy] (2x1)
    """
    from irsim.lib.behavior.behavior_methods import OmniDash
    
    range_low = kwargs.get("range_low", [0, 0, -np.pi])
    range_high = kwargs.get("range_high", [100, 100, np.pi])
    wander_goal_threshold = kwargs.get("wander_goal_threshold", 2.0)
    
    # Update goal if needed
    check_and_update_wander_goal(ego_object, range_low, range_high, wander_goal_threshold)
    
    # Use OmniDash to move to current goal
    state = ego_object.state
    goal = ego_object.goal
    _, max_vel = ego_object.get_vel_range()
    
    behavior_vel = OmniDash(state, goal, max_vel, wander_goal_threshold)
    
    return behavior_vel


@register_behavior("otter_usv", "wander")
def beh_otter_wander(ego_object, external_objects, **kwargs):
    """
    Wander behavior for Otter USV.
    
    Args:
        ego_object: The ego robot object
        external_objects (list): List of external objects
        **kwargs: Additional keyword arguments:
            - range_low (list): Lower bounds [x, y, theta]
            - range_high (list): Upper bounds [x, y, theta]
            - angle_tolerance (float): Allowable angular deviation
            - wander_goal_threshold (float): Goal threshold, default 2.0
    
    Returns:
        np.array: Velocity [u_ref, r_ref] (2x1)
    """
    from irsim.lib.behavior.behavior_methods import DiffDash
    
    range_low = kwargs.get("range_low", [0, 0, -np.pi])
    range_high = kwargs.get("range_high", [100, 100, np.pi])
    angle_tolerance = kwargs.get("angle_tolerance", 0.1)
    wander_goal_threshold = kwargs.get("wander_goal_threshold", 2.0)
    
    # Update goal if needed
    check_and_update_wander_goal(ego_object, range_low, range_high, wander_goal_threshold)
    
    # Use DiffDash to move to current goal
    state = ego_object.state
    goal = ego_object.goal
    _, max_vel = ego_object.get_vel_range()
    
    behavior_vel = DiffDash(state, goal, max_vel, wander_goal_threshold, angle_tolerance)
    
    return behavior_vel


@register_behavior("acker", "wander")
def beh_acker_wander(ego_object, external_objects, **kwargs):
    """
    Wander behavior for Ackermann steering robots.
    
    Args:
        ego_object: The ego robot object
        external_objects (list): List of external objects
        **kwargs: Additional keyword arguments:
            - range_low (list): Lower bounds [x, y, theta]
            - range_high (list): Upper bounds [x, y, theta]
            - angle_tolerance (float): Allowable angular deviation
            - wander_goal_threshold (float): Goal threshold, default 2.0
    
    Returns:
        np.array: Velocity [linear, steering] (2x1)
    """
    from irsim.lib.behavior.behavior_methods import AckerDash
    
    range_low = kwargs.get("range_low", [0, 0, -np.pi])
    range_high = kwargs.get("range_high", [100, 100, np.pi])
    angle_tolerance = kwargs.get("angle_tolerance", 0.1)
    wander_goal_threshold = kwargs.get("wander_goal_threshold", 2.0)
    
    # Update goal if needed
    check_and_update_wander_goal(ego_object, range_low, range_high, wander_goal_threshold)
    
    # Use AckerDash to move to current goal
    state = ego_object.state
    goal = ego_object.goal
    _, max_vel = ego_object.get_vel_range()
    
    behavior_vel = AckerDash(state, goal, max_vel, wander_goal_threshold, angle_tolerance)
    
    return behavior_vel
