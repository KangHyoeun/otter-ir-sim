'''
This file is the implementation of the kinematics for different robots.

reference: Lynch, Kevin M., and Frank C. Park. Modern Robotics: Mechanics, Planning, and Control. 1st ed. Cambridge, MA: Cambridge University Press, 2017.
'''

import numpy as np
from math import cos, sin, tan
from irsim.util.util import WrapToPi

def differential_kinematics(
    state, velocity, step_time, noise=False, alpha=[0.03, 0, 0, 0.03]
):
    """
    Calculate the next state for a differential wheel robot.

    Args:
        state: A 3x1 vector [x, y, theta] representing the current position and orientation.
        velocity: A 2x1 vector [linear, angular] representing the current velocities.
        step_time: The time step for the simulation.
        noise: Boolean indicating whether to add noise to the velocity (default False).
        alpha: List of noise parameters for the velocity model (default [0.03, 0, 0, 0.03]). alpha[0] and alpha[1] are for linear velocity, alpha[2] and alpha[3] are for angular velocity.

    Returns:
        next_state: A 3x1 vector [x, y, theta] representing the next state.
    """
    assert state.shape[0] >= 3 and velocity.shape[0] >= 2

    if noise:
        assert len(alpha) >= 4
        std_linear = np.sqrt(
            alpha[0] * (velocity[0, 0] ** 2) + alpha[1] * (velocity[1, 0] ** 2)
        )
        std_angular = np.sqrt(
            alpha[2] * (velocity[0, 0] ** 2) + alpha[3] * (velocity[1, 0] ** 2)
        )
        real_velocity = velocity + np.random.normal(
            [[0], [0]], scale=[[std_linear], [std_angular]]
        )
    else:
        real_velocity = velocity

    phi = state[2, 0]
    co_matrix = np.array([[cos(phi), 0], [sin(phi), 0], [0, 1]])
    next_state = state[0:3] + co_matrix @ real_velocity * step_time
    next_state[2, 0] = WrapToPi(next_state[2, 0])

    return next_state


def ackermann_kinematics(
    state,
    velocity,
    step_time,
    noise=False,
    alpha=[0.03, 0, 0, 0.03],
    mode="steer",
    wheelbase=1,
):
    """
    Calculate the next state for an Ackermann steering vehicle.

    Args:
        state: A 4x1 vector [x, y, theta, steer_angle] representing the current state.
        velocity: A 2x1 vector representing the current velocities, format depends on mode.
            For "steer" mode, [linear, steer_angle] is expected.
            For "angular" mode, [linear, angular] is expected.

        step_time: The time step for the simulation.
        noise: Boolean indicating whether to add noise to the velocity (default False).
        alpha: List of noise parameters for the velocity model (default [0.03, 0, 0, 0.03]). alpha[0] and alpha[1] are for linear velocity, alpha[2] and alpha[3] are for angular velocity.
        mode: The kinematic mode, either "steer" or "angular" (default "steer").
        wheelbase: The distance between the front and rear axles (default 1).

    Returns:
        new_state: A 4x1 vector representing the next state.
    """
    assert state.shape[0] >= 4 and velocity.shape[0] >= 2

    phi = state[2, 0]
    psi = state[3, 0]

    if noise:
        assert len(alpha) >= 4
        std_linear = np.sqrt(
            alpha[0] * (velocity[0, 0] ** 2) + alpha[1] * (velocity[1, 0] ** 2)
        )
        std_angular = np.sqrt(
            alpha[2] * (velocity[0, 0] ** 2) + alpha[3] * (velocity[1, 0] ** 2)
        )
        real_velocity = velocity + np.random.normal(
            [[0], [0]], scale=[[std_linear], [std_angular]]
        )
    else:
        real_velocity = velocity

    if mode == "steer":
        co_matrix = np.array(
            [[cos(phi), 0], [sin(phi), 0], [tan(psi) / wheelbase, 0], [0, 1]]
        )
    elif mode == "angular":
        co_matrix = np.array(
            [[cos(phi), 0], [sin(phi), 0], [tan(psi) / wheelbase, 0], [0, 1]]
        )

    d_state = co_matrix @ real_velocity
    new_state = state + d_state * step_time

    if mode == "steer":
        new_state[3, 0] = real_velocity[1, 0]

    new_state[2, 0] = WrapToPi(new_state[2, 0])

    return new_state


def omni_kinematics(state, velocity, step_time, noise=False, alpha=[0.03, 0, 0, 0.03]):
    """
    Calculate the next position for an omnidirectional robot.

    Args:
        state: A 2x1 vector [x, y] representing the current position.
        velocity: A 2x1 vector [vx, vy] representing the current velocities.
        step_time: The time step for the simulation.
        noise: Boolean indicating whether to add noise to the velocity (default False).
        alpha: List of noise parameters for the velocity model (default [0.03, 0.03]). alpha[0] is for x velocity, alpha[1] is for y velocity.

    Returns:
        new_position: A 2x1 vector [x, y] representing the next position.
    """

    assert velocity.shape[0] >= 2 and state.shape[0] >= 2

    if noise:
        assert len(alpha) >= 2
        std_vx = np.sqrt(alpha[0])
        std_vy = np.sqrt(alpha[-1])
        real_velocity = velocity + np.random.normal(
            [[0], [0]], scale=[[std_vx], [std_vy]]
        )
    else:
        real_velocity = velocity

    new_position = state[0:2] + real_velocity * step_time

    return new_position


def otter_usv_kinematics(state, velocity, step_time, otter_dynamics=None, noise=False, alpha=[0.03, 0, 0, 0.03]):   
    """
    Calculate the next state for an Otter USV using full 6-DOF dynamics with velocity controller.
    
    This implements the Maritime Robotics Otter USV model from:
    T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and Motion Control. 2nd Edition, Wiley.
    
    Args:
        state: Extended state vector [x, y, psi, u, v, r, u_actual1, u_actual2] (8x1)
            - [x, y, psi]: Position and heading (m, m, rad)
            - [u, v, r]: Body-fixed velocities (m/s, m/s, rad/s)
            - [u_actual1, u_actual2]: Propeller states (rad/s)
        velocity: Desired velocity commands [u_ref, r_ref] (2x1)
            - u_ref: desired surge velocity (m/s)
            - r_ref: desired yaw rate (rad/s)
        step_time: Time step for simulation (s)
        otter_dynamics: Dictionary containing Otter dynamics object and states
            - 'otter': otter vehicle object from Python Vehicle Simulator
            - 'eta': 6-DOF position/orientation state
            - 'nu': 6-DOF velocity state
            If None, uses simplified kinematics without full dynamics
        noise: Boolean indicating whether to add noise to the velocity (default False)
        alpha: Noise parameters for velocity model (default [0.03, 0, 0, 0.03])
    
    Returns:
        next_state: Updated state vector (8x1) with same structure as input state
        
    Note:
        - For IR-SIM integration, only [x, y, psi] are used for visualization
        - Full state is maintained for accurate dynamics simulation
        - If otter_dynamics is None, falls back to differential kinematics
    """
    
    # Validate state dimensions
    if state.shape[0] < 3:
        raise ValueError(f"State must have at least 3 dimensions, got {state.shape[0]}")
    
    if velocity.shape[0] < 2:
        raise ValueError(f"Velocity must have 2 dimensions, got {velocity.shape[0]}")
    
    # If otter_dynamics not provided, use simplified differential kinematics
    if otter_dynamics is None:
        # Simple kinematic model without full dynamics
        if state.shape[0] < 6:
            # Initialize extended state if needed
            next_state = np.zeros((8, 1))
            next_state[0:3] = state[0:3]
        else:
            next_state = state.copy()
        
        # Apply velocity commands directly (simplified)
        u_ref = velocity[0, 0]
        r_ref = velocity[1, 0]
        
        if noise:
            std_linear = np.sqrt(
                alpha[0] * (u_ref ** 2) + alpha[1] * (r_ref ** 2)
            )
            std_angular = np.sqrt(
                alpha[2] * (u_ref ** 2) + alpha[3] * (r_ref ** 2)
            )
            u_ref += np.random.normal(0, std_linear)
            r_ref += np.random.normal(0, std_angular)
        
        # Simple first-order response (no full dynamics)
        if state.shape[0] >= 6:
            # Update velocities with simple dynamics
            tau_u = 2.0  # surge time constant (s)
            tau_r = 1.0  # yaw time constant (s)
            
            next_state[3, 0] += (u_ref - state[3, 0]) / tau_u * step_time  # u
            next_state[4, 0] = 0.0  # v (sway)
            next_state[5, 0] += (r_ref - state[5, 0]) / tau_r * step_time  # r
        else:
            # Use commands directly
            next_state[3, 0] = u_ref
            next_state[4, 0] = 0.0
            next_state[5, 0] = r_ref
        
        # Update position using current velocities
        psi = state[2, 0]
        u = next_state[3, 0]
        v = next_state[4, 0]
        r = next_state[5, 0]
        
        next_state[0, 0] = state[0, 0] + (u * cos(psi) - v * sin(psi)) * step_time
        next_state[1, 0] = state[1, 0] + (u * sin(psi) + v * cos(psi)) * step_time
        next_state[2, 0] = state[2, 0] + r * step_time
        next_state[2, 0] = WrapToPi(next_state[2, 0])
        
        return next_state
    
    # Full Otter USV dynamics with Python Vehicle Simulator
    otter = otter_dynamics['otter']
    eta = otter_dynamics['eta']  # 6-DOF position/orientation
    nu = otter_dynamics['nu']    # 6-DOF velocities
    u_actual = otter_dynamics.get('u_actual', np.zeros(2))  # Propeller states
    
    # Extract velocity commands
    u_ref = velocity[0, 0]
    r_ref = velocity[1, 0]
    
    # Apply noise if requested
    if noise:
        std_linear = np.sqrt(alpha[0] * (u_ref ** 2) + alpha[1] * (r_ref ** 2))
        std_angular = np.sqrt(alpha[2] * (u_ref ** 2) + alpha[3] * (r_ref ** 2))
        u_ref += np.random.normal(0, std_linear)
        r_ref += np.random.normal(0, std_angular)
    
    # Otter velocity controller
    u_control = otter.velocityControl(nu, u_ref, r_ref, step_time)
    
    # Otter dynamics (6-DOF)
    [nu_next, u_actual_next] = otter.dynamics(eta, nu, u_actual, u_control, step_time)
    
    # Update position using kinematic equations
    # For surface vessel: z=0, phi=0, theta=0, w=0, p=0, q=0
    eta_next = eta.copy()
    eta_next[0] += step_time * (nu_next[0] * np.cos(eta[5]) - nu_next[1] * np.sin(eta[5]))
    eta_next[1] += step_time * (nu_next[0] * np.sin(eta[5]) + nu_next[1] * np.cos(eta[5]))
    eta_next[5] += step_time * nu_next[5]
    eta_next[5] = WrapToPi(eta_next[5])
    
    # Update otter_dynamics for next iteration
    otter_dynamics['eta'] = eta_next
    otter_dynamics['nu'] = nu_next
    otter_dynamics['u_actual'] = u_actual_next
    
    # Pack into extended state for IR-SIM
    next_state = np.zeros((8, 1))
    next_state[0, 0] = eta_next[0]  # x
    next_state[1, 0] = eta_next[1]  # y
    next_state[2, 0] = eta_next[5]  # psi
    next_state[3, 0] = nu_next[0]   # u
    next_state[4, 0] = nu_next[1]   # v
    next_state[5, 0] = nu_next[5]   # r
    next_state[6, 0] = u_actual_next[0]  # n1
    next_state[7, 0] = u_actual_next[1]  # n2
    
    return next_state 
