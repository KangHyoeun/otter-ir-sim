"""
obstacle_otter.py

Otter USV obstacle class for IR-SIM with full 6-DOF dynamics integration.

This obstacle class represents a dynamic Otter USV obstacle with the same
full 6-DOF dynamics as RobotOtter, providing realistic marine craft dynamics
for multi-agent scenarios.

Reference:
    T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and Motion Control.
    2nd Edition, Wiley. https://wiley.fossen.biz

Author: Integration for IR-SIM
Date: 2025-10-21
"""

import numpy as np
from irsim.world import ObjectBase

try:
    from python_vehicle_simulator.vehicles import otter as OtterVehicle
    FULL_DYNAMICS_AVAILABLE = True
except ImportError:
    FULL_DYNAMICS_AVAILABLE = False
    print("Warning: Python Vehicle Simulator not found. ObstacleOtter will use simplified kinematics.")

class ObstacleOtter(ObjectBase):
    """
    Otter USV obstacle class for IR-SIM with full 6-DOF dynamics.
    
    This class represents a dynamic Otter USV obstacle that moves with realistic
    marine craft dynamics, including:
    - Mass, damping, and restoring forces (M, D, G matrices)
    - Velocity controller (PID control for u and r)
    - Propeller dynamics (shaft speed, saturation)
    - Control allocation (thrust forces → propeller commands)
    
    Uses the same extended state as RobotOtter: [x, y, psi, u, v, r, n1, n2].
    
    Args:
        color (str): Color of the obstacle. Defaults to "k" (black).
        state_dim (int): State dimension. Defaults to 8 for extended Otter state.
        use_full_dynamics (bool): Whether to use full 6-DOF dynamics. Defaults to True.
        **kwargs: Additional arguments passed to ObjectBase.
        
    Example:
        >>> obstacle = ObstacleOtter(
        ...     kinematics={'name': 'otter_usv'},
        ...     shape={'name': 'rectangle', 'length': 2.0, 'width': 1.08},
        ...     state=[50, 50, 0, 1.0, 0, 0, 0, 0],
        ...     goal=[90, 90, 0],
        ...     behavior={'name': 'dash'},
        ...     color='r',
        ...     use_full_dynamics=True
        ... )
    """
    
    def __init__(self, color="k", state_dim=8, use_full_dynamics=True, **kwargs):
        """
        Initialize Otter USV obstacle.
        
        Args:
            color (str): Color of the obstacle
            state_dim (int): State dimension (default 8 for Otter)
            use_full_dynamics (bool): Whether to use full 6-DOF dynamics
            **kwargs: Additional arguments for ObjectBase
        """
        # Initialize Otter dynamics BEFORE parent init (like RobotOtter)
        self.use_full_dynamics = use_full_dynamics and FULL_DYNAMICS_AVAILABLE
        self.otter_vehicle = None
        self.otter_dynamics = None
        
        if self.use_full_dynamics:
            self._init_otter_dynamics_pre()
        
        # Pass otter_dynamics to parent through kinematics
        kinematics = kwargs.get('kinematics', {'name': 'otter_usv'})
        kinematics['otter_dynamics'] = self.otter_dynamics
        kwargs['kinematics'] = kinematics
        
        # Initialize base class
        super(ObstacleOtter, self).__init__(
            color=color, 
            role="obstacle", 
            state_dim=state_dim, 
            **kwargs
        )
        
        # Update otter_dynamics with initial state after parent init
        if self.use_full_dynamics and self.otter_dynamics is not None:
            self._update_otter_from_state()
    
    def _init_otter_dynamics_pre(self):
        """
        Pre-initialize the Otter USV dynamics (before parent __init__).
        Same as RobotOtter.
        """
        try:
            # Create Otter vehicle with velocity controller
            self.otter_vehicle = OtterVehicle('velocityControl', r=1.5)
            
            # Initialize dynamics state
            self.otter_dynamics = {
                'otter': self.otter_vehicle,
                'eta': np.zeros(6),  # [x, y, z, phi, theta, psi]
                'nu': np.zeros(6),   # [u, v, w, p, q, r]
                'u_actual': np.zeros(2)  # [n1, n2] propeller states
            }
            
            # Note: Don't print here to avoid spam with multiple obstacles
            
        except Exception as e:
            print(f"Warning: Failed to initialize Otter dynamics for obstacle: {e}")
            print("Falling back to simplified kinematics")
            self.use_full_dynamics = False
            self.otter_vehicle = None
            self.otter_dynamics = None
    
    def _update_otter_from_state(self):
        """
        Update otter_dynamics from current IR-SIM state.
        Same as RobotOtter.
        """
        if self.otter_dynamics is None:
            return
            
        # Set initial position from IR-SIM state
        if hasattr(self, 'state') and self.state is not None:
            self.otter_dynamics['eta'][0] = self.state[0, 0]  # x
            self.otter_dynamics['eta'][1] = self.state[1, 0]  # y
            if self.state.shape[0] >= 3:
                self.otter_dynamics['eta'][5] = self.state[2, 0]  # psi
            if self.state.shape[0] >= 6:
                self.otter_dynamics['nu'][0] = self.state[3, 0]  # u
                self.otter_dynamics['nu'][1] = self.state[4, 0]  # v
                self.otter_dynamics['nu'][5] = self.state[5, 0]  # r
            if self.state.shape[0] >= 8:
                self.otter_dynamics['u_actual'][0] = self.state[6, 0]  # n1
                self.otter_dynamics['u_actual'][1] = self.state[7, 0]  # n2
        
        # Update kinematics handler reference
        if hasattr(self, 'kinematics_model') and self.kinematics_model is not None:
            if hasattr(self.kinematics_model, 'set_otter_dynamics'):
                self.kinematics_model.set_otter_dynamics(self.otter_dynamics)
    
    def step(self, velocity=None):
        """
        Override step method to handle Otter-specific dynamics update.
        Same as RobotOtter but simpler (no velocity text plotting).
        
        Args:
            velocity: Velocity commands [u_ref, r_ref] or None for behavior-based control
        """
        # Update otter_dynamics with current state before kinematics
        if self.use_full_dynamics and self.otter_dynamics is not None:
            self._update_otter_from_state()
        
        # Call parent step method (handles kinematics)
        super().step(velocity)
    
    def get_velocities(self):
        """
        Get current body-fixed velocities.
        
        Returns:
            dict: Dictionary with 'u' (surge), 'v' (sway), 'r' (yaw rate)
        """
        if self.state.shape[0] >= 6:
            return {
                'u': self.state[3, 0],
                'v': self.state[4, 0],
                'r': self.state[5, 0]
            }
        return {'u': 0.0, 'v': 0.0, 'r': 0.0}
    
    def get_propeller_states(self):
        """
        Get current propeller shaft speeds (if using full dynamics).
        
        Returns:
            tuple: (n1, n2) propeller speeds in rad/s, or (0, 0) if not available
        """
        if self.state.shape[0] >= 8:
            return self.state[6, 0], self.state[7, 0]
        return 0.0, 0.0
