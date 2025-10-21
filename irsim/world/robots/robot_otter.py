"""
robot_otter.py

Otter USV robot class for IR-SIM with full 6-DOF dynamics integration.

This class integrates the Maritime Robotics Otter USV from Python Vehicle Simulator
into the IR-SIM framework, providing realistic marine craft dynamics for autonomous
navigation research.

Reference:
    T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and Motion Control.
    2nd Edition, Wiley. https://wiley.fossen.biz

Author: Integration for IR-SIM
Date: 2025-10-17
"""

import numpy as np
from irsim.world import ObjectBase

from python_vehicle_simulator.vehicles import otter as OtterVehicle

class RobotOtter(ObjectBase):
    """
    Otter USV robot class for IR-SIM.
    
    This class represents the Maritime Robotics Otter USV with full 6-DOF dynamics
    or simplified kinematics (fallback if Python Vehicle Simulator is not available).
    
    The Otter USV is a 2.0m long unmanned surface vehicle controlled by two propellers.
    
    Attributes:
        otter_vehicle: Python Vehicle Simulator Otter object (if available)
        otter_dynamics: Dictionary containing dynamics state for kinematics function
        state_dim: Extended to 8 for full state [x, y, psi, u, v, r, n1, n2]
        
    Args:
        kinematics (dict): Kinematics configuration, should include 'name': 'otter_usv'
        use_full_dynamics (bool): Whether to use full 6-DOF dynamics (default True)
        **kwargs: Additional arguments passed to ObjectBase
        
    Example:
        >>> robot = RobotOtter(
        ...     kinematics={'name': 'otter_usv'},
        ...     shape={'name': 'rectangle', 'length': 2.0, 'width': 1.08},
        ...     state=[10, 10, 0],
        ...     velocity=[1.5, 0.2],
        ...     use_full_dynamics=True
        ... )
    """
    
    def __init__(
        self,
        kinematics=None,
        use_full_dynamics=True,
        color="b",
        state_dim=8,  # Extended state for Otter
        **kwargs
    ):
        # Set default kinematics if not provided
        if kinematics is None:
            kinematics = {'name': 'otter_usv'}
        
        # Ensure kinematics name is otter_usv
        if 'name' not in kinematics or kinematics['name'] != 'otter_usv':
            kinematics['name'] = 'otter_usv'
        
        # Initialize Otter Vehicle Simulator integration BEFORE parent init
        self.use_full_dynamics = use_full_dynamics
        self.otter_vehicle = None
        self.otter_dynamics = None
        
        if self.use_full_dynamics:
            self._init_otter_dynamics_pre()
        
        # Pass otter_dynamics to parent through kinematics
        kinematics['otter_dynamics'] = self.otter_dynamics
        
        # Initialize base class
        super(RobotOtter, self).__init__(
            role="robot",
            kinematics=kinematics,
            color=color,
            state_dim=state_dim,
            **kwargs,
        )
        
        # Validate state dimension
        assert (
            state_dim >= 3
        ), "For Otter USV, the state dimension should be at least 3 (x, y, psi)"
        
        # Update otter_dynamics with initial state after parent init
        if self.use_full_dynamics and self.otter_dynamics is not None:
            self._update_otter_from_state()
        else:
            print("Otter USV: Using simplified kinematics (full dynamics not available)")
        
        # Store current velocities for dynamics
        self._current_u = 0.0  # surge velocity
        self._current_v = 0.0  # sway velocity
        self._current_r = 0.0  # yaw rate
        
        # Store last commanded velocities for plotting
        self._last_u_ref = 0.0
        self._last_r_ref = 0.0
    
    def _init_otter_dynamics_pre(self):
        """
        Pre-initialize the Otter USV dynamics (before parent __init__).
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
            
            print("Otter USV: Full 6-DOF dynamics initialized successfully")
            
        except Exception as e:
            print(f"Warning: Failed to initialize Otter dynamics: {e}")
            print("Falling back to simplified kinematics")
            self.use_full_dynamics = False
            self.otter_vehicle = None
            self.otter_dynamics = None
    
    def _update_otter_from_state(self):
        """
        Update otter_dynamics from current IR-SIM state.
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
    
    def _init_plot(self, ax, **kwargs):
        """
        Initialize plotting for Otter USV.
        
        Args:
            ax: Matplotlib axis object
            **kwargs: Additional plotting arguments
        """
        show_goal = self.plot_kwargs.get("show_goal", True)
        show_arrow = self.plot_kwargs.get("show_arrow", True)
        
        # Call parent _init_plot
        super()._init_plot(
            ax, 
            show_goal=show_goal, 
            show_arrow=show_arrow,
            **kwargs
        )
    
    def step(self, action=None):
        """
        Override step method to handle Otter-specific dynamics update.
        
        Args:
            action: Velocity commands [u_ref, r_ref] or None for behavior-based control
        """
        # Store action for velocity tracking plot
        if action is not None and action.shape[0] >= 2:
            self.set_commanded_velocities(action[0, 0], action[1, 0])
        else:
            # If no action, use current velocity as reference (for behavior-based control)
            u_ref = self.state[3, 0] if self.state.shape[0] >= 4 else 0.0
            r_ref = self.state[5, 0] if self.state.shape[0] >= 6 else 0.0
            self.set_commanded_velocities(u_ref, r_ref)
        
        # Update otter_dynamics with current state before kinematics
        if self.use_full_dynamics and self.otter_dynamics is not None:
            self._update_otter_from_state()
        
        # Call parent step method (handles kinematics)
        super().step(action)
        
        # Extract updated velocities from state
        if self.state.shape[0] >= 6:
            self._current_u = self.state[3, 0]
            self._current_v = self.state[4, 0]
            self._current_r = self.state[5, 0]
    
    def get_velocities(self):
        """
        Get current body-fixed velocities.
        
        Returns:
            dict: Dictionary with 'u' (surge), 'v' (sway), 'r' (yaw rate)
        """
        return {
            'u': self._current_u,
            'v': self._current_v,
            'r': self._current_r
        }
    
    def get_propeller_states(self):
        """
        Get current propeller shaft speeds (if using full dynamics).
        
        Returns:
            tuple: (n1, n2) propeller speeds in rad/s, or (0, 0) if not available
        """
        if self.state.shape[0] >= 8:
            return self.state[6, 0], self.state[7, 0]
        return 0.0, 0.0
    
    def set_otter_state(self, x=None, y=None, psi=None, u=None, v=None, r=None):
        """
        Convenience method to set Otter USV state.
        
        Args:
            x, y, psi: Position and heading
            u, v, r: Body-fixed velocities
        """
        if x is not None:
            self.state[0, 0] = x
        if y is not None:
            self.state[1, 0] = y
        if psi is not None and self.state.shape[0] >= 3:
            self.state[2, 0] = psi
        if u is not None and self.state.shape[0] >= 4:
            self.state[3, 0] = u
        if v is not None and self.state.shape[0] >= 5:
            self.state[4, 0] = v
        if r is not None and self.state.shape[0] >= 6:
            self.state[5, 0] = r
        
        # Update otter_dynamics if available
        if self.use_full_dynamics and self.otter_dynamics is not None:
            if x is not None:
                self.otter_dynamics['eta'][0] = x
            if y is not None:
                self.otter_dynamics['eta'][1] = y
            if psi is not None:
                self.otter_dynamics['eta'][5] = psi
            if u is not None:
                self.otter_dynamics['nu'][0] = u
            if v is not None:
                self.otter_dynamics['nu'][1] = v
            if r is not None:
                self.otter_dynamics['nu'][5] = r