class BaseSim:
    """
    BaseSim serves as the foundation for all simulation classes, providing common functionality
    like state tracking, reset logic, and stepping through simulation time.

    Attributes:
        step_index (int): Tracks the current step in the simulation.
    """

    def __init__(self):
        """
        Initializes the base simulation with a default step index.
        """
        self.step_index = 0

    def reset(self):
        """
        Resets the simulation to the initial state.
        """
        self.step_index = 0

    def step(self):
        """
        Advances the simulation by one time step. This method should be overridden by subclasses.
        """
        self.step_index += 1
