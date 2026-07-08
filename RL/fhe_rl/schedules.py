import numpy as np

def step_schedule(total_timesteps, transition_point=0.75, max_value=0.01):
    """Step function: 0 until transition_point, then 0.01"""
    def schedule(timestep):
        if timestep >= total_timesteps * transition_point:
            return max_value
        return 0.0
    return schedule


def linear_schedule(total_timesteps, start_point=0.75, saturation_point=0.9, max_value=0.01):
    """Linear increase from start_point to end"""
    def schedule(timestep):
        if timestep < total_timesteps * start_point:
            return 0.0
        elif timestep < total_timesteps * saturation_point:
            progress = (timestep - total_timesteps * start_point) / (total_timesteps * (saturation_point - start_point))
            return max_value * progress
        else:
            return max_value
    return schedule


def sigmoid_schedule(total_timesteps, midpoint=0.7, saturation_point=0.8, steepness=10, max_value=0.01):
    """Smooth sigmoid transition around midpoint"""
    def schedule(timestep):
        if timestep < total_timesteps * midpoint:
            return 0.0
        elif timestep < total_timesteps * saturation_point:
            x = (timestep / total_timesteps - midpoint) * steepness
            return max_value / (1.0 + np.exp(-x))
        else:
            return max_value
    return schedule


def cosine_schedule(total_timesteps, start_point=0.75, saturation_point=0.9, max_value=0.01):
    """Smooth cosine transition from start_point"""
    def schedule(timestep):
        if timestep < total_timesteps * start_point:
            return 0.0
        elif timestep < total_timesteps * saturation_point:
            progress = (timestep - total_timesteps * start_point) / (total_timesteps * (1 - start_point))
            return max_value * 0.5 * (1 - np.cos(np.pi * min(1.0, progress)))
        else:
            return max_value
    return schedule