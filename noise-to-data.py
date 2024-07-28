import numpy as np
import matplotlib.pyplot as plt

# Parameters
sampling_rate = 100 
duration = 10  
noise_level = 0.1  

# Generate time vector
t = np.linspace(0, duration, int(sampling_rate * duration))

# Generate linear motion trajectory
linear_motion = np.sin(2 * np.pi * t)

# Add Gaussian noise
noise = np.random.normal(0, noise_level, linear_motion.shape)
acceleration_data = linear_motion + noise

# Plot original and noisy data
plt.plot(t, linear_motion, label='Original Motion')
plt.plot(t, acceleration_data, label='Noisy Motion')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.legend()
plt.show()
