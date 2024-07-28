import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Parameters
sampling_rate = 100  # Hz
duration = 10  # seconds
noise_level = 0.1  # Standard deviation of noise

# Generate time vector
t = np.linspace(0, duration, int(sampling_rate * duration))

# Generate linear motion trajectory for accelerometer data
accelerometer_motion = np.sin(2 * np.pi * t)

# Generate rotational motion trajectory for gyroscope data
gyroscope_motion = np.cos(2 * np.pi * t)

# Add noise to accelerometer data
accelerometer_noise = np.random.normal(0, noise_level, accelerometer_motion.shape)
accelerometer_data = accelerometer_motion + accelerometer_noise

# Add noise to gyroscope data
gyroscope_noise = np.random.normal(0, noise_level, gyroscope_motion.shape)
gyroscope_data = gyroscope_motion + gyroscope_noise

accelerometer_labels = np.zeros_like(accelerometer_motion)
gyroscope_labels = np.ones_like(gyroscope_motion)

# Combine accelerometer and gyroscope data
X = np.concatenate([accelerometer_data[:, np.newaxis], gyroscope_data[:, np.newaxis]], axis=1)
y = np.concatenate([accelerometer_labels, gyroscope_labels])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict labels for the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot accelerometer and gyroscope data
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(t, accelerometer_data, label='Accelerometer Data')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, gyroscope_data, label='Gyroscope Data')
plt.xlabel('Time')
plt.ylabel('Angular Velocity')
plt.legend()

plt.tight_layout()
plt.show()
