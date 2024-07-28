import numpy as np
import matplotlib.pyplot as plt

# Parameters
sampling_rate = 100  # Hz
duration = 10  # seconds
noise_level = 0.1  # Standard deviation of noise

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

# Format data (assuming binary classification for simplicity)
# For accelerometer data, label as 0 (representing linear motion)
# For gyroscope data, label as 1 (representing rotational motion)
accelerometer_labels = np.zeros_like(accelerometer_motion)
gyroscope_labels = np.ones_like(gyroscope_motion)

# Combine accelerometer and gyroscope data
X = np.concatenate([accelerometer_data[:, np.newaxis], gyroscope_data[:, np.newaxis]], axis=1)
y = np.concatenate([accelerometer_labels[:, np.newaxis], gyroscope_labels[:, np.newaxis]], axis=1)

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

# Format data (assuming binary classification for simplicity)
# For accelerometer data, label as 0 (representing linear motion)
# For gyroscope data, label as 1 (representing rotational motion)
accelerometer_labels = np.zeros_like(accelerometer_motion)
gyroscope_labels = np.ones_like(gyroscope_motion)

# Combine accelerometer and gyroscope data
X_accelerometer = np.concatenate([accelerometer_data[:, np.newaxis], np.zeros_like(accelerometer_data[:, np.newaxis])], axis=1)
X_gyroscope = np.concatenate([np.zeros_like(gyroscope_data[:, np.newaxis]), gyroscope_data[:, np.newaxis]], axis=1)
X_combined = np.concatenate([X_accelerometer, X_gyroscope], axis=0)

y_combined = np.concatenate([accelerometer_labels[:, np.newaxis], gyroscope_labels[:, np.newaxis]], axis=0)

# Flatten target arrays
y_train_flat = y_train.ravel()
y_test_flat = y_test.ravel()

# Train a simple Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train_flat)


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

!pip install tabgan
!pip install -r requirements.txt
!pip install lightgbm
!pip install tqdm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import io
import os
import requests
import numpy as np
from sklearn import metrics

df_gyroscope = pd.read_csv("sensor_ gyroscope.csv")
df_magnetometer = pd.read_csv("sensor_ magnetometer.csv")
df_accelerometer = pd.read_csv("sensor_ accelerometer.csv")

print(df_gyroscope)
print(df_magnetometer)
print(df_accelerometer)

COLS_USED = ['Walking', 'Running', 'Standing', 'Walking Upstairs',
          'Walking Downstairs']
COLS_TRAIN = ['Walking', 'Running', 'Standing', 'Walking Upstairs',
          'Walking Downstairs']

df = df_gyroscope[COLS_USED]

df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(
    df.drop("Walking", axis=1),
    df["Walking"],
    test_size=0.3,
    random_state=10,
    train_size=0.7,
)


# Create dataframe versions for tabular GAN
df_x_test, df_y_test = df_x_test.reset_index(drop=True), \
  df_y_test.reset_index(drop=True)
df_y_train = pd.DataFrame(df_y_train)
df_y_test = pd.DataFrame(df_y_test)

# Pandas to Numpy
x_train = df_x_train.values
x_test = df_x_test.values
y_train = df_y_train.values
y_test = df_y_test.values

# Build the neural network
model = Sequential()
# Hidden 1
model.add(Dense(50, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(25, activation='relu')) # Hidden 2
model.add(Dense(12, activation='relu')) # Hidden 2
model.add(Dense(1)) # Output
model.compile(loss='mean_squared_error', optimizer='adam')

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3,
        patience=5, verbose=1, mode='auto',
        restore_best_weights=True)
model.fit(x_train,y_train,validation_data=(x_test,y_test),
        callbacks=[monitor], verbose=2,epochs=1000)

pred = model.predict(x_test)
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print("Final score (RMSE): {}".format(score))

from tabgan.sampler import GANGenerator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

gen_x, gen_y = GANGenerator(gen_x_times=1.1, cat_cols=None,
           bot_filter_quantile=0.001, top_filter_quantile=0.999, \
              is_post_process=True,
           adversarial_model_params={
               "metrics": "rmse", "max_depth": 1, "max_bin": 10,
               "learning_rate": 0.02, "random_state": \
                42, "n_estimators": 500,
           }, pregeneration_frac=2, only_generated_data=False).generate_data_pipe(df_x_train, df_y_train,\
          df_x_test, deep_copy=True, only_adversarial=False, )

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


df_gyroscope = pd.read_csv("sensor_ gyroscope.csv")
df_magnetometer = pd.read_csv("sensor_ magnetometer.csv")
df_accelerometer = pd.read_csv("sensor_ accelerometer.csv")

X = df_gyroscope.drop("Parameter", axis=1)
y = df_gyroscope["Parameter"]



scaler = MinMaxScaler()
X = scaler.fit_transform(X)


real_data = pd.DataFrame(X, columns=['a', 'b', 'c', 'd', 'e'])
real_labels = y


one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_labels = one_hot_encoder.fit_transform(np.array(real_labels).reshape(-1, 1))


NOISE_DIM = 100
NUM_CLASSES = 11
NUM_FEATURES = 5
BATCH_SIZE = 64
TRAINING_STEPS = 100


def create_generator():
    noise_input = Input(shape=(NOISE_DIM,))
    class_input = Input(shape=(NUM_CLASSES,))
    merged_input = Concatenate()([noise_input, class_input])
    hidden = Dense(128, activation='relu')(merged_input)
    output = Dense(NUM_FEATURES, activation='linear')(hidden)
    model = Model(inputs=[noise_input, class_input], outputs=output)
    return model


def create_discriminator():
    data_input = Input(shape=(NUM_FEATURES,))
    class_input = Input(shape=(NUM_CLASSES,))
    merged_input = Concatenate()([data_input, class_input])
    hidden = Dense(128, activation='relu')(merged_input)
    output = Dense(1, activation='relu')(hidden)
    model = Model(inputs=[data_input, class_input], outputs=output)
    return model




def create_cgan(generator, discriminator):
    noise_input = Input(shape=(NOISE_DIM,))
    class_input = Input(shape=(NUM_CLASSES,))
    generated_data = generator([noise_input, class_input])
    validity = discriminator([generated_data, class_input])
    model = Model(inputs=[noise_input, class_input], outputs=validity)
    return model


discriminator = create_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam())


generator = create_generator()


gan = create_cgan(generator, discriminator)


discriminator.trainable = False

gan.compile(loss='binary_crossentropy', optimizer=Adam())


for step in range(TRAINING_STEPS):
    idx = np.random.randint(0, real_data.shape[0], BATCH_SIZE)
    real_batch = real_data.iloc[idx].values
    labels_batch = one_hot_labels[idx]
    noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
    generated_batch = generator.predict([noise, labels_batch])
    real_loss = discriminator.train_on_batch([real_batch, labels_batch], np.ones((BATCH_SIZE, 1)))
    fake_loss = discriminator.train_on_batch([generated_batch, labels_batch], np.zeros((BATCH_SIZE, 1)))
    discriminator_loss = 0.5 * np.add(real_loss, fake_loss)
    generator_loss = gan.train_on_batch([noise, labels_batch], np.ones((BATCH_SIZE, 1)))

    if step % 500 == 0:
        print(f"Step: {step}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}")

def generate_data(generator, data_class, num_instances):
    one_hot_class = one_hot_encoder.transform(np.array([[data_class]]))
    noise = np.random.normal(0, 1, (num_instances, NOISE_DIM))
    generated_data = generator.predict([noise, np.repeat(one_hot_class, num_instances, axis=0)])
    synthetic_df = pd.DataFrame(generated_data, columns=['a', 'b', 'c', 'd', 'e'])
    synthetic_df['label'] = data_class
    return synthetic_df

generated_data = generate_data(generator, 0, 40)
print(generated_data)
