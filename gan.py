from tabgan.sampler import GANGenerator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load your data
df_gyroscope = pd.read_csv("sensor_gyroscope.csv")
df_accelerometer = pd.read_csv("sensor_accelerometer.csv")

# Example of data preprocessing
X = df_gyroscope.drop("Parameter", axis=1)
y = df_gyroscope["Parameter"]

# Train-test split
df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Create dataframe versions for tabular GAN
df_x_test, df_y_test = df_x_test.reset_index(drop=True), df_y_test.reset_index(drop=True)
df_y_train = pd.DataFrame(df_y_train)
df_y_test = pd.DataFrame(df_y_test)

# Generate augmented data
gen_x, gen_y = GANGenerator(gen_x_times=1.1, cat_cols=None,
           bot_filter_quantile=0.001, top_filter_quantile=0.999, \
           is_post_process=True,
           adversarial_model_params={
               "metrics": "rmse", "max_depth": 1, "max_bin": 10,
               "learning_rate": 0.02, "random_state": 42, "n_estimators": 500,
           }, pregeneration_frac=2, only_generated_data=False).generate_data_pipe(df_x_train, df_y_train, df_x_test, deep_copy=True, only_adversarial=False)
