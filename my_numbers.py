# my_numbers.py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def train_and_plot_graph():

    try:
        # Generate some data
        x_data = np.linspace(-5, 5, 100)
        y_data = 2 * x_data + 1 + np.random.normal(0, 1, size=len(x_data))

        # Define input tensor using tf.keras.Input
        x = tf.keras.Input(shape=(1,), dtype=tf.float32, name='x')

        # Define variables for slope and intercept
        slope = tf.Variable(1.0, dtype=tf.float32, name='slope')
        intercept = tf.Variable(0.0, dtype=tf.float32, name='intercept')

        # Define the linear model using functional API
        y_pred = slope * x + intercept

        # Define the loss function (mean squared error)
        loss = tf.reduce_mean(tf.square(y_data - y_pred), name='loss')

        # Define the optimizer
        optimizer = tf.optimizers.SGD(learning_rate=0.01)

    except Exception as e:
        print(f"An error occurred: {e}")


    # Training step using tf.GradientTape
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = slope * x + intercept
            loss = tf.reduce_mean(tf.square(y - y_pred), name='loss')

        gradients = tape.gradient(loss, [slope, intercept])
        optimizer.apply_gradients(zip(gradients, [slope, intercept]))

    # Train the model
    for _ in range(100):
        train_step(x_data, y_data)

    # Get the final slope and intercept values
    final_slope, final_intercept = slope.numpy(), intercept.numpy()

    # Plot the data and the linear regression line
    plt.scatter(x_data, y_data, label='Data points')
    plt.plot(x_data, final_slope * x_data + final_intercept, color='red', label='Linear Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_and_plot_graph()
