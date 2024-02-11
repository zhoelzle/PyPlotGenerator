# main.py
import os

import matplotlib.pyplot as plt

from my_numbers import train_and_plot_graph


if __name__ == "__main__":
    train_and_plot_graph()


def generate_unique_filename(base_filename):
    # Check if the file exists
    if os.path.exists(base_filename):
        # Split the filename and extension
        name, extension = os.path.splitext(base_filename)

        # Try incrementing the number until finding a unique filename
        count = 1
        while os.path.exists(f"{name}{count}{extension}"):
            count += 1

        # Return the new unique filename
        return f"{name}{count}{extension}"
    else:
        # If the file doesn't exist, return the original filename
        return base_filename

# Example usage
filename = 'plot.png'
unique_filename = generate_unique_filename(filename)
print(f"Original Filename: {filename}")
print(f"Unique Filename: {unique_filename}")

# Save the plot to an image file within the notebook
plt.savefig(unique_filename)