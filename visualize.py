import matplotlib.pyplot as plt
import pandas as pd

# Load the data
df = pd.read_csv("mouse_data.csv")

# Plot the mouse trajectory
plt.figure(figsize=(8, 6))
plt.plot(df['x'], df['y'], marker='o', markersize=2, linestyle='-')
plt.title("Mouse Trajectory")
plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (pixels)")
plt.gca().invert_yaxis()  # Invert Y-axis for screen coordinates
plt.show()
