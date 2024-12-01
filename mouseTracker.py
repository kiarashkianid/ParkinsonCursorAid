import pyautogui
import time
import csv

# File to store mouse movement data
output_file = "mouse_data.csv"

# Create the CSV file and add a header
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "x", "y"])

# Collect mouse positions every 0.1 seconds
try:
    while True:
        x, y = pyautogui.position()
        timestamp = time.time()
        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, x, y])
        time.sleep(0.1)  # Adjust the time for your desired sampling rate
except KeyboardInterrupt:
    print("Data collection stopped.")
