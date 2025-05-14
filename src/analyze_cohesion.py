
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Find all cohesion log files

# log_files = glob.glob("C:\\Users\\aryan\\Downloads\\logs\\miro*_cohesion_log.csv")
# /tmp by default csv are created in temp
log_files = glob.glob("/tmp/miro*_cohesion_log.csv")


# Initialize the plot
plt.figure(figsize=(12, 6))

# Load and plot each file
for file in log_files:
    df = pd.read_csv(file)
    label = df['agent'][0] if not df.empty else file
    plt.plot(df['timestamp'], df['avg_pairwise_dist'], label=label)

# Format plot
plt.xlabel('Time (s)')
plt.ylabel('Avg Pairwise Distance (m)')
plt.title('Cohesion Over Time - All MiRo Agents')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()
