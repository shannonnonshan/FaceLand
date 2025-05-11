import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv("fps_cpu_data.csv", header=None, names=["timestamp", "fps", "cpu", "ram"])

# Chuyển cột timestamp thành định dạng datetime
data["timestamp"] = pd.to_datetime(data["timestamp"])

# Vẽ biểu đồ cho FPS, CPU và RAM
plt.figure(figsize=(10, 6))

# Vẽ FPS
plt.subplot(3, 1, 1)
plt.plot(data["timestamp"], data["fps"], label="FPS", color='green')
plt.title("FPS Over Time")
plt.xlabel("Time")
plt.ylabel("FPS")
plt.xticks(rotation=45)

# Vẽ CPU usage
plt.subplot(3, 1, 2)
plt.plot(data["timestamp"], data["cpu"], label="CPU Usage", color='orange')
plt.title("CPU Usage Over Time")
plt.xlabel("Time")
plt.ylabel("CPU (%)")
plt.xticks(rotation=45)

# Vẽ RAM usage
plt.subplot(3, 1, 3)
plt.plot(data["timestamp"], data["ram"], label="RAM Usage", color='red')
plt.title("RAM Usage Over Time")
plt.xlabel("Time")
plt.ylabel("RAM (%)")
plt.xticks(rotation=45)

# Cải thiện hiển thị
plt.tight_layout()
plt.show()
