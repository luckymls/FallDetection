import serial
import serial.tools.list_ports
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def connect_to_esp32(baudrate=115200):
    target_descriptions = ["USB-SERIAL CH340", "CP210x"]  

    while True:
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            if any(desc in port.description for desc in target_descriptions):
                try:
                    ser = serial.Serial(port.device, baudrate, timeout=1)
                    print(f"Connected to {port.device} - {port.description}")
                    return ser
                except serial.SerialException:
                    print(f"Failed to connect to {port.device} - {port.description}")
        print("ESP32-C3 not found. Retrying in 5 seconds...")
        time.sleep(5)

ser = connect_to_esp32()

ax_vals = []
ay_vals = []
az_vals = []
gx_vals = []
gy_vals = []
gz_vals = []

def update_plot(frame):
    line = ser.readline().decode('utf-8').strip()
    try:
        ax, ay, az, gx, gy, gz = map(int, line.split(','))

        ax_vals.append(ax)
        ay_vals.append(ay)
        az_vals.append(az)
        gx_vals.append(gx)
        gy_vals.append(gy)
        gz_vals.append(gz)

        # Keep the lists to a fixed size for better performance
        max_length = 100
        if len(ax_vals) > max_length:
            ax_vals.pop(0)
            ay_vals.pop(0)
            az_vals.pop(0)
            gx_vals.pop(0)
            gy_vals.pop(0)
            gz_vals.pop(0)

        ax_plot.clear()
        ay_plot.clear()
        az_plot.clear()
        gx_plot.clear()
        gy_plot.clear()
        gz_plot.clear()

        ax_plot.plot(ax_vals, label='AX')
        ay_plot.plot(ay_vals, label='AY')
        az_plot.plot(az_vals, label='AZ')

        gx_plot.plot(gx_vals, label='GX')
        gy_plot.plot(gy_vals, label='GY')
        gz_plot.plot(gz_vals, label='GZ')

        ax_plot.legend(loc='upper left')
        ay_plot.legend(loc='upper left')
        az_plot.legend(loc='upper left')
        gx_plot.legend(loc='upper left')
        gy_plot.legend(loc='upper left')
        gz_plot.legend(loc='upper left')

    except ValueError:
        pass


fig, ((ax_plot, ay_plot, az_plot), (gx_plot, gy_plot, gz_plot)) = plt.subplots(2, 3)

ani = animation.FuncAnimation(fig, update_plot, interval=10)

plt.show()
