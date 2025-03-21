# This file reads the analog output of multiple
# MyoWare sensor. It connects to the serial port,
# reads in the data and then saves it in a .txt
# file. From here the data can be manipulated in
# any way.
#
# DISCLAIMER: 
# Be sure to close the Serial Plotter and Serial
# Monitor when running the script.
#
# Apparently there is a bug in the new
# version of the Arduino IDE. If the Python script
# says it cannot connect to the port because it is
# taken, close the Serial Plotter/Monitor and/or
# close and restart the IDE.

import serial
import sys

SERIAL_PORT = "/dev/cu.usbserial-110"
BAUD_RATE = 115200 
FILENAME = sys.argv[1]

print(f"Starting Arduino Multi-Sensor Logger\nFilename: {FILENAME}")
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
print(f"Connected to Arduino port: {SERIAL_PORT}")
file = open(FILENAME, "w")
print("Opened file")
print(f"Reading sensor in port {SERIAL_PORT}")

try:
  while True:
    getData = ser.readline()
    dataString = getData.decode('utf-8')
    print(dataString[:-1])
    file.write(dataString[:-1])
except KeyboardInterrupt:
  print("Closing logger...")
  file.close()
  print("Logger closed")