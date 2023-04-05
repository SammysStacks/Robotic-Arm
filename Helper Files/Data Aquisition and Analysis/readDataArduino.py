#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:44:26 2021

@author: samuelsolomon
"""

# General Modules
import re
import sys
import time
import threading
import matplotlib.pyplot as plt
# Stream Data from Arduino
import serial
import serial.tools.list_ports
# Import Bioelectric Analysis Files
from emgAnalysis import emgProtocol


# --------------------------------------------------------------------------- #
# ----------------- Stream Data from Arduino Can Edit ----------------------- #

class arduinoRead():
    def __init__(self, emgSerialNum = None, handSerialNum = None):
        # Save Arduino Serial Numbers
        self.emgSerialNum = emgSerialNum
        self.handSerialNum = handSerialNum

        # Connect to the Arduinos
        self.emgArduino = self.initiateArduino(self.emgSerialNum)
        self.handArduino = self.initiateArduino(self.handSerialNum)

    def printPortNums(self):
        ports = serial.tools.list_ports.comports()
        for port in ports:
            print(port.serial_number)

    def initiateArduino(self, arduinoSerialNum):
        arduinoControl = None
        if arduinoSerialNum:
            try:
                # Try to Connect to the Arduino
                arduinoPort = self.findArduino(serialNum = arduinoSerialNum)
                arduinoControl = serial.Serial(arduinoPort, baudrate=115200, timeout=1)

            except Exception as e:
                # If No Connection Established, Exit Program and Inform User
                print("Cannot Connect to Arudino", arduinoSerialNum);
                print("Error Message:", e)
                sys.exit()
        # Retun the Arduino actionControl
        return arduinoControl

    def resetArduino(self, arduino, arduinoSerialNum, numTrashReads):
        # Toss any data already received, see
        arduino.flushInput()
        
        # Read and throw out first few reads
        for i in range(numTrashReads):
            self.readAll(arduino)
            arduino.read_until()
        arduino.read_until()
        arduino.read_until()
        return arduino

    def findArduino(self, serialNum):
        """Get the name of the port that is connected to the Arduino."""
        port = None  # Initialize Blank Port
        # Get all Ports Connected to the Computer
        ports = serial.tools.list_ports.comports()
        # Loop Through Ports Until you Find the One you Want
        for p in ports:
            if p.serial_number == serialNum:
                port = p.device
        return port

    def handshakeArduino(self, arduino, sleep_time=1, print_handshake_message=False, handshake_code=0):
        """Make sure connection is established by sending
        and receiving bytes."""
        # Close and Reopen the Arduino
        arduino.close(); arduino.open()
        # Give the Arudino Some Time to Settle
        time.sleep(sleep_time)
        # Set a long timeout to complete handshake
        timeout = arduino.timeout
        arduino.timeout = 2

        # Read and discard everything that may be in the input buffer
        arduino.readAll()
        # Send request to Arduino
        arduino.write(bytes([handshake_code]))
        # Read in what Arduino sent
        handshake_message = arduino.read_until()
        # Send and receive request again
        arduino.write(bytes([handshake_code]))
        handshake_message = arduino.read_until()

        # Print the handshake message, if desired
        if print_handshake_message:
            print("Handshake message: " + handshake_message.decode())

        # Reset the timeout
        arduino.timeout = timeout


    def readAll(self, ser, readBuffer=b"", **args):
        """Read all available bytes from the serial port
        and append to the read buffer.

        Parameters
        ----------
        ser : serial.Serial() instance
            The device we are reading from.
        readBuffer : bytes, default b''
            Previous read buffer that is appended to.

        Returns
        -------
        output : bytes
            Bytes object that contains readBuffer + read.

        Notes
        -----
        .. `**args` appears, but is never used. This is for
           compatibility with `readAllNewlines()` as a
           drop-in replacement for this function.
        """
        # Set timeout to None to make sure we read all bytes
        previous_timeout = ser.timeout
        ser.timeout = None

        in_waiting = ser.in_waiting
        read = ser.read(size=in_waiting)

        # Reset to previous timeout
        ser.timeout = previous_timeout

        return readBuffer + read


    def readAllNewlines(self, ser, readBuffer=b"", n_reads=400):
        """Read data in until encountering newlines.

        Parameters
        ----------
        ser : serial.Serial() instance
            The device we are reading from.
        n_reads : int
            The number of reads up to newlines
        readBuffer : bytes, default b''
            Previous read buffer that is appended to.

        Returns
        -------
        output : bytes
            Bytes object that contains readBuffer + read.

        Notes
        -----
        .. This is a drop-in replacement for readAll().
        """
        raw = readBuffer
        for _ in range(n_reads):
            raw += ser.read_until()
        return raw


    def parseRead(self, read, numChannels):
        """Parse a read with time, volage data

        Parameters
        ----------
        read : byte string
            Byte string with comma delimited time/voltage
            measurements.

        Returns a List of:
        -------
        voltage : list of floats; Voltages in volts.
        time: x-axis data
        remaining_bytes : byte string remaining, unparsed bytes.
        """
        # Initiate Variables to Hold [[Voltages (Y) -> ...], Time (X), Buffer]
        arduinoData = [ [[] for channel in range(numChannels)], [] ]
        # Keep Track of Buffer
        if len(read) == 0:
            arduinoData.append(b"")
        else:
            arduinoData.append(b'')  # Last Element Should = raw_list[-1].encode()

        # Separate Time and Voltage Measurements
        pattern = re.compile(b"\d+|,|-")
        raw_list = [b"".join(pattern.findall(raw)).decode() for raw in read.split(b"\r\n")]

        for raw in raw_list[:-1]:
            try:
                # Seperate the Arduino Data
                arduinoValues = raw.split(",")

                if len(arduinoValues) == numChannels + 1:
                    # Store the Time and Voltage Data
                    arduinoData[1].append(int(arduinoValues[0])/1000)
                    for channelIndex in range(numChannels):
                        # Convert Arduino Data to Voltage Before Storing
                        arduinoData[0][channelIndex].append(int(arduinoValues[channelIndex+1]) * 5/1023)
                else:
                    print("Bad Arduino Reading:", arduinoValues)
                    print("You May Want to Inrease 'moveDataFinger' to Not Fall Behind in Reading Points")
            except:
                print("Cannot Read Arduino Value:", raw)
                pass
        # Return the Values
        return arduinoData

class emgArduinoRead(emgProtocol):

    def __init__(self, arduinoRead, numTimePoints, moveDataFinger, numChannels, samplingFreq, gestureClasses, plotStreamedData, guiApp = None):
        # Get Variables from Peak Analysis File
        super().__init__(numTimePoints, moveDataFinger, numChannels, samplingFreq, gestureClasses, plotStreamedData)

        # Store the arduinoRead Instance
        self.arduinoRead = arduinoRead
        self.emgArduino = arduinoRead.emgArduino
        self.handArduino = arduinoRead.handArduino

        # Variables for Hand Arduino's DistanceRead Funcion
        self.speed_x = 1 # speed_x = 1 when the arm is in fast mode, otherwise, speed_x = 0
        self.stop_x = 0  # stop_x = 1 when robot hand is stopped by the pressure sensor
        self.distance_slow = 120 # robot arm will slow down if the distance is less than this number,
        self.speed_slow = 0.05 # speed of arm in slow mode
        self.speed_fast = 0.15 # speed of arm in fast mode
        self.STOP = 9999 # when hand touch something, int(9999) will be sent to computer
        self.MOVE = 8888 # when hand does not touch anything, int(8888) will be sent to computer
        self.killDistanceRead = False

        # Initiate the GUI: a Copy of the UI Window
        self.guiApp = guiApp
        if self.guiApp:
            self.guiApp.handArduino = self.handArduino
            self.guiApp.initiateRoboticMovement()
            
    def controlRobotManually(self, numPointsRead, controlTimeSeconds, actionControl=None, numPointsPerRead=400):
        # Set Up Hand Arduino if Needed
        if self.handArduino:
            # Set Up Laser Reading
            threading.Thread(target = self.distanceRead, args = (actionControl, numPointsRead), daemon=True).start()
            
        # Wait while the user is controlling the device
        time.sleep(controlTimeSeconds)
        
        # Close and reset everything back to the starting point.
        if self.handArduino:
            self.handArduino.write(str.encode('s0')) # turn off the laser
            self.handArduino.close()
        if self.guiApp:
            self.guiApp.handArduino = None
            self.guiApp.resetButton()        

    def streamEMGData(self, numPointsRead, predictionModel = None, actionControl=None, numTrashReads=500, numPointsPerRead=400):
        """Obtain `numPointsRead` data points from an Arduino stream"""
        print("Streaming in EMG Data from the Arduino")
        
        # Read and throw out first few reads
        self.emgArduino.read_until(b'')
        for i in range(numTrashReads):
            self.emgArduino.read_until()
        
        # Set Up Hand Arduino if Needed
        if self.handArduino:
            # Set Up Laser Reading
            threading.Thread(target = self.distanceRead, args = (actionControl, numPointsRead), daemon=True).start()
            
        try:
            readBuffer = b""; dataFinger = 0
            # Loop Through and Read the Arduino Data in Real-Time
            while len(self.data["timePoints"]) < numPointsRead:

                # Read in chunk of data
                raw = self.arduinoRead.readAllNewlines(ser=self.emgArduino, readBuffer=readBuffer, n_reads=numPointsPerRead)
                # Parse it, passing if it is gibberish
                Voltages, timePoints, readBuffer = self.arduinoRead.parseRead(raw, self.numChannels)

                # Update data dictionary
                self.data["timePoints"].extend(timePoints)
                for channelIndex in range(self.numChannels):
                    self.data['Channel' + str(channelIndex+1)].extend(Voltages[channelIndex])

                # When Ready, Send Data Off for Analysis
                pointNum = len(self.data["timePoints"])
                while pointNum - dataFinger >= self.numTimePoints:
                    self.analyzeData(dataFinger, self.plotStreamedData, predictionModel, actionControl)
                    dataFinger += self.moveDataFinger
            # At the End, Analyze Any Data Left
            if dataFinger < len(self.data["timePoints"]):
                self.analyzeData(dataFinger, self.plotStreamedData, predictionModel, actionControl)

        finally:
            self.emgArduino.close()

        print("Finished Streaming in Data; Closing Arduino\n")
        # Close the Arduinos at the End
        self.emgArduino.close()
        if self.handArduino:
            self.handArduino.write(str.encode('s0')) # turn off the laser
            self.handArduino.close()
        if self.guiApp:
            self.guiApp.handArduino = None
            self.guiApp.resetButton()

    def distanceRead(self, RoboArm, numPointsRead):
        print("In Distance Read")
        for _ in range(5):
            self.handArduino.read_until()
        l_time = 0
        while len(self.data["timePoints"]) < numPointsRead and not self.killDistanceRead:
            if self.handArduino.in_waiting > 0:
                d_laser = self.handArduino.read_until()
                distance = d_laser.decode()

                # Update Gui App Text
                if self.guiApp:
                    self.guiApp.Number_distance.setText(self.guiApp.translate("MainWindow", str(distance)))
                distance = int(distance)
                l_time = l_time + 100
                if distance < self.distance_slow and self.speed_x == 1:
                    self.handArduino.read_until()
                    RoboArm.updateMovementParams([self.speed_slow]*5, 'speed')
                    self.handArduino.read_until()
                    self.speed_x = 0
                    print('slow')
                elif distance >= self.distance_slow and self.speed_x == 0 and distance <= 2000:
                    self.handArduino.read_until()
                    RoboArm.updateMovementParams([self.speed_fast]*5, 'speed')
                    self.handArduino.read_until()
                    self.speed_x = 1
                    print('fast')
                elif distance == self.STOP and self.stop_x == 0:
                    print('stop!!')
                    RoboArm.stopRobot()
                    self.stop_x = 1
                elif distance == self.MOVE:
                    self.stop_x =0
            time.sleep(0.05)

