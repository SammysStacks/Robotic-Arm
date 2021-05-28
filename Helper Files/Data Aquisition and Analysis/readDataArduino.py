#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:44:26 2021

@author: samuelsolomon
"""

# Input Data from Arduino
import serial
import serial.tools.list_ports
# General Modules
import time
import sys
import re
import threading

from peakAnalysis import globalParam


# --------------------------------------------------------------------------- #
# ----------------- Stream Data from Arduino Can Edit ----------------------- #

class arduinoRead(globalParam):
    def __init__(self, emgSerialNum, handSerialNum, xWidth = 2000, moveDataFinger = 200, numChannels = 4, movementOptions = [], guiApp = None):
        # Get Variables from Peak Analysis File
        super().__init__(xWidth, moveDataFinger, numChannels, movementOptions)

        # Set up Connection to Arduino
        self.HANDSHAKE = 0
        self.VOLTAGE_REQUEST = 1
        self.ON_REQUEST = 2;
        self.STREAM = 3;
        self.READ_DAQ_DELAY = 4;
        
        # Copy of the UI Window
        self.guiApp = guiApp
        
        # Variables for Hand Arduino's DistanceRead Funcion
        self.speed_x = 1 # speed_x = 1 when the arm is in fast mode, otherwise, speed_x = 0
        self.stop_x = 0  # stop_x = 1 when robot hand is stopped by the pressure sensor
        self.distance_slow = 120 # robot arm will slow down if the distance is less than this number,
        self.speed_slow = 0.05 # speed of arm in slow mode
        self.speed_fast = 0.15 # speed of arm in fast mode
        self.STOP = 9999 # when hand touch something, int(9999) will be sent to computer
        self.MOVE = 8888 # when hand does not touch anything, int(8888) will be sent to computer
     
        # Initiate Arduinos
        self.emgArduino, self.handArduino = self.initiateArduino(emgSerialNum, handSerialNum)
            
        self.killDistanceRead = False
        
    def initiateArduino(self, emgSerialNum, handSerialNum):
        if handSerialNum:
            # Initiate Hand Arduino
            try:
                handPort = self.find_arduino(serialNum = handSerialNum)
                handArduino = serial.Serial(handPort, baudrate=115200, timeout=1)
                if self.guiApp:
                    self.guiApp.handArduino = handArduino
                    self.guiApp.initiateRoboticMovement()
            except Exception as e:
                    print("No Hand Port Found: ", e)
                    sys.exit()
        else:
            handArduino = None
        
        # Find EMG Arduino Port. If None: print Error
        try:
            port = self.find_arduino(serialNum = emgSerialNum)
            emgArduino = serial.Serial(port, baudrate=115200)
        except Exception as e:
                print("No EMG Arduino Port Found: ", e)
                sys.exit() 
        
        return emgArduino, handArduino
    
    def find_arduino(self, serialNum = 'NA'):
        """Get the name of the port that is connected to Arduino."""
        # Get all Ports Connected to the Computer
        ports = serial.tools.list_ports.comports()
        # Loop Through Ports Until you Find the One you Want
        port = None  # Initialize Blank Port
        for p in ports:
            if p.serial_number == serialNum:
                port = p.device
        return port
    
    
    def handshake_arduino(self, arduino, sleep_time=1, print_handshake_message=False, handshake_code=0):
        """Make sure connection is established by sending
        and receiving bytes."""
        # Close and reopen
        arduino.close()
        arduino.open()
    
        # Chill out while everything gets set
        time.sleep(sleep_time)
    
        # Set a long timeout to complete handshake
        timeout = arduino.timeout
        arduino.timeout = 2
    
        # Read and discard everything that may be in the input buffer
        _ = arduino.read_all()
    
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
    
    
    def read_all(self, ser, read_buffer=b"", **args):
        """Read all available bytes from the serial port
        and append to the read buffer.
    
        Parameters
        ----------
        ser : serial.Serial() instance
            The device we are reading from.
        read_buffer : bytes, default b''
            Previous read buffer that is appended to.
    
        Returns
        -------
        output : bytes
            Bytes object that contains read_buffer + read.
    
        Notes
        -----
        .. `**args` appears, but is never used. This is for
           compatibility with `read_all_newlines()` as a
           drop-in replacement for this function.
        """
        # Set timeout to None to make sure we read all bytes
        previous_timeout = ser.timeout
        ser.timeout = None
    
        in_waiting = ser.in_waiting
        read = ser.read(size=in_waiting)
    
        # Reset to previous timeout
        ser.timeout = previous_timeout
    
        return read_buffer + read
    
    
    def read_all_newlines(self, ser, read_buffer=b"", n_reads=4):
        """Read data in until encountering newlines.
    
        Parameters
        ----------
        ser : serial.Serial() instance
            The device we are reading from.
        n_reads : int
            The number of reads up to newlines
        read_buffer : bytes, default b''
            Previous read buffer that is appended to.
    
        Returns
        -------
        output : bytes
            Bytes object that contains read_buffer + read.
    
        Notes
        -----
        .. This is a drop-in replacement for read_all().
        """
        raw = read_buffer
        for _ in range(n_reads):
            raw += ser.read_until()
        return raw
    
    
    def parse_read(self, read):
        """Parse a read with time, volage data
    
        Parameters
        ----------
        read : byte string
            Byte string with comma delimited time/voltage
            measurements.
    
        Returns
        -------
        voltage : list of floats; Voltages in volts.
        remaining_bytes : byte string remaining, unparsed bytes.
        """
        # Initiate Variables to Hold Voltages
        channelList = {0:[], 1:[], 2:[], 3:[]}
        time_ms = []
    
        # Separate independent time/voltage measurements
        pattern = re.compile(b"\d+|,|-")
        raw_list = [
            b"".join(pattern.findall(raw)).decode()
            for raw in read.split(b"\r\n")
        ]
    
        for raw in raw_list[:-1]:
            try:
                t, V1, V2, V3, v4 = raw.split(",")
                
                time_ms.append(int(t))
                for i,V in enumerate([V1, V2, V3, v4]):
                    channelList[i].append(int(V) * 5/1023)
            except:
                pass
        
        # Return the Voltages You Read in
        if len(read) == 0:
            return time_ms, channelList[0], channelList[1], channelList[2], channelList[3], b""
        else:
            return time_ms, channelList[0], channelList[1], channelList[2], channelList[3], b'' #raw_list[-1].encode()
    
    def streamArduinoData(self, n_data, seeFullPlot, myModel = None, Controller=None, n_trash_reads=500, n_reads_per_chunk=400, delay=100):
        """Obtain `n_data` data points from an Arduino stream
        with a delay of `delay` milliseconds between each."""
        print("Streaming in Data from the Arduino")
        
        # Specify delay
        self.emgArduino.write(bytes([self.READ_DAQ_DELAY]) + (str(delay) + "x").encode())
        
        # Turn on the stream
        self.emgArduino.write(bytes([self.STREAM]))
    
        # Read and throw out first few reads
        for i in range(n_trash_reads):
            _ = self.emgArduino.read_until()
        _ = self.handArduino.read_all()
        
        # Set Up Laser Reading
        threading.Thread(target = self.distanceRead, args = (Controller, n_data), daemon=True).start()
        
        # Receive data
        read_buffer = b""
        dataFinger = 0
        while len(self.data["time_ms"]) < n_data:
            # Read in chunk of data
            raw = self.read_all_newlines(self.emgArduino, read_buffer=read_buffer, n_reads=n_reads_per_chunk)
            
            # Parse it, passing if it is gibberish
            try:
                time_ms, V1, V2, V3, V4, read_buffer = self.parse_read(raw)
                
                # Update data dictionary
                startNum = len(self.data["time_ms"])
                for i in range(len(time_ms)):
                    self.data["time_ms"].append(startNum+i)
                self.data["Channel1"] += V1
                self.data["Channel2"] += V2
                self.data["Channel3"] += V3
                self.data["Channel4"] += V4
                            
                # When Ready, Send Data Off for Analysis
                pointNum = len(self.data["time_ms"])
                while pointNum - dataFinger >= self.xWidth:
                    self.live_plotter(dataFinger, seeFullPlot, myModel = myModel, Controller = Controller)
                    dataFinger += self.moveDataFinger
            except Exception as e:
                print(e)
                pass
            
        # Close the Arduino at the End
        print("Finished Streaming in Data; Closing Arduino\n")
        self.emgArduino.close()
        self.handArduino.write(str.encode('s0')) # turn off the laser
        self.handArduino.close()
        if self.guiApp:
            self.guiApp.handArduino = None
            self.guiApp.resetButton()
        
    
    def distanceRead(self, RoboArm, n_data):
        print("In Distance Read")
        for _ in range(5):
            self.handArduino.read_until()
        l_time = 0
        while len(self.data["time_ms"]) < n_data and not self.killDistanceRead:
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
         
 
    
 