void setup() {
   Serial.begin(115200);         //Use 115200 baud rate for serial communication
}

void loop() {

  int Channel1 = analogRead(A0);    //Read the voltage value of A0 port (Channel1)
  int Channel2 = analogRead(A1);    //Read the voltage value of A1 port (Channel2)
  int Channel3 = analogRead(A2);    //Read the voltage value of A2 port (Channel3)
  int Channel4 = analogRead(A3);    //Read the voltage value of A3 port (Channel4)
  
  Serial.print(Channel1);           //Output Channel1 data
  Serial.print(Channel2);           //Output Channel2 data
  Serial.print(Channel3);           //Output Channel3 data
  Serial.println(Channel4);         //Output Channel4 data and newline character, which represent the end of a group of data
}
