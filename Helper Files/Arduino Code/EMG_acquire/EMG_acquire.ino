// Global Time Variables
unsigned long  totalTime; unsigned long startTime;
// Global Channel Variables
int Channel1; int Channel2;
int Channel3; int Channel4;

void setup() {
   Serial.begin(115200);         //Use 115200 baud rate for serial communication
   startTime = millis();
}

void loop() {
  // Read Channel Information
  Channel1 = analogRead(A0);    //Read the voltage value of A0 port (Channel1)
  Channel2 = analogRead(A1);    //Read the voltage value of A1 port (Channel2)
  Channel3 = analogRead(A2);    //Read the voltage value of A2 port (Channel3)
  Channel4 = analogRead(A3);    //Read the voltage value of A3 port (Channel4)
  // Get Current Time
  totalTime = millis() - startTime;
  
  // Print Out Information
  Serial.println(String(totalTime) + ',' + String(Channel1) + ',' + String(Channel2) + ',' + String(Channel3) + ',' + String(Channel4));
}
