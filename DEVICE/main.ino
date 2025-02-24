#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <SPIFFS.h>


#define MPU6050_CONFIG_DATA_PATH "/mpu6050_config.txt"

#define SERVICE_UUID "12345678-1234-5678-1234-56789abcde01"
#define DATA_CHARACTERISTIC_UUID "12345678-1234-5678-1234-56789abcde02"
#define ALARM_CHARACTERISTIC_UUID "12345678-1234-5678-1234-56789abcde03"

#define BUFFER_SIZE 300

int pinVib = 0;
int pinLed_B = 3;
int pinLed_R = 2;
int pinLed_G = 1;
int pinMPU_INT = 5;
int pinMPU_SDA = 6;
int pinMPU_SCL = 7;
int pinButton = 10;

volatile bool buttonPressed = false;

// PARAMETRI DA MODIFICARE
int PRE_TRIGGER_TIME = 0.5;
int POST_TRIGGER_TIME = 1;
float THRESHOLD_ACC = 1.66;
int SAMPLE_RATE = 200;


int PRE_TRIGGER = PRE_TRIGGER_TIME * SAMPLE_RATE;
int POST_TRIGGER = POST_TRIGGER_TIME * SAMPLE_RATE;
Adafruit_MPU6050 mpu;


typedef struct {
  float acc_x;
  float acc_y;
  float acc_z;
  float gyro_x;
  float gyro_y;
  float gyro_z;
  uint32_t relativeTimeMs;
} SensorData;


float offset_acc = 0.0;
float offset_gyro = 0.0;


enum Events { DEVICE_ON,
              CAL_START,
              CAL_END,
              BLE_CONN,
              FALL_DETECTED,
              MAYBE_FALL,
              ERROR = -1 };



SensorData buffer[BUFFER_SIZE];  
long long int bufferIndex = 0;

uint32_t startTime = 0;

// --- BLE ---

BLEServer *pServer = nullptr;
BLECharacteristic *pDataCharacteristic = nullptr;
BLECharacteristic *pAlarmCharacteristic = nullptr;

bool deviceConnected = false;

// --- POST TRIGGER ---

bool fallDetected = false;
uint16_t postTriggerCount = 0;

  void IRAM_ATTR handleButtonPress() {
    buttonPressed = true;
  }

void setRGBColor(int r, int g, int b) {
  analogWrite(pinLed_R, r);
  analogWrite(pinLed_G, g);
  analogWrite(pinLed_B, b);
}

void blinkRGB(int r, int g, int b, int duration, int count) {
  for (int i = 0; i < count; i++) {
    setRGBColor(r, g, b);  // Accendi il LED
    delay(duration);
    setRGBColor(0, 0, 0);  // Spegni il LED
    delay(duration);
  }
}

void vibrate(int duration, int count) {
  for (int i = 0; i < count; i++) {
    digitalWrite(pinVib, HIGH);  // Accendi il motore
    delay(duration);
    digitalWrite(pinVib, LOW);  // Spegni il motore
    delay(duration);
  }
}

void notify(Events event) {
  switch (event) {
    case DEVICE_ON:  // Bianco per 1 secondo + 1 vibrazione
      setRGBColor(255, 255, 255);
      vibrate(200, 1);
      delay(1500);
      setRGBColor(0, 0, 0);
      break;

    case CAL_START:  // Giallo per 1.5 secondi più vibrazione
      setRGBColor(255, 255, 0);
      vibrate(150, 2);
      delay(1500);
      setRGBColor(0, 0, 0);
      break;

    case CAL_END:
      // Blue lampeggio 2 volte più due vibrazioni
      blinkRGB(0, 0, 255, 300, 2);
      vibrate(200, 2);
      break;

    case BLE_CONN:
      // Ciano per 1 secondo + una vibrazione breve
      setRGBColor(0, 255, 255);
      vibrate(200, 1);
      delay(1000);
      setRGBColor(0, 0, 0);
      break;

    case FALL_DETECTED:
      // Rosso veloce lampeggiante per 10 + vibrazione lunga
      vibrate(300, 2);
      blinkRGB(255, 0, 0, 200, 10);
      vibrate(300, 2);
      break;
    
    case MAYBE_FALL:
      // Rosso veloce lampeggiante per 10 + vibrazione lunga
      vibrate(250, 1);
      blinkRGB(255, 0, 0, 200, 3);
      vibrate(300, 1);
      break;

    case ERROR:
      vibrate(200, 3);
      // Rosso e blu che si alternano + 3 vibrazioni brevi
      for (int i = 0; i < 3; i++) {
        setRGBColor(255, 0, 0);  // Rosso
        delay(300);
        setRGBColor(0, 0, 255);  // Blu
        delay(300);
      }

      setRGBColor(0, 0, 0);
      break;

    default:
      
      setRGBColor(0, 0, 0);
      digitalWrite(pinVib, LOW);
      break;
  }
}


class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *pServer) {
    deviceConnected = true;
    notify(BLE_CONN);
  }

  void onDisconnect(BLEServer *pServer) {
    deviceConnected = false;

    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();  // Restart advertising per nuove connessioni
    pAdvertising->start();
    Serial.println("Bluetooth disconnected.");
    Serial.println("Advertising restarted");
  }
};


void BLE_initialize() {
  BLEDevice::init("Guardian-Angel-BLE");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());
  BLEService *pService = pServer->createService(SERVICE_UUID);

  // Se vuoi usare pDataCharacteristic per notifiche, aggiungi PROPERTY_NOTIFY
  pDataCharacteristic = pService->createCharacteristic(
                          DATA_CHARACTERISTIC_UUID,
                          BLECharacteristic::PROPERTY_READ |
                          BLECharacteristic::PROPERTY_WRITE |
                          BLECharacteristic::PROPERTY_NOTIFY
                        );
  // Caratteristica per notifiche di allarme
  pAlarmCharacteristic = pService->createCharacteristic(
    ALARM_CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_NOTIFY);
  pService->start();
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->start();
  Serial.println("BLE initialized. Ready for connections.");
}


float computeAverageDataTimeAcquisition() {
  int timeSum = 0;
  int timeCount = 0;


  uint32_t lastTimeStamp = buffer[bufferIndex % BUFFER_SIZE].relativeTimeMs;  // Most recent timestamp

  for (int i = 1; i < BUFFER_SIZE; i++) {
    int index = (bufferIndex + i) % BUFFER_SIZE;


    uint32_t currentTimeStamp = buffer[index].relativeTimeMs;
    int timeDifference = currentTimeStamp - lastTimeStamp;


    timeSum += timeDifference;
    timeCount++;

    lastTimeStamp = currentTimeStamp;
  }

  if (timeCount == 0) {
    return 0.0f;
  }

  return static_cast<float>(timeSum) / timeCount;
}


void sendDataOverBLE(int alarm=0) {
  if (!deviceConnected) {
    Serial.println("Device not connected.");
    while (!deviceConnected) {
      delay(30);
    }
    Serial.println("BLE connection established.");
  }

  if(alarm == 0){
    
    String jsonString = "{\"avgTime\":" + String(computeAverageDataTimeAcquisition()) + ",\"data\":[";

    for (int i = 0; i < BUFFER_SIZE; i++) {
      int index = (bufferIndex + i) % BUFFER_SIZE;
      jsonString += "[" + String(buffer[index].acc_x) + "," + String(buffer[index].acc_y) + "," + String(buffer[index].acc_z) + ",";
      jsonString += String(buffer[index].gyro_x) + "," + String(buffer[index].gyro_y) + "," + String(buffer[index].gyro_z) + "]";
      if (i < BUFFER_SIZE - 1) jsonString += ","; 
    }

    jsonString += "]}"; // Close JSON


    int chunkSize = 20;  
    for (long int i = 0; i < jsonString.length(); i += chunkSize) {
      String chunk = jsonString.substring(i, min(i + chunkSize, (long)jsonString.length()));
      pDataCharacteristic->setValue(chunk);
      pDataCharacteristic->notify();
      delay(7);
    }
  }else{ // Invia allarme
      
      pDataCharacteristic->setValue("alarm");
      pDataCharacteristic->notify();
  }
}




float accelerationSVM(int x, int y, int z) {

  return sqrt(x * x + y * y + z * z);
}


void MPU_init() {

  mpu.setAccelerometerRange(MPU6050_RANGE_16_G);
  mpu.setGyroRange(MPU6050_RANGE_2000_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_10_HZ);
}


bool loadMPU6050Calibration(float &offset_acc, float &offset_gyro) {



  File file = SPIFFS.open(MPU6050_CONFIG_DATA_PATH, "r");
  if (!file) {
    return false;
  }

  String content = file.readString();
  file.close();

  if (content.length() > 0) {
    sscanf(content.c_str(), "%f %f", &offset_acc, &offset_gyro);
    Serial.println("Calibration data loaded.");
    return true;
  } else {
    return false;
  }
}

void saveMPU6050Calibration(float offset_acc, float offset_gyro) {


  File file = SPIFFS.open(MPU6050_CONFIG_DATA_PATH, "w");
  if (!file) {
    Serial.println("Calibration data not saved.");
    return;
  }

  file.printf("%f %f", offset_acc, offset_gyro);
  file.close();

  Serial.println("Calibration data saved successfully");
}

void calibrateMPU6050(int numSamples) {

  double accxSum = 0, accySum = 0, acczSum = 0;
  double gyroxSum = 0, gyroySum = 0, gyrozSum = 0;

  float offset_accx = 0.0;
  float offset_accy = 0.0;
  float offset_accz = 0.0;
  float offset_gyrox = 0.0;
  float offset_gyroy = 0.0;
  float offset_gyroz = 0.0;

  for (int i = 0; i < numSamples; i++) {

    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    accxSum += a.acceleration.x;
    accySum += a.acceleration.y;
    acczSum += a.acceleration.z;
    gyroxSum += g.gyro.x;
    gyroySum += g.gyro.y;
    gyrozSum += g.gyro.z;

    delay(1000 / SAMPLE_RATE);  // Sample frequency
  }

  // Calculate offsets

  offset_accx = accxSum / numSamples;
  offset_accy = accySum / numSamples;
  offset_accz = (acczSum / numSamples);  // Correct for gravity (zero at rest, gravità è  circa 16384 nell'asse z)
  offset_gyrox = gyroxSum / numSamples;
  offset_gyroy = gyroySum / numSamples;
  offset_gyroz = gyrozSum / numSamples;

  offset_acc = accelerationSVM(offset_accx, offset_accy, offset_accz);
  offset_gyro = accelerationSVM(offset_gyrox, offset_gyroy, offset_gyroz);
  

  Serial.println("Calibration completed.");
  Serial.print("Accel Offsets: ");
  Serial.print(offset_acc);
  Serial.print("Gyro Offsets: ");
  Serial.println(offset_gyro);
}

void MPU_READ(sensors_event_t &a_res, sensors_event_t &g_res) {

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);


  a_res.acceleration.x = a.acceleration.x;
  a_res.acceleration.y = a.acceleration.y;
  a_res.acceleration.z = a.acceleration.z;

  g_res.gyro.x = g.gyro.x;
  g_res.gyro.y = g.gyro.y;
  g_res.gyro.z = g.gyro.z;
  
}

void print_data(sensors_event_t &a_res, sensors_event_t &g_res, float accSVM, float gyroSVM) {

  Serial.print(a_res.acceleration.x);
  Serial.print(",");
  Serial.print(a_res.acceleration.y);
  Serial.print(",");
  Serial.print(a_res.acceleration.z);
  Serial.print(",");
  Serial.print(g_res.gyro.x);
  Serial.print(",");
  Serial.print(g_res.gyro.y);
  Serial.print(",");
  Serial.print(g_res.gyro.z);
  Serial.print(",");
  Serial.print(accSVM);
  Serial.print(",");
  Serial.print(gyroSVM);
  Serial.println("");
}


void setup() {

  bool isCalibrated = false;
  bool calibrateAnyway = true;  // DEBUG PURPOSE 

  Serial.begin(115200);
  Wire.begin(6, 7);

  pinMode(pinLed_R, OUTPUT);
  pinMode(pinLed_G, OUTPUT);
  pinMode(pinLed_B, OUTPUT);
  pinMode(pinVib, OUTPUT);
  pinMode(pinButton, INPUT_PULLUP);
  pinMode(pinMPU_INT, INPUT);
  attachInterrupt(digitalPinToInterrupt(pinButton), handleButtonPress, FALLING);


  notify(DEVICE_ON);

  if (!SPIFFS.begin(true)) {
    Serial.println("Failed to mount SPIFFS. Skipping calibration loading.");
  }

  if (!mpu.begin()) {
    Serial.println("MPU6050 connection failed");
    notify(ERROR);
    while (!mpu.begin()) {
      delay(30);
    }
  }

  MPU_init();

  isCalibrated = loadMPU6050Calibration(offset_acc, offset_gyro);
  if (!isCalibrated || calibrateAnyway) {
    Serial.println("Sensor calibration. Make sure it's on a flat surface. Starting in 5 seconds...");
    notify(CAL_START);
    delay(5000);
    calibrateMPU6050(300);
    saveMPU6050Calibration(offset_acc, offset_gyro);
    notify(CAL_END);
  } else {
    Serial.println("Calibration offsets: ");
    Serial.print("Accel Offsets: ");
    Serial.print(offset_acc);
    Serial.print("Gyro Offset: ");
    Serial.println(offset_gyro);
  }

  BLE_initialize();

  // I should fetch the parameters here (sliding window threshold)

  startTime = millis();
}




void loop() {
  static unsigned long lastSampleTime = micros();   // Store the time of the last sample
  unsigned long currentTime = micros();
  unsigned long sampleInterval = 1000000UL / SAMPLE_RATE; // 1000000 microseconds per second

  if(buttonPressed){
    buttonPressed = false;
    sendDataOverBLE(1);
    Serial.println("Button pressed");
    notify(FALL_DETECTED);

  }
  // Check if enough time has passed since the last sample
  if (currentTime - lastSampleTime >= sampleInterval) {
    lastSampleTime += sampleInterval; // Advance to the next sample time

    float accSVM, gyroSVM;
    float acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z;
    float gforce_acc;
    sensors_event_t a_res, g_res;
    uint32_t relativeTimeMs;

    // Read sensor data (with calibration offsets if enabled)
    MPU_READ(a_res, g_res);

    acc_x = a_res.acceleration.x;
    acc_y = a_res.acceleration.y;
    acc_z = a_res.acceleration.z;

    
    gyro_x = g_res.gyro.x;
    gyro_y = g_res.gyro.y;
    gyro_z = g_res.gyro.z;
    accSVM = accelerationSVM(a_res.acceleration.x, a_res.acceleration.y, a_res.acceleration.z); // - offset_acc if I do this it should remove the g force in one axis...

    relativeTimeMs = millis() - startTime;

    gforce_acc = accSVM / 9.81;

    buffer[bufferIndex] = { acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, relativeTimeMs };
    bufferIndex = (bufferIndex + 1) % BUFFER_SIZE;

    postTriggerCount++;
    if (fallDetected && postTriggerCount > POST_TRIGGER) {
      Serial.println("Sending fall data...");
      sendDataOverBLE();
      notify(MAYBE_FALL);
      Serial.println("Data sent.");
      fallDetected = false;
    }

    
    print_data(a_res, g_res, accSVM, gyroSVM); // DEBUG

    if (gforce_acc > THRESHOLD_ACC && fallDetected == false) {
      fallDetected = true;
      Serial.println("Threshold detected...");
      Serial.println("PostFall data recording.");
      postTriggerCount = 0;
    }
  }
}
