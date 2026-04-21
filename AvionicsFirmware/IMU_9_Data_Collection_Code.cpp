//Author : Jackson Riker
//Date: 2025-10-01
//version 1.0

// Example code to read data from ICM20948 IMU sensor and log it to an SD card in CSV format
// Libraries required: Wire.h, ICM20948_WE.h, SD.h 




#include <Wire.h>
#include <ICM20948_WE.h>
#include <SD.h>
#include <fstream>
#include <iostream>
#include <string>
#define ICM20948_ADDR 0x68

ICM20948_WE myIMU= ICM20948_WE(ICM20948_ADDR);

void setup() {
    
delay(2000) // Delay to allow time to open Serial Monitor + examples said may be needed

    // just in examples idrk
    Serial.begin(115200);
    Wire.begin();
    myIMU.begin();
    
    // Set Ranges of Gyroscope and Accelerometer
    myIMU.setGyroRange(ICM20948_GYRO_RANGE_250);
    myIMU.setAccelRange(ICM20948_ACCEL_RANGE_16G);
    
    // Check connection
    if (myICM20948.isConnected()) {
        Serial.println("ICM20948 connected successfully.");
    } else {
        Serial.println("ICM20948 connection failed.");
        while (1);
    }


    // Auto offset calibration for Gyroscope and Accelerometer
    Serial.println("Position the sensor flat and still for accurate readings.");
    delay(2000); // Wait for 2 seconds to allow user to position the sensor
    MyIMU.autoOffset();
    Serial.println("Auto offset calibration complete.");
    
    // Set Digital Low Pass Filter (DLPF) settings
    myIMU.setGyroDLPF(ICM20948_DLPF_6);
    myIMU.setAccelDLPF(ICM20948_DLPF_6);
    myIMU.setMagOpMode(Ak09916_CONT_MODE_20hz)


    // Initialize SD card
    const int chipSelect = 10; // Adjust based on your wiring
    
    if (!SD.begin(chipSelect)) {
        Serial.println("SD card initialization failed!");
        while (1);
    }
    Serial.println("SD card initialized.");

    File dataFile = SD.open("IMU9_sensor_data.csv", FILE_WRITE);
    if (dataFile) {
        dataFile.println("AccX,AccY,AccZ,CorAccX,CorAccY,CorAccZ,GValX,GValY,GValZ,ResultantG,GyrX,GyrY,GyrZ,CorGyrX,CorGyrY,CorGyrZ,MagX,MagY,MagZ");
        dataFile.close();
    } else {
        Serial.println("Error opening IMU9_sensor_data.csv");
    }

    
}
void loop() {
   
  
  // Variables to hold sensor data
   
  // Holding Acceleration data
  xyzFloat accRaw;
   xyzFloat accCorr;
   xyzFloat accG;

    // Holding Gyroscope data
   xyzFloat gyrRaw;
   xyzFloat gyr;

    // Holding Magnetometer data
   xyzFloat magValue;

   
   // Commands to get data from the sensor using given Library functions
    myIMU.readSensor()
    myIMU.getAccelRawValues(&accRaw);
    myIMU.getCorrectedAccRawValues(&accCorr);
    myIMU.getGValues(&accG);
    float resultantG = myIMU.getResultantG(&accG)
    
    myIMU.getGyrRawValues(&gyrRaw);
    myIMU.getCorrectedGyrValues(&gyr);

    myIMU.getMagValues(&magValue);


    // If File opened successfully, write data
    if (dataFile) {
        dataFile.print(accRaw.XAxis); dataFile.print(",");
        dataFile.print(accRaw.YAxis); dataFile.print(",");
        dataFile.print(accRaw.ZAxis); dataFile.print(",");
        dataFile.print(accCorr.XAxis); dataFile.print(",");
        dataFile.print(accCorr.YAxis); dataFile.print(",");
        dataFile.print(accCorr.ZAxis); dataFile.print(",");
        dataFile.print(accG.XAxis); dataFile.print(",");
        dataFile.print(accG.YAxis); dataFile.print(",");
        dataFile.print(accG.ZAxis); dataFile.print(",");
        dataFile.print(resultantG); dataFile.print(",");
        dataFile.print(gyrRaw.XAxis); dataFile.print(",");
        dataFile.print(gyrRaw.YAxis); dataFile.print(",");
        dataFile.print(gyrRaw.ZAxis); dataFile.print(",");
        dataFile.print(gyr.XAxis); dataFile.print(",");
        dataFile.print(gyr.YAxis); dataFile.print(",");
        dataFile.print(gyr.ZAxis); dataFile.print(",");
        dataFile.print(magValue.XAxis); dataFile.print(",");
        dataFile.print(magValue.YAxis); dataFile.print(",");
        dataFile.println(magValue.ZAxis);
        dataFile.close();
    } else {
        Serial.println("Error opening IMU9_sensor_data.csv");
    }
    delay(100); // Adjust the delay as needed for your sampling rate







}   