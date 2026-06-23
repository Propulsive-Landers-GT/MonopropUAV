/* Documentation + Resources
    Component Datasheet: https://content.u-blox.com/sites/default/files/products/documents/u-blox8-M8_ReceiverDescrProtSpec_UBX-13003221.pdf?utm_content=UBX-13003221
    (Manual parsing instructions given on page 154 including formating of data in that section)
    
    The packet library: https://github.com/loginov-rocks/UbxGps/blob/main/README.md (same as sample)

    SD library + repo: https://docs.arduino.cc/libraries/sd/

    Altitude Resources (difference between orthonometric and ellipsoid height): 
    https://www.unavco.org/education/education-resources/tutorials/the-shape-of-earth-and-reference-ellipsoids/
    https://www.unavco.org/education/education-resources/tutorials/the-geoid-and-receiver-measurements/
*/

/* Flow control (to be populated), will work on this soon
*/

// GPS.cpp - By Aazam - Version 1
#include <SoftwareSerial.h>
#include <UbxGpsNavPvt.h>
#include <SD.h>

#define BAUDRATE 9600
#define GPS_RX 4
#define GPS_TX 5
#define SD_CS_PIN 10
bool sdAvailable = false;
//Placeholder values ^^^

SoftwareSerial gpsSerial(GPS_RX, GPS_TX); // RX, TX on Arduino pins - "extra serial ports"
//Only one hardware serial port exists on arduino that is usually connected to the computer through usb so must create software ones
UbxGpsNavPvt<SoftwareSerial> gps(gpsSerial); //Library that parses all the data automatically

void serialConnection() {
    Serial.begin(BAUDRATE); //Computer
    gpsSerial.begin(BAUDRATE); //GPS connection
}

void createDataFile() {
    File gpsFile = SD.open("gps_data.csv", FILE_WRITE); //append if not exist else create one
    if (gpsFile) {
        gpsFile.println("Date,Time,Latitude,Longitude,Altitude");
        gpsFile.close();
    } else {
        Serial.println("Error creating file!");
    } //Checks file properly created
}

void LogGpsData() {
    gps.read(); // parse incoming GPS data

    if (gps.ready()) {
        File gpsFile = SD.open("gps_data.csv", FILE_WRITE);

        if (gpsFile) {
            char timeStr[9]; // Allocates 9 spaces in memory for HH:MM:SS
            snprintf(timeStr, sizeof(timeStr), "%02d:%02d:%02d", gps.hour, gps.min, gps.sec); //Writes into the string memory

            char dateStr[11]; // Allocates 11 spaces in memory for MM/DD/YYYY - American Format
            snprintf(dateStr, sizeof(dateStr), "%02d/%02d/%04d", gps.month, gps.day, gps.year);

            gpsFile.print(dateStr); gpsFile.print(",");
            gpsFile.print(timeStr); gpsFile.print(",");
            gpsFile.print(gps.lat * 1e-7, 7); gpsFile.print(",");
            gpsFile.print(gps.lon * 1e-7, 7); gpsFile.print(",");
            gpsFile.println(gps.height / 1000.0, 3); //3dp of precision, change as necessary
            //Lat,long and height need to be corrected for true SI unit values (scaled in the UBX library not NMEA fyi)

            gpsFile.close();
        } 
        
        else {
            Serial.println("Error opening file for append!");
        } //Checks file created properly
    }
}

void setup() {
    // Start serial connections
    serialConnection(); 

    // Initialize SD card
    sdAvailable = SD.begin(SD_CS_PIN); //true if initialized correctly
    if (!sdAvailable) {
        Serial.println("SD card initialization failed!");
    }
    else {
        Serial.println("SD card initialized.");
    }

    createDataFile(); //Create CSV File with header

    Serial.println("Setup complete. Logging GPS data...");
}

void loop() {
    // Continuously log GPS data
    if (sdAvailable) {
        LogGpsData(); //only logs data if the sd card is available
    }

    // Small delay to avoid flooding SD card (adjust as needed)
    delay(100); // 100 ms = 10 Hz logging
}

//File is opened and closed continously to avoid keeping it open for long periods of time and big chunks of data getting corrupted during flight issues

/* A few notes
* Ellipsoid height used currently, can be switched to orthonometric depending on rocket/monoprop trajectories
* Logged all data into a single csv file, if too messy we can create different csv files
* Feel free to remove scales on the lat,long + height to avoid decimals
* Handed parsing to the packet (also done in sample), if there was/is an issue, I can parse it manually instead!
*/