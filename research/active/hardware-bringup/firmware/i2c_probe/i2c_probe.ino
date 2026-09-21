#include <Arduino.h>
#include <Wire.h>

// XIAO ESP32-S3: D4 = GPIO5 (SDA), D5 = GPIO6 (SCL).
// No camera, network, or other GPIO peripherals are initialized.
static int readReg(uint16_t reg) {
  Wire.beginTransmission(0x29);
  Wire.write(uint8_t(reg >> 8)); Wire.write(uint8_t(reg));
  if (Wire.endTransmission(false) != 0) return -1;
  if (Wire.requestFrom(uint8_t(0x29), uint8_t(1)) != 1) return -1;
  return Wire.read();
}
void setup() {
  Serial.begin(115200);
  delay(1500);
  Wire.begin(5, 6, 100000);
  Wire.setTimeOut(50);
}
void loop() {
  Serial.printf("PROBE ms=%lu SDA=5 SCL=6 sda_level=%d scl_level=%d\n",
                millis(), digitalRead(5), digitalRead(6));
  int count = 0;
  for (uint8_t addr = 8; addr < 0x78; ++addr) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      ++count;
      Serial.printf("I2C_FOUND 0x%02X\n", addr);
      if (addr == 0x29) {
        Wire.beginTransmission(0x29);
        Wire.write(0x7f); Wire.write(0xff); Wire.write(0x00);
        int status = Wire.endTransmission();
        int id = readReg(0), rev = readReg(1);
        Serial.printf("TOF_ID page_status=%d device=0x%02X revision=0x%02X\n", status, id, rev);
        Wire.beginTransmission(0x29);
        Wire.write(0x7f); Wire.write(0xff); Wire.write(0x02);
        Wire.endTransmission();
      }
    }
  }
  Serial.printf("SCAN_DONE devices=%d\n", count);
  delay(3000);
}
