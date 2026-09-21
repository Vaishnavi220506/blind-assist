// Task-owned Arduino Wire transport for the supplied ST VL53LMZ ULD.
#include <Arduino.h>
#include <Wire.h>
#include "platform.h"
extern "C" {
uint8_t WrMulti(VL53LMZ_Platform* p, uint16_t reg, uint8_t* bytes, uint32_t n) {
  while (n) {
    size_t chunk = n > 120 ? 120 : n;
    Wire.beginTransmission(p->address >> 1);
    Wire.write(uint8_t(reg >> 8)); Wire.write(uint8_t(reg));
    if (Wire.write(bytes, chunk) != chunk) { Wire.endTransmission(); return 1; }
    if (Wire.endTransmission() != 0) return 1;
    reg += chunk; bytes += chunk; n -= chunk;
  }
  return 0;
}
uint8_t RdMulti(VL53LMZ_Platform* p, uint16_t reg, uint8_t* bytes, uint32_t n) {
  while (n) {
    size_t chunk = n > 120 ? 120 : n;
    Wire.beginTransmission(p->address >> 1);
    Wire.write(uint8_t(reg >> 8)); Wire.write(uint8_t(reg));
    if (Wire.endTransmission(false) != 0) return 1;
    if (Wire.requestFrom(uint8_t(p->address >> 1), chunk) != chunk) return 1;
    for (size_t i=0; i<chunk; ++i) bytes[i] = Wire.read();
    reg += chunk; bytes += chunk; n -= chunk;
  }
  return 0;
}
uint8_t RdByte(VL53LMZ_Platform* p, uint16_t r, uint8_t* b) { return RdMulti(p,r,b,1); }
uint8_t WrByte(VL53LMZ_Platform* p, uint16_t r, uint8_t b) { return WrMulti(p,r,&b,1); }
void SwapBuffer(uint8_t* b, uint16_t n) {
  for (uint16_t i=0; i+3<n; i+=4) {
    uint8_t t=b[i]; b[i]=b[i+3]; b[i+3]=t;
    t=b[i+1]; b[i+1]=b[i+2]; b[i+2]=t;
  }
}
uint8_t WaitMs(VL53LMZ_Platform*, uint32_t ms) { delay(ms); return 0; }
}
