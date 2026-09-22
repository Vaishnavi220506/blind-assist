#include <Arduino.h>
#include <Wire.h>
extern "C" {
#include "vl53lmz_api.h"
}
static VL53LMZ_Configuration device;
static VL53LMZ_ResultsData result;
static bool running=false;
static uint32_t frames=0, lastFrame=0, lastNote=0;
static bool check(const char* op, uint8_t status) {
  Serial.printf("{\"type\":\"step\",\"op\":\"%s\",\"status\":%u}\n",op,status);
  return status==0;
}
void setup() {
  Serial.begin(115200); delay(2000);
  Serial.println("{\"type\":\"boot\",\"firmware\":\"xiao-tof-8x8-v1\",\"sda\":5,\"scl\":6}");
  Wire.begin(5,6,400000); Wire.setTimeOut(100);
  device.platform.address=0x52;
  uint8_t alive=0;
  if(!check("is_alive",vl53lmz_is_alive(&device,&alive)) || !alive) return;
  Serial.printf("{\"type\":\"identity\",\"device_id\":%u,\"revision\":%u,\"module_type\":%u}\n",device.device_id,device.revision_id,device.module_type);
  if(!check("init",vl53lmz_init(&device))) return;
  if(!check("resolution_8x8",vl53lmz_set_resolution(&device,VL53LMZ_RESOLUTION_8X8))) return;
  if(!check("frequency_5hz",vl53lmz_set_ranging_frequency_hz(&device,5))) return;
  if(!check("start",vl53lmz_start_ranging(&device))) return;
  running=true; lastFrame=millis();
}
void loop() {
  if(!running) {
    Serial.println("{\"type\":\"error\",\"message\":\"initialization_failed\"}"); delay(3000); return;
  }
  uint8_t ready=0, status=vl53lmz_check_data_ready(&device,&ready);
  if(status) { check("data_ready",status); delay(500); return; }
  if(ready) {
    if(!check("read",vl53lmz_get_ranging_data(&device,&result))) {delay(100);return;}
    lastFrame=millis(); ++frames;
    Serial.printf("{\"type\":\"frame\",\"seq\":%lu,\"ms\":%lu,\"stream\":%u,\"rows\":8,\"cols\":8,\"distance_mm\":[",frames,lastFrame,device.streamcount);
    for(int i=0;i<64;++i) {if(i)Serial.print(',');Serial.print(result.distance_mm[i]);}
    Serial.print("],\"target_status\":[");
    for(int i=0;i<64;++i) {if(i)Serial.print(',');Serial.print(result.target_status[i]);}
    Serial.print("],\"nb_target\":[");
    for(int i=0;i<64;++i) {if(i)Serial.print(',');Serial.print(result.nb_target_detected[i]);}
    Serial.println("]}");
  }
  if(millis()-lastFrame>3000 && millis()-lastNote>3000) {
    lastNote=millis();Serial.println("{\"type\":\"error\",\"message\":\"no_fresh_frame\"}");
  }
  delay(10);
}
