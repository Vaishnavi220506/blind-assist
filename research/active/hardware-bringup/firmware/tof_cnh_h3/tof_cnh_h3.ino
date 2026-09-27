// 8x8-zone CNH in the simulator's H3 layout: start_bin 0, 16 bins, sub_sample 8
// (0.30 m bins to 4.8 m), 5 Hz autonomous, 20 ms per-zone integration as tof_cnh.
// Bring-up/frame-rate check only; same JSON framing as tof_cnh with 8x8 arrays.
#include <Arduino.h>
#include <Wire.h>
extern "C" {
#include "vl53lmz_api.h"
#include "vl53lmz_plugin_cnh.h"
}
#ifndef CNH_I2C_HZ
#define CNH_I2C_HZ 1000000  // 400 kHz measured 3.81 Hz (169.5 ms per read) on 2026-09-27
#endif
static const int Z=64, NB=16;
static VL53LMZ_Configuration dev;
static VL53LMZ_ResultsData result;
static VL53LMZ_Motion_Configuration cfg;
static cnh_data_buffer_t cnh;
static uint32_t bytes=0, seq=0, lastFrame=0, lastNote=0, readUs=0;
static bool running=false;
static const char* lastOp="none";
static unsigned lastStatus=0;
static bool ok(const char* op, uint8_t s) {
  Serial.printf("{\"type\":\"step\",\"op\":\"%s\",\"status\":%u}\n",op,s);
  lastOp=op; lastStatus=s;
  return s==0;
}
void setup() {
  Serial.begin(115200); delay(2000);
  Serial.println("{\"type\":\"boot\",\"firmware\":\"xiao-cnh-h3-v1\"}");
  Wire.begin(5,6,CNH_I2C_HZ); Wire.setTimeOut(100);
  dev.platform.address=0x52;
  uint8_t alive=0;
  if(!ok("is_alive",vl53lmz_is_alive(&dev,&alive)) || !alive) return;
  if(!ok("init",vl53lmz_init(&dev))) return;
  if(!ok("resolution",vl53lmz_set_resolution(&dev,Z))) return;
  if(!ok("mode",vl53lmz_set_ranging_mode(&dev,VL53LMZ_RANGING_MODE_AUTONOMOUS))) return;
  if(!ok("frequency",vl53lmz_set_ranging_frequency_hz(&dev,5))) return;
  if(!ok("integration",vl53lmz_set_integration_time_ms(&dev,20))) return;
  if(!ok("cnh_init",vl53lmz_cnh_init_config(&cfg,0,NB,8))) return;
  if(!ok("cnh_map",vl53lmz_cnh_create_agg_map(&cfg,Z,0,0,1,1,8,8))) return;
  if(!ok("cnh_memory",vl53lmz_cnh_calc_required_memory(&cfg,&bytes))) return;
  if(bytes>sizeof(cnh) || bytes%4) {Serial.printf("{\"type\":\"error\",\"message\":\"cnh_size\",\"bytes\":%lu}\n",bytes);return;}
  if(!ok("cnh_config",vl53lmz_cnh_send_config(&dev,&cfg))) return;
  if(!ok("output_config",vl53lmz_create_output_config(&dev))) return;
  union Block_header bh;
  bh.idx=VL53LMZ_CNH_DATA_IDX; bh.type=4; bh.size=bytes/4;
  if(!ok("cnh_block",vl53lmz_add_output_block(&dev,bh.bytes))) return;
  if(!ok("start",vl53lmz_send_output_config_and_start(&dev))) return;
  Serial.printf("{\"type\":\"config\",\"zones\":%d,\"bins\":%d,\"start_bin\":0,\"sub_sample\":8,\"base_bin_mm\":37.5348,\"i2c_hz\":%lu,\"cnh_bytes\":%lu,\"read_bytes\":%lu}\n",
                Z,NB,(unsigned long)CNH_I2C_HZ,bytes,dev.data_read_size);
  running=true; lastFrame=millis();
}
void loop() {
  if(!running) {Serial.printf("{\"type\":\"error\",\"message\":\"init_failed\",\"op\":\"%s\",\"status\":%u,\"cnh_bytes\":%lu}\n",lastOp,lastStatus,bytes);delay(3000);return;}
  uint8_t ready=0, s=vl53lmz_check_data_ready(&dev,&ready);
  if(s) {ok("ready",s);delay(500);return;}
  if(ready) {
    uint32_t t0=micros();
    s=vl53lmz_get_ranging_data(&dev,&result);
    readUs=micros()-t0;
    if(s) {ok("read",s);delay(100);return;}
    s=vl53lmz_results_extract_block(&dev,VL53LMZ_CNH_DATA_IDX,(uint8_t*)cnh,bytes);
    if(s) {ok("extract",s);delay(100);return;}
    // Validate header and pointer ranges before dereferencing variable CNH blocks.
    if((cnh[4]&0xffff)!=Z || (cnh[4]>>16)!=NB) {
      Serial.printf("{\"type\":\"error\",\"message\":\"cnh_header\",\"value\":%lu}\n",cnh[4]);delay(500);return;
    }
    int32_t *hist[Z], *ambient[Z]; int8_t *scale[Z], *ambScale[Z];
    uintptr_t begin=(uintptr_t)cnh, end=begin+bytes;
    for(int a=0;a<Z;++a) {
      vl53lmz_cnh_get_block_addresses(&cfg,a,cnh,&hist[a],&scale[a],&ambient[a],&ambScale[a]);
      if((uintptr_t)hist[a]<begin || (uintptr_t)(hist[a]+NB)>end ||
         (uintptr_t)scale[a]<begin || (uintptr_t)(scale[a]+NB)>end ||
         (uintptr_t)ambient[a]<begin || (uintptr_t)(ambient[a]+1)>end ||
         (uintptr_t)ambScale[a]<begin || (uintptr_t)(ambScale[a]+1)>end) {
        Serial.println("{\"type\":\"error\",\"message\":\"cnh_bounds\"}"); running=false;return;
      }
    }
    lastFrame=millis(); ++seq;
    Serial.printf("{\"type\":\"cnh_frame\",\"seq\":%lu,\"ms\":%lu,\"read_us\":%lu,\"stream\":%u,\"rows\":8,\"cols\":8,\"bins\":%d,\"distance_mm\":[",seq,lastFrame,readUs,dev.streamcount,NB);
    for(int i=0;i<Z;++i){if(i)Serial.print(',');Serial.print(result.distance_mm[i]);}
    Serial.print("],\"target_status\":[");
    for(int i=0;i<Z;++i){if(i)Serial.print(',');Serial.print(result.target_status[i]);}
    Serial.print("],\"nb_target\":[");
    for(int i=0;i<Z;++i){if(i)Serial.print(',');Serial.print(result.nb_target_detected[i]);}
    Serial.print("],\"hist_raw\":[");
    for(int a=0;a<Z;++a){if(a)Serial.print(',');Serial.print('[');for(int b=0;b<NB;++b){if(b)Serial.print(',');Serial.print(hist[a][b]);}Serial.print(']');}
    Serial.print("],\"hist_scaler\":[");
    for(int a=0;a<Z;++a){if(a)Serial.print(',');Serial.print('[');for(int b=0;b<NB;++b){if(b)Serial.print(',');Serial.print(scale[a][b]);}Serial.print(']');}
    Serial.print("],\"ambient_raw\":[");
    for(int a=0;a<Z;++a){if(a)Serial.print(',');Serial.print(*ambient[a]);}
    Serial.print("],\"ambient_scaler\":[");
    for(int a=0;a<Z;++a){if(a)Serial.print(',');Serial.print(*ambScale[a]);}
    Serial.print("],\"cnh_header\":[");
    for(int i=0;i<5;++i){if(i)Serial.print(',');Serial.print(cnh[i]);}
    Serial.println("]}");
  }
  if(millis()-lastFrame>3000 && millis()-lastNote>3000){lastNote=millis();Serial.println("{\"type\":\"error\",\"message\":\"no_fresh_frame\"}");}
  delay(2);
}
