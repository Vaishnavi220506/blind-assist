#include <Arduino.h>
#include <Wire.h>
extern "C" {
#include "vl53lmz_api.h"
#include "vl53lmz_plugin_cnh.h"
}
static VL53LMZ_Configuration dev;
static VL53LMZ_ResultsData result;
static VL53LMZ_Motion_Configuration cfg;
static cnh_data_buffer_t cnh;
static uint32_t bytes=0, seq=0, lastFrame=0, lastNote=0;
static bool running=false;
static_assert(VL53LMZ_NB_TARGET_PER_ZONE == 1U, "Diagnostic v2 expects one target per zone");
static int16_t distanceQ2[16];
static uint8_t actualResolution=0, actualHz=0, actualMode=0, actualOrder=0;
static uint32_t actualIntegration=0, readbackMs=0;
static bool configReadbackValid=false;
static char command[16];
static size_t commandUsed=0;
static bool commandOverflow=false;

static void printArray(const int16_t* values) {
  Serial.print('[');
  for(int i=0;i<16;++i){if(i)Serial.print(',');Serial.print(values[i]);}
  Serial.print(']');
}
static void printArray(const uint32_t* values) {
  Serial.print('[');
  for(int i=0;i<16;++i){if(i)Serial.print(',');Serial.print(values[i]);}
  Serial.print(']');
}
static void printArray(const uint16_t* values) {
  Serial.print('[');
  for(int i=0;i<16;++i){if(i)Serial.print(',');Serial.print(values[i]);}
  Serial.print(']');
}
static void printArray(const uint8_t* values) {
  Serial.print('[');
  for(int i=0;i<16;++i){if(i)Serial.print(',');Serial.print(values[i]);}
  Serial.print(']');
}
static void emitConfig() {
  Serial.printf("{\"type\":\"config\",\"firmware\":\"xiao-cnh-diag-v2\",\"status\":%u,\"running\":%s,"
    "\"readback_source\":\"cached_boot_device_getters\",\"readback_ms\":%lu,\"emitted_ms\":%lu,"
    "\"actual_resolution\":%u,\"actual_hz\":%u,\"actual_integration_ms\":%lu,"
    "\"actual_ranging_mode\":%u,\"actual_target_order\":%u,\"targets_per_zone\":1,"
    "\"zones\":16,\"bins\":24,\"start_bin\":0,\"sub_sample\":4,"
    "\"cnh_bytes\":%lu,\"read_bytes\":%lu,\"app_md5\":\"%s\"}\n",
    configReadbackValid?0:1,running?"true":"false",readbackMs,millis(),actualResolution,actualHz,
    actualIntegration,actualMode,actualOrder,bytes,dev.data_read_size,ESP.getSketchMD5().c_str());
}
static void handleCommand() {
  while(Serial.available()) {
    char ch=Serial.read();
    if(ch=='\n') {
      if(!commandOverflow) {
        command[commandUsed]=0;
        if(strcmp(command,"CONFIG")==0) emitConfig();
      }
      commandUsed=0; commandOverflow=false;
    } else if(ch!='\r') {
      if(commandUsed<sizeof(command)-1) command[commandUsed++]=ch;
      else commandOverflow=true;
    }
  }
}
static bool ok(const char* op, uint8_t s) {
  Serial.printf("{\"type\":\"step\",\"op\":\"%s\",\"status\":%u}\n",op,s);
  return s==0;
}
void setup() {
  Serial.begin(115200); delay(2000);
  Serial.println("{\"type\":\"boot\",\"firmware\":\"xiao-cnh-diag-v2\"}");
  Wire.begin(5,6,400000); Wire.setTimeOut(100);
  dev.platform.address=0x52;
  uint8_t alive=0;
  if(!ok("is_alive",vl53lmz_is_alive(&dev,&alive)) || !alive) return;
  if(!ok("init",vl53lmz_init(&dev))) return;
  if(!ok("resolution",vl53lmz_set_resolution(&dev,16))) return;
  if(!ok("mode",vl53lmz_set_ranging_mode(&dev,VL53LMZ_RANGING_MODE_AUTONOMOUS))) return;
  if(!ok("frequency",vl53lmz_set_ranging_frequency_hz(&dev,5))) return;
  if(!ok("integration",vl53lmz_set_integration_time_ms(&dev,20))) return;
  if(!ok("cnh_init",vl53lmz_cnh_init_config(&cfg,0,24,4))) return;
  if(!ok("cnh_map",vl53lmz_cnh_create_agg_map(&cfg,16,0,0,1,1,4,4))) return;
  if(!ok("cnh_memory",vl53lmz_cnh_calc_required_memory(&cfg,&bytes))) return;
  if(bytes>sizeof(cnh) || bytes%4) return;
  if(!ok("cnh_config",vl53lmz_cnh_send_config(&dev,&cfg))) return;
  // Read DCI only before ranging. Later CONFIG responses use this snapshot;
  // they never overwrite the current measurement's temp_buffer.
  if(!ok("get_resolution",vl53lmz_get_resolution(&dev,&actualResolution))) return;
  if(!ok("get_frequency",vl53lmz_get_ranging_frequency_hz(&dev,&actualHz))) return;
  if(!ok("get_integration",vl53lmz_get_integration_time_ms(&dev,&actualIntegration))) return;
  if(!ok("get_mode",vl53lmz_get_ranging_mode(&dev,&actualMode))) return;
  if(!ok("get_target_order",vl53lmz_get_target_order(&dev,&actualOrder))) return;
  readbackMs=millis(); configReadbackValid=true;
  if(!ok("output_config",vl53lmz_create_output_config(&dev))) return;
  union Block_header bh;
  bh.idx=VL53LMZ_CNH_DATA_IDX; bh.type=4; bh.size=bytes/4;
  if(!ok("cnh_block",vl53lmz_add_output_block(&dev,bh.bytes))) return;
  if(!ok("start",vl53lmz_send_output_config_and_start(&dev))) return;
  running=true; lastFrame=millis();
  emitConfig();
}
void loop() {
  handleCommand();
  if(!running) {Serial.println("{\"type\":\"error\",\"message\":\"init_failed\"}");delay(3000);return;}
  uint8_t ready=0, s=vl53lmz_check_data_ready(&dev,&ready);
  if(s) {ok("ready",s);delay(500);return;}
  if(ready) {
    s=vl53lmz_get_ranging_data(&dev,&result);
    if(s) {ok("read",s);delay(100);return;}
    // Extract the original signed Q2 distance block from the SAME received frame.
    // ULD converted its result copy, not temp_buffer. No extra sensor read here.
    s=vl53lmz_results_extract_block(&dev,VL53LMZ_DISTANCE_IDX,(uint8_t*)distanceQ2,sizeof(distanceQ2));
    if(s) {ok("extract_distance_q2",s);delay(100);return;}
    s=vl53lmz_results_extract_block(&dev,VL53LMZ_CNH_DATA_IDX,(uint8_t*)cnh,bytes);
    if(s) {ok("extract",s);delay(100);return;}
    // Validate header and pointer ranges before dereferencing variable CNH blocks.
    if((cnh[4]&0xffff)!=16 || (cnh[4]>>16)!=24) {
      Serial.printf("{\"type\":\"error\",\"message\":\"cnh_header\",\"value\":%lu}\n",cnh[4]);delay(500);return;
    }
    int32_t *hist[16], *ambient[16]; int8_t *scale[16], *ambScale[16];
    uintptr_t begin=(uintptr_t)cnh, end=begin+bytes;
    for(int a=0;a<16;++a) {
      vl53lmz_cnh_get_block_addresses(&cfg,a,cnh,&hist[a],&scale[a],&ambient[a],&ambScale[a]);
      if((uintptr_t)hist[a]<begin || (uintptr_t)(hist[a]+24)>end ||
         (uintptr_t)scale[a]<begin || (uintptr_t)(scale[a]+24)>end ||
         (uintptr_t)ambient[a]<begin || (uintptr_t)(ambient[a]+1)>end ||
         (uintptr_t)ambScale[a]<begin || (uintptr_t)(ambScale[a]+1)>end) {
        Serial.println("{\"type\":\"error\",\"message\":\"cnh_bounds\"}"); running=false;return;
      }
    }
    lastFrame=millis(); ++seq;
    Serial.printf("{\"type\":\"cnh_frame\",\"seq\":%lu,\"ms\":%lu,\"stream\":%u,\"rows\":4,\"cols\":4,\"bins\":24,\"distance_mm\":[",seq,lastFrame,dev.streamcount);
    for(int i=0;i<16;++i){if(i)Serial.print(',');Serial.print(result.distance_mm[i]);}
    Serial.print("],\"target_status\":[");
    for(int i=0;i<16;++i){if(i)Serial.print(',');Serial.print(result.target_status[i]);}
    Serial.print("],\"nb_target\":[");
    for(int i=0;i<16;++i){if(i)Serial.print(',');Serial.print(result.nb_target_detected[i]);}
    Serial.print("],\"hist_raw\":[");
    for(int a=0;a<16;++a){if(a)Serial.print(',');Serial.print('[');for(int b=0;b<24;++b){if(b)Serial.print(',');Serial.print(hist[a][b]);}Serial.print(']');}
    Serial.print("],\"hist_scaler\":[");
    for(int a=0;a<16;++a){if(a)Serial.print(',');Serial.print('[');for(int b=0;b<24;++b){if(b)Serial.print(',');Serial.print(scale[a][b]);}Serial.print(']');}
    Serial.print("],\"ambient_raw\":[");
    for(int a=0;a<16;++a){if(a)Serial.print(',');Serial.print(*ambient[a]);}
    Serial.print("],\"ambient_scaler\":[");
    for(int a=0;a<16;++a){if(a)Serial.print(',');Serial.print(*ambScale[a]);}
    Serial.print("],\"cnh_header\":[");
    for(int i=0;i<5;++i){if(i)Serial.print(',');Serial.print(cnh[i]);}
    Serial.print("],\"diagnostic\":{\"schema\":1,\"distance_q2\":");
    printArray(distanceQ2);
    Serial.print(",\"signal_kcps_spad\":"); printArray(result.signal_per_spad);
    Serial.print(",\"ambient_kcps_spad\":"); printArray(result.ambient_per_spad);
    Serial.print(",\"range_sigma_mm\":"); printArray(result.range_sigma_mm);
    Serial.print(",\"reflectance_percent\":"); printArray(result.reflectance);
    Serial.print(",\"nb_spads_enabled\":"); printArray(result.nb_spads_enabled);
    Serial.printf(",\"silicon_temp_degc\":%d}}\n",(int)result.silicon_temp_degc);
  }
  if(millis()-lastFrame>3000 && millis()-lastNote>3000){lastNote=millis();Serial.println("{\"type\":\"error\",\"message\":\"no_fresh_frame\"}");}
  delay(10);
}
