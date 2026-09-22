// Standalone AtomS3R-M12 USB JPEG bring-up. No ToF, Wi-Fi or alert logic.
// Pin/power reference: M5Stack M5AtomS3 examples/Basics/camera (MIT),
// https://github.com/m5stack/M5AtomS3/tree/main/examples/Basics/camera
#include <Arduino.h>
#include "esp_camera.h"
#include "esp_timer.h"

static uint32_t sequence = 0;
static bool ready = false;
static char command[32];
static size_t used = 0;

static bool sendAll(const uint8_t* data, size_t size) {
  uint32_t started = millis();
  while (size) {
    size_t chunk = size > 512 ? 512 : size;
    size_t sent = Serial.write(data, chunk);
    data += sent; size -= sent;
    if (millis()-started > 1500) return false;
    if (!sent) delay(1);
  }
  return true;
}

void setup() {
  Serial.begin(115200);
  Serial.setTxTimeoutMs(1000);
  delay(1500);
  Serial.printf("{\"type\":\"boot\",\"firmware\":\"atom-usb-jpeg-v3-otg\",\"psram_bytes\":%u}\n", ESP.getPsramSize());
  if (!psramFound()) {
    Serial.println("{\"type\":\"error\",\"message\":\"psram_missing\"}");
    return;
  }
  pinMode(18, OUTPUT);
  digitalWrite(18, LOW);
  delay(500);
  camera_config_t c = {};
  c.pin_pwdn = -1; c.pin_reset = -1;
  c.pin_xclk = 21; c.pin_sccb_sda = 12; c.pin_sccb_scl = 9;
  c.pin_d7 = 13; c.pin_d6 = 11; c.pin_d5 = 17; c.pin_d4 = 4;
  c.pin_d3 = 48; c.pin_d2 = 46; c.pin_d1 = 42; c.pin_d0 = 3;
  c.pin_vsync = 10; c.pin_href = 14; c.pin_pclk = 40;
  c.xclk_freq_hz = 20000000;
  c.ledc_timer = LEDC_TIMER_0; c.ledc_channel = LEDC_CHANNEL_0;
  c.pixel_format = PIXFORMAT_JPEG; c.frame_size = FRAMESIZE_VGA;
  c.jpeg_quality = 12; c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM; c.grab_mode = CAMERA_GRAB_LATEST;
  c.sccb_i2c_port = 0;
  esp_err_t result = esp_camera_init(&c);
  Serial.printf("{\"type\":\"camera_init\",\"status\":%d}\n", (int)result);
  ready = result == ESP_OK;
  if (ready) {
    sensor_t* sensor = esp_camera_sensor_get();
    Serial.printf("{\"type\":\"sensor\",\"pid\":%u,\"width\":640,\"height\":480}\n", sensor->id.PID);
  }
}

static void capture() {
  if (!ready) {
    Serial.println("{\"type\":\"error\",\"message\":\"camera_not_ready\"}"); return;
  }
  camera_fb_t* frame = esp_camera_fb_get();
  // Readout time after obtaining the buffer, not exposure or synchronized time.
  int64_t readout = esp_timer_get_time();
  if (!frame) {
    Serial.println("{\"type\":\"error\",\"message\":\"frame_missing\"}"); return;
  }
  char header[192];
  int length = snprintf(header, sizeof(header), "{\"type\":\"jpeg\",\"seq\":%lu,\"device_readout_us\":%lld,\"width\":%u,\"height\":%u,\"bytes\":%u}\n",
                ++sequence, readout, (unsigned)frame->width, (unsigned)frame->height, (unsigned)frame->len);
  bool sent = length > 0 && length < (int)sizeof(header) &&
              sendAll((const uint8_t*)header, length) &&
              sendAll(frame->buf, frame->len) && sendAll((const uint8_t*)"\n", 1);
  Serial.flush();
  esp_camera_fb_return(frame);
  // A partial transfer must fail visibly at the host, not silently skip bytes.
  if (!sent) ready = false;
}

void loop() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n') {
      command[used] = 0;
      if (strcmp(command, "CAPTURE") == 0) capture();
      used = 0;
    } else if (ch != '\r') {
      if (used < sizeof(command)-1) command[used++] = ch;
      else used = 0;
    }
  }
  delay(1);
}
