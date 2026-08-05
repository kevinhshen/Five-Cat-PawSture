#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// Arduino UNO R4 Minima + common 128x64 I2C SSD1306 OLED.
// Install libraries in Arduino IDE:
//   - Adafruit SSD1306
//   - Adafruit GFX Library

const unsigned long BAUD_RATE = 115200;
const int SCREEN_WIDTH = 128;
const int SCREEN_HEIGHT = 64;
const int OLED_RESET = -1;
const int OLED_ADDR = 0x3C;

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

String currentState = "WAITING";
unsigned long lastMessageMs = 0;
const unsigned long CONNECTION_TIMEOUT_MS = 6000;

void setup() {
  Serial.begin(BAUD_RATE);

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    while (true) {
      delay(100);
    }
  }

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  showState("WAITING", "Start Python app");
}

void loop() {
  readSerialState();

  if (currentState != "WAITING" && millis() - lastMessageMs > CONNECTION_TIMEOUT_MS) {
    currentState = "WAITING";
    showState("WAITING", "No signal");
  }
}

void readSerialState() {
  if (!Serial.available()) {
    return;
  }

  String incoming = Serial.readStringUntil('\n');
  incoming.trim();
  incoming.toUpperCase();

  if (incoming == "HAPPY" || incoming == "NEUTRAL" || incoming == "SAD") {
    currentState = incoming;
    lastMessageMs = millis();
    showPostureState(currentState);
  }
}

void showPostureState(String state) {
  if (state == "HAPPY") {
    showState("HAPPY", "Good posture");
  } else if (state == "NEUTRAL") {
    showState("NEUTRAL", "Adjust soon");
  } else if (state == "SAD") {
    showState("SAD", "Sit upright");
  }
}

void showState(String title, String subtitle) {
  display.clearDisplay();

  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("Posture Cube");

  display.drawLine(0, 12, SCREEN_WIDTH, 12, SSD1306_WHITE);

  display.setTextSize(2);
  int16_t x1;
  int16_t y1;
  uint16_t w;
  uint16_t h;
  display.getTextBounds(title, 0, 0, &x1, &y1, &w, &h);
  display.setCursor((SCREEN_WIDTH - w) / 2, 24);
  display.print(title);

  display.setTextSize(1);
  display.getTextBounds(subtitle, 0, 0, &x1, &y1, &w, &h);
  display.setCursor((SCREEN_WIDTH - w) / 2, 52);
  display.print(subtitle);

  display.display();
}
