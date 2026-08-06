#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

Adafruit_SSD1306 oled(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

String inputBuffer = "";
String currentState = "NEUTRAL";

unsigned long lastFrameTime = 0;
const unsigned long frameInterval = 80;  // ms between animation frames (~12 fps)
int animFrame = 0;

void setup() {
  Serial.begin(9600);  // must match --baud on the Python side

  if (!oled.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 allocation failed"));
    while (true);
  }

  oled.setRotation(2);  // flips display 180 degrees to match your breadboard orientation

  delay(2000);
  oled.clearDisplay();
  oled.display();
}

void loop() {
  // Non-blocking serial read: buffer chars until a full line arrives
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      inputBuffer.trim();
      if (inputBuffer.length() > 0) {
        handleState(inputBuffer);
      }
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }

  // Advance the animation on its own steady clock, independent of serial timing,
  // so the face keeps moving even between posture updates from Python.
  unsigned long now = millis();
  if (now - lastFrameTime >= frameInterval) {
    lastFrameTime = now;
    animFrame++;
    drawFace(currentState, animFrame);
  }
}

void handleState(const String& state) {
  Serial.print("Received: ");
  Serial.println(state);

  if (state == "HAPPY" || state == "NEUTRAL" || state == "SAD") {
    if (state != currentState) {
      currentState = state;
      animFrame = 0;  // restart animation cleanly on state change
    }
  } else {
    Serial.println("Unknown state received");
  }
}

void drawFace(const String& state, int frame) {
  oled.clearDisplay();

  int cx = SCREEN_WIDTH / 2;
  int cy = SCREEN_HEIGHT / 2;
  int r = 20;

  if (state == "HAPPY") {
    // Bounce up/down using a triangle wave
    int cyclePos = frame % 20;
    int bounce = (cyclePos < 10) ? cyclePos : (20 - cyclePos);
    int yOffset = -(bounce - 5);

    drawSparkles(cx, cy + yOffset, r, frame);  // twinkling dots = monochrome "energy" substitute for color

    oled.drawCircle(cx, cy + yOffset, r, WHITE);
    oled.fillCircle(cx - 7, cy + yOffset - 5, 2, WHITE);
    oled.fillCircle(cx + 7, cy + yOffset - 5, 2, WHITE);
    for (int x = -10; x <= 10; x++) {
      oled.drawPixel(cx + x, cy + yOffset + 10 - (x * x) / 25, WHITE);
    }

  } else if (state == "NEUTRAL") {
    // Small, gentle idle bob
    int cyclePos = frame % 40;
    int bounce = (cyclePos < 20) ? cyclePos : (40 - cyclePos);
    int yOffset = -(bounce / 10 - 1);

    oled.drawCircle(cx, cy + yOffset, r, WHITE);
    oled.fillCircle(cx - 7, cy + yOffset - 5, 2, WHITE);
    oled.fillCircle(cx + 7, cy + yOffset - 5, 2, WHITE);
    oled.drawLine(cx - 10, cy + yOffset + 10, cx + 10, cy + yOffset + 10, WHITE);

  } else if (state == "SAD") {
    // Slight side-to-side shiver, drooped down
    int cyclePos = frame % 10;
    int shiver = (cyclePos < 5) ? cyclePos : (10 - cyclePos);
    int xOffset = shiver - 2;
    int yOffset = 3;

    oled.drawCircle(cx + xOffset, cy + yOffset, r, WHITE);
    oled.fillCircle(cx + xOffset - 7, cy + yOffset - 5, 2, WHITE);
    oled.fillCircle(cx + xOffset + 7, cy + yOffset - 5, 2, WHITE);
    for (int x = -10; x <= 10; x++) {
      oled.drawPixel(cx + xOffset + x, cy + yOffset + 14 + (x * x) / 25, WHITE);
    }
  }

  oled.display();
}

void drawSparkles(int cx, int cy, int r, int frame) {
  // Simulates "rainbow energy" on a monochrome display via twinkling dots
  // orbiting the happy face — some appear/disappear each frame for sparkle effect.
  for (int i = 0; i < 6; i++) {
    int angle = (frame * 15 + i * 60) % 360;
    float rad = angle * PI / 180.0;
    int sx = cx + (r + 10) * cos(rad);
    int sy = cy + (r + 10) * sin(rad);
    if ((frame + i) % 3 != 0) {
      oled.drawPixel(sx, sy, WHITE);
      oled.drawPixel(sx + 1, sy, WHITE);
      oled.drawPixel(sx, sy + 1, WHITE);
    }
  }
}