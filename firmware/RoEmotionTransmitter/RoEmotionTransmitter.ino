// Board: Seeed Studio XIAO ESP32C3. Active-high green LED on D4; red on D2.
// Default preserves the ORIGINAL steady-state green waveform for the bundled model.
#define USER_ID 1
#define CANONICAL_LABEL_PATTERN 0

static_assert(USER_ID >= 1 && USER_ID <= 3, "USER_ID must be 1, 2, or 3");
constexpr unsigned int SYMBOL_US = 166;
constexpr int RED_PIN = D2;
constexpr int GREEN_PIN = D4;
const char* legacyPatterns[] = {"01010", "01110", "00110"};
const char* canonicalPatterns[] = {"10101", "01110", "00110"};

void setup() {
  pinMode(RED_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  digitalWrite(RED_PIN, LOW);
  digitalWrite(GREEN_PIN, LOW);
}

void loop() {
  const char* pattern = CANONICAL_LABEL_PATTERN
      ? canonicalPatterns[USER_ID - 1] : legacyPatterns[USER_ID - 1];
  for (int symbol = 0; symbol < 5; ++symbol) {
    digitalWrite(GREEN_PIN, pattern[symbol] == '1' ? HIGH : LOW);
    delayMicroseconds(SYMBOL_US);
  }
}
