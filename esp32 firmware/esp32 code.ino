#include <WiFi.h>
#include "esp_wifi.h"
#include <math.h>

/* ================= WIFI CREDENTIALS ================= */
const char* ssid = "Wifi name";
const char* password = "Wifi password";

/* ================= CSI CONFIG ================= */
#define NUM_SUBCARRIERS 52   // Use full useful range
bool csi_enabled = false;
bool traffic_enabled = false;

/* ================= TRAFFIC CONFIG ================= */
#define PACKET_INTERVAL 10   // ~100 Hz
unsigned long last_packet_time = 0;

/* ================= CSI CALLBACK ================= */
void IRAM_ATTR wifi_csi_cb(void *ctx, wifi_csi_info_t *info) {

    if (!info || !info->buf || info->len < 2) return;

    int total_pairs = info->len / 2;
    if (total_pairs < NUM_SUBCARRIERS) return;

    int step = total_pairs / NUM_SUBCARRIERS;
    if (step < 1) step = 1;

    Serial.print("CSI_DATA,");

    for (int i = 0; i < NUM_SUBCARRIERS; i++) {

        int idx = i * step;
        if (idx * 2 + 1 >= info->len) break;

        int8_t I = info->buf[2 * idx];
        int8_t Q = info->buf[2 * idx + 1];

        float magnitude = sqrtf((float)(I * I + Q * Q));

        Serial.print(magnitude, 1);
        if (i < NUM_SUBCARRIERS - 1)
            Serial.print(",");
    }

    Serial.println();
}

/* ================= START CSI ================= */
void start_csi() {

    if (csi_enabled) return;

    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = true,
        .manu_scale = false,
        .shift = false
    };

    esp_wifi_set_csi_config(&csi_config);
    esp_wifi_set_csi_rx_cb(wifi_csi_cb, NULL);
    esp_wifi_set_csi(true);

    esp_wifi_set_promiscuous(true);

    wifi_promiscuous_filter_t filter = {
        .filter_mask = WIFI_PROMIS_FILTER_MASK_DATA
    };

    esp_wifi_set_promiscuous_filter(&filter);

    csi_enabled = true;

    Serial.println("CSI_STARTED");
}

/* ================= STOP CSI ================= */
void stop_csi() {

    if (!csi_enabled) return;

    esp_wifi_set_promiscuous(false);
    esp_wifi_set_csi(false);

    csi_enabled = false;
    traffic_enabled = false;

    Serial.println("CSI_STOPPED");
}

/* ================= TRAFFIC GENERATION ================= */
void generate_traffic() {

    static WiFiClient client;

    if (millis() - last_packet_time >= PACKET_INTERVAL) {

        if (client.connect(WiFi.gatewayIP(), 80)) {
            client.print("GET / HTTP/1.1\r\n\r\n");
            client.stop();
        }

        last_packet_time = millis();
    }
}

/* ================= SETUP ================= */
void setup() {

    Serial.begin(115200);
    delay(2000);

    Serial.println("\n=== ESP32 CSI 3-CLASS COLLECTOR ===");
    Serial.println("Commands: START | STOP | INFO");

    WiFi.mode(WIFI_STA);

    esp_wifi_set_ps(WIFI_PS_NONE);  // VERY IMPORTANT

    WiFi.begin(ssid, password);

    Serial.print("Connecting");

    int retry = 0;

    while (WiFi.status() != WL_CONNECTED && retry < 40) {
        delay(500);
        Serial.print(".");
        retry++;
    }

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("\nWiFi FAILED");
        ESP.restart();
    }

    Serial.println("\nWiFi Connected");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
    Serial.print("Channel: ");
    Serial.println(WiFi.channel());

    Serial.println("READY");
}

/* ================= LOOP ================= */
void loop() {

    if (Serial.available()) {

        String cmd = Serial.readStringUntil('\n');
        cmd.trim();

        if (cmd == "START") {
            start_csi();
            traffic_enabled = true;
        }
        else if (cmd == "STOP") {
            stop_csi();
        }
        else if (cmd == "INFO") {
            Serial.print("CSI: ");
            Serial.println(csi_enabled ? "ON" : "OFF");
            Serial.print("RSSI: ");
            Serial.println(WiFi.RSSI());
        }
    }

    if (csi_enabled && traffic_enabled) {
        generate_traffic();
        delay(1);   // very small delay
    }
}
