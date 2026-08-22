#include <Arduino.h>
#include <driver/i2s.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <UniversalTelegramBot.h>
#include <ArduinoJson.h>

/* ============================================================
   KONFIGURASI - SESUAIKAN SEBELUM UPLOAD
   ============================================================ */
#define WIFI_SSID             "Redmi"
#define WIFI_PASSWORD         "12345678"
#define BOT_TOKEN             "8770817520:AAGPvvBS-Gvogo7S2nUsfpgVOfHaJkknMiI"
#define CHAT_ID               "5879026426"

/* [BARU v3] KONFIGURASI GOOGLE SHEETS (via Google Apps Script Web App)
   ------------------------------------------------------------
   Cara mendapatkan nilai-nilai ini: lihat file GoogleAppsScript_Code.gs
   (skrip yang harus Anda deploy di Google Apps Script), bagian
   komentar "CARA DEPLOY" di paling atas file tersebut.
   Ringkasnya:
   1. Buat Google Sheet baru -> Extensions -> Apps Script
   2. Tempel isi GoogleAppsScript_Code.gs, ganti SECRET_TOKEN
   3. Deploy -> New deployment -> Web app -> Execute as: Me,
      Who has access: Anyone
   4. Salin "Web app URL" -> GOOGLE_SCRIPT_URL di bawah
   5. Samakan GOOGLE_SCRIPT_TOKEN dengan SECRET_TOKEN di Apps Script
   ------------------------------------------------------------
   CATATAN: hari/tanggal/waktu DIHITUNG OTOMATIS oleh Google Apps
   Script (server Google), sehingga ESP32 TIDAK butuh modul RTC
   ataupun sinkronisasi NTP sama sekali. ESP32 cukup mengirim
   pesan alertnya saja. */
#define GOOGLE_SCRIPT_URL      "https://script.google.com/macros/s/AKfycbxA-jgsptLIEbzOYaQTCqV94P-dj07dIn4qfIRYLrTdakO61ej8Iirw1e4189HCvPkf/exec"
#define GOOGLE_SCRIPT_TOKEN    "chainsawdetectorsheets"   /* harus SAMA PERSIS dgn SECRET_TOKEN di Apps Script */

/* [BARU v3] Jumlah riwayat yang ditampilkan sbg teks / dikirim sbg file .csv */
#define HISTORY_TAMPIL_JUMLAH   10
#define HISTORY_DOWNLOAD_JUMLAH 50

/* ============================================================
   PIN I2S INMP441
   ============================================================ */
#define I2S_PORT              I2S_NUM_0
#define I2S_WS_PIN            15
#define I2S_SCK_PIN           14
#define I2S_SD_PIN            32

/* ============================================================
   PARAMETER AUDIO
   ============================================================ */
#define SAMPLE_RATE           16000
#define DMA_BUF_COUNT         8
#define DMA_BUF_LEN           256
#define PACKET_SAMPLES        256
#define PACKET_BYTES          (PACKET_SAMPLES * 2)

/* ============================================================
   PIN LED & BUZZER
   ============================================================ */
#define LED_PIN               2
#define BUZZER_PIN            -1    /* -1 = tidak ada buzzer */

/* ============================================================
   TIMING
   ============================================================ */
#define TELEGRAM_COOLDOWN_MS   15000UL  /* cooldown alert chainsaw */
#define MAINTENANCE_INTERVAL   3000UL   /* cek WiFi + polling Telegram setiap 3 detik */
#define WIFI_RECONNECT_MAX     20       /* max retry reconnect */

/* ============================================================
   VARIABEL GLOBAL
   ============================================================ */
volatile bool streamingAktif  = false;
volatile bool alertDiterima   = false;
float         alertProbCS     = 0.0f;

/* set true sesaat oleh cekPerintahTelegram() saat /off diterima,
   supaya jalankanMaintenanceBerkala() tahu untuk TIDAK menyalakan lagi
   streamingAktif setelah jendela maintenance selesai */
volatile bool permintaanOffDariTelegram = false;

/* Status WiFi - untuk deteksi perubahan ON/OFF */
bool          wifiTerhubung   = false;   /* status WiFi saat ini */
bool          notifOnTerkirim = false;   /* agar notif ON tidak spam */

unsigned long lastTelegramMs     = 0;
unsigned long lastMaintenanceMs  = 0;
unsigned long lastLedMs          = 0;
bool          ledState           = false;

WiFiClientSecure       telegramClient;
UniversalTelegramBot*  bot = nullptr;

/* Buffer audio I2S */
int16_t  i2sBuffer[PACKET_SAMPLES];
uint8_t  packetBuf[PACKET_BYTES + 4];

/* [BARU v3] buffer CSV riwayat (diisi langsung dari respons Apps Script)
   untuk dikirim sbg dokumen Telegram */
String  riwayatCSVBuffer;
size_t  riwayatCSVIndex = 0;

/* ============================================================
   DEKLARASI FUNGSI
   ============================================================ */
void setupI2S();
void setupWifi();
void cekStatusWifi();
void jalankanMaintenanceBerkala();
void cekPerintahTelegram();
void prosesPerintah(String cmd);
void bacaDanKirimAudio();
void prosesAlert(float probCS);
bool kirimTelegram(String pesan);
void kirimNotifOn();
void kirimNotifOff();
bool simpanRiwayatSheets(String pesan);
void kirimRiwayatTeks(int jumlah);
void kirimRiwayatFile(int jumlah);
bool csvMoreDataAvailable();
uint8_t csvGetNextByte();

/* ============================================================
   SETUP
   ============================================================ */
void setup() {
    Serial.begin(921600);
    delay(300);
    Serial.setTimeout(50);

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    #if BUZZER_PIN >= 0
        pinMode(BUZZER_PIN, OUTPUT);
        digitalWrite(BUZZER_PIN, LOW);
    #endif

    /* Setup I2S dulu agar mikrofon siap */
    setupI2S();

    /* Hubungkan WiFi dan kirim notif ON */
    setupWifi();

    /* Beri tahu MATLAB bahwa ESP32 siap menerima perintah */
    Serial.println("[ESP32] READY");
}

/* ============================================================
   LOOP UTAMA
   ============================================================ */
void loop() {

    /* --- 1. Cek perintah dari MATLAB --- */
    if (Serial.available()) {
        String perintah = Serial.readStringUntil('\n');
        perintah.trim();
        perintah.replace("\r", "");
        perintah.replace("\n", "");
        if (perintah.length() > 0) {
            prosesPerintah(perintah);
        }
    }

    /* --- 2. Streaming audio ke MATLAB --- */
    if (streamingAktif) {
        bacaDanKirimAudio();
    }

    /* --- 3. Proses alert chainsaw dari MATLAB --- */
    if (alertDiterima) {
        alertDiterima = false;
        bool wasStreaming = streamingAktif;
        streamingAktif = false;     /* pause streaming saat kirim Telegram/Sheets */
        prosesAlert(alertProbCS);
        streamingAktif = wasStreaming;
    }

    /* --- 4. Maintenance berkala: WiFi + Telegram (/on /off /history) --- */
    /* Dijalankan TERLEPAS dari status streaming (bukan hanya saat idle),
       supaya /off tetap bisa diterima walau ESP32 sedang mengirim PCM.
       Streaming dijeda sesaat (pola sama seperti pengiriman alert di atas)
       agar teks perintah "#CMD:ON"/"#CMD:OFF" tidak pernah tercampur
       dengan paket biner PCM yang sedang dikirim ke MATLAB. */
    unsigned long now = millis();
    if (now - lastMaintenanceMs >= MAINTENANCE_INTERVAL) {
        lastMaintenanceMs = now;
        jalankanMaintenanceBerkala();
    }

    /* --- 5. LED kedip saat streaming --- */
    if (streamingAktif) {
        unsigned long nowMs = millis();
        if (nowMs - lastLedMs > 500) {
            ledState = !ledState;
            digitalWrite(LED_PIN, ledState ? HIGH : LOW);
            lastLedMs = nowMs;
        }
    }
}

/* ============================================================
   FUNGSI: MAINTENANCE BERKALA
   Menjeda streaming sesaat, cek WiFi, lalu poll perintah Telegram.
   ============================================================ */
void jalankanMaintenanceBerkala() {
    bool wasStreaming = streamingAktif;
    streamingAktif = false;   /* jeda dulu, supaya Serial bersih dari PCM */

    cekStatusWifi();

    permintaanOffDariTelegram = false;
    if (wifiTerhubung) {
        cekPerintahTelegram();
    }

    /* Jika /off baru saja diterima, JANGAN nyalakan lagi streaming */
    if (permintaanOffDariTelegram) {
        wasStreaming = false;
    }

    streamingAktif = wasStreaming;
}

/* ============================================================
   FUNGSI: CEK & PROSES PERINTAH DARI TELEGRAM
   Perintah yang didukung: /on, /off, /history, /help
   Hanya pesan dari CHAT_ID yang diizinkan (keamanan dasar).
   ============================================================ */
void cekPerintahTelegram() {
    if (bot == nullptr) return;

    int numNew = bot->getUpdates(bot->last_message_received + 1);

    while (numNew) {
        for (int i = 0; i < numNew; i++) {

            /* Hanya proses perintah dari CHAT_ID yang terdaftar */
            if (bot->messages[i].chat_id != String(CHAT_ID)) {
                continue;
            }

            String teks = bot->messages[i].text;
            teks.trim();

            if (teks.equalsIgnoreCase("/on")) {
                /* Teruskan ke MATLAB lewat Serial (dijeda dari PCM, aman) */
                Serial.print("#CMD:ON\n");
                kirimTelegram("Perintah diterima.\nMengaktifkan uji realtime MATLAB...");

            } else if (teks.equalsIgnoreCase("/off")) {
                Serial.print("#CMD:OFF\n");
                permintaanOffDariTelegram = true;
                kirimTelegram("Perintah diterima.\nMenghentikan uji realtime MATLAB.");

            } else if (teks.equalsIgnoreCase("/history")) {
                kirimRiwayatTeks(HISTORY_TAMPIL_JUMLAH);
                kirimLinkDownloadCSV(HISTORY_DOWNLOAD_JUMLAH);

            } else if (teks.equalsIgnoreCase("/start") || teks.equalsIgnoreCase("/help")) {
                kirimTelegram(
                    "Sistem Deteksi Chainsaw - Perintah tersedia:\n"
                    "/on      - mulai uji realtime MATLAB\n"
                    "/off     - hentikan uji realtime MATLAB\n"
                    "/history - tampilkan & unduh riwayat deteksi"
                );
            }
            /* Perintah tak dikenal -> diabaikan (tidak dibalas, hindari spam) */
        }
        numNew = bot->getUpdates(bot->last_message_received + 1);
    }
}

/* ============================================================
   FUNGSI: SETUP WIFI + KIRIM NOTIF ON
   Dipanggil sekali di setup()
   ============================================================ */
void setupWifi() {
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int retry = 0;
    while (WiFi.status() != WL_CONNECTED && retry < WIFI_RECONNECT_MAX) {
        delay(500);
        retry++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        wifiTerhubung   = true;
        notifOnTerkirim = false;

        /* Inisialisasi Telegram bot */
        telegramClient.setInsecure();
        if (bot == nullptr) {
            bot = new UniversalTelegramBot(BOT_TOKEN, telegramClient);
        }

        /* Kirim notif ESP32 ON ke Telegram */
        kirimNotifOn();

    } else {
        wifiTerhubung = false;
        /* Tidak bisa kirim Telegram karena WiFi gagal */
        /* Streaming ke MATLAB tetap bisa berjalan */
    }
}

void kirimLinkDownloadCSV(int jumlah) {
    if (WiFi.status() != WL_CONNECTED) {
        kirimTelegram("WiFi tidak terhubung, link download tidak dapat dibuat.");
        return;
    }

    String url = String(GOOGLE_SCRIPT_URL) +
                 "?token=" + GOOGLE_SCRIPT_TOKEN +
                 "&mode=csv&jumlah=" + String(jumlah);

    String pesan =
        "📥 <b>DOWNLOAD RIWAYAT DETEKSI</b>\n"
        "================================\n"
        "Jumlah data: " + String(jumlah) + " data terakhir\n\n"
        "Klik tombol/link berikut untuk mengunduh file CSV:\n\n"
        "<a href=\"" + url + "\">📥 DOWNLOAD CSV</a>";

    bot->sendMessage(CHAT_ID, pesan, "HTML");
}

/* ============================================================
   FUNGSI: CEK STATUS WIFI SECARA BERKALA
   Deteksi perubahan status dan kirim notif ON/OFF
   ============================================================ */
void cekStatusWifi() {
    bool statusSekarang = (WiFi.status() == WL_CONNECTED);

    /* ---- WiFi baru saja PUTUS ---- */
    if (wifiTerhubung && !statusSekarang) {
        wifiTerhubung   = false;
        notifOnTerkirim = false;
        /* Tidak bisa kirim Telegram karena sudah offline */
        /* LED nyala terus sebagai indikator WiFi putus */
        digitalWrite(LED_PIN, HIGH);

        /* Coba reconnect */
        WiFi.disconnect();
        delay(100);
        WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    }

    /* ---- WiFi baru saja TERHUBUNG LAGI ---- */
    if (!wifiTerhubung && statusSekarang) {
        wifiTerhubung = true;

        /* Inisialisasi ulang bot jika belum ada */
        if (bot == nullptr) {
            telegramClient.setInsecure();
            bot = new UniversalTelegramBot(BOT_TOKEN, telegramClient);
        }

        /* Kirim notif ON (reconnect) */
        if (!notifOnTerkirim) {
            kirimNotifOn();
        }
    }

    /* ---- WiFi tetap terputus, belum berhasil reconnect ---- */
    if (!wifiTerhubung && !statusSekarang) {
        /* Coba reconnect setiap siklus cek */
        if (WiFi.status() == WL_DISCONNECTED || WiFi.status() == WL_IDLE_STATUS) {
            WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
        }
    }
}

/* ============================================================
   FUNGSI: KIRIM NOTIF ESP32 ON
   ============================================================ */
void kirimNotifOn() {
    char buf[350];
    snprintf(buf, sizeof(buf),
        "============================================\n"
        "STATUS: ESP32 ON\n"
        "============================================\n"
        "Sistem Deteksi Chainsaw menyala!\n\n"
        "WiFi     : %s\n"
        "IP       : %s\n"
        "Sensor   : INMP441 (GPIO %d)\n"
        "Uptime   : %lu second\n\n"
        "Perintah: /on /off /history\n"
        "============================================",
        WIFI_SSID,
        WiFi.localIP().toString().c_str(),
        I2S_SD_PIN,
        millis() / 1000UL
    );

    if (kirimTelegram(String(buf))) {
        notifOnTerkirim = true;
    }
}

/* ============================================================
   FUNGSI: KIRIM NOTIF ESP32 OFF
   Dipanggil dari WiFi event saat koneksi putus mendadak.
   Catatan: saat WiFi sudah putus, pesan ini mungkin tidak
   terkirim. Ini keterbatasan hardware - tidak ada koneksi
   maka tidak ada cara kirim pesan. Solusi: gunakan watchdog
   dari sisi Telegram (lihat komentar di bawah).
   ============================================================ */
void kirimNotifOff() {
    char buf[200];
    snprintf(buf, sizeof(buf),
        "============================================\n"
        "STATUS: ESP32 OFF / TIDAK TERHUBUNG\n"
        "============================================\n"
        "ESP32 kehilangan koneksi WiFi!\n\n"
        "SSID    : %s\n"
        "Uptime  : %lu detik\n\n"
        "Perangkat akan coba reconnect otomatis.\n"
        "============================================",
        WIFI_SSID,
        millis() / 1000UL
    );
    /* Coba kirim meski mungkin gagal karena WiFi putus */
    kirimTelegram(String(buf));
}

/* ============================================================
   FUNGSI: PROSES PERINTAH DARI MATLAB
   ============================================================ */
void prosesPerintah(String cmd) {

    if (cmd.equals("START")) {
        i2s_zero_dma_buffer(I2S_PORT);   /* flush buffer stale */
        streamingAktif = true;
        digitalWrite(LED_PIN, HIGH);
        /* JANGAN Serial.print/println di sini - akan masuk stream PCM */

    } else if (cmd.equals("STOP")) {
        streamingAktif = false;
        digitalWrite(LED_PIN, LOW);
        Serial.println("[ESP32] STOP OK");   /* aman karena streaming sudah off */

    } else if (cmd.startsWith("ALERT:")) {
        String probStr = cmd.substring(6);
        alertProbCS    = probStr.toFloat();
        alertDiterima  = true;
    }
}

/* ============================================================
   FUNGSI: BACA I2S DAN KIRIM PAKET KE MATLAB
   Format: [0xAA][0x55][LEN_H][LEN_L][PCM int16...]
   ============================================================ */
void bacaDanKirimAudio() {
    size_t bytesRead = 0;

    esp_err_t result = i2s_read(
        I2S_PORT,
        (void*)i2sBuffer,
        PACKET_BYTES,
        &bytesRead,
        pdMS_TO_TICKS(10)   /* timeout 10ms agar loop() tetap responsif */
    );

    if (result != ESP_OK || bytesRead == 0) return;

    uint16_t dataLen = (uint16_t)bytesRead;

    packetBuf[0] = 0xAA;
    packetBuf[1] = 0x55;
    packetBuf[2] = (uint8_t)(dataLen >> 8);
    packetBuf[3] = (uint8_t)(dataLen & 0xFF);
    memcpy(packetBuf + 4, (uint8_t*)i2sBuffer, dataLen);

    Serial.write(packetBuf, (size_t)(dataLen + 4));
}

/* ============================================================
   FUNGSI: PROSES ALERT CHAINSAW
   Selain kirim Telegram, sekarang juga menyimpan riwayat
   ke Google Sheets: {pesan, hari, tanggal, waktu}
   ============================================================ */
void prosesAlert(float probCS) {

    unsigned long now = millis();
    if ((now - lastTelegramMs) >= TELEGRAM_COOLDOWN_MS) {
        char buf[350];
        snprintf(buf, sizeof(buf),
            "=====================================\n"
            "DANGER - CHAINSAW DETECTED!!\n"
            "=====================================\n"
            "Illegal Logging Detected!\n\n"
            "Probability : %.1f%%\n"
            "Sensor    : ESP32 + INMP441\n"
            "Uptime    : %lu second\n\n"
            "Please inspect the area immediately!\n"
            "======================================",
            probCS,
            millis() / 1000UL
        );
        if (kirimTelegram(String(buf))) {
            lastTelegramMs = now;
        }

        /* Simpan ke Google Sheets terlepas dari status kirim Telegram,
           supaya riwayat tetap tercatat walau Telegram gagal terkirim */
        char pesanRingkas[100];
        snprintf(pesanRingkas, sizeof(pesanRingkas),
            "DANGER - Chainsaw terdeteksi (%.1f%%)", probCS);
        simpanRiwayatSheets(String(pesanRingkas));
    }
}

/* ============================================================
   FUNGSI: KIRIM PESAN KE TELEGRAM
   ============================================================ */
bool kirimTelegram(String pesan) {
    if (WiFi.status() != WL_CONNECTED) return false;
    if (bot == nullptr) {
        telegramClient.setInsecure();
        bot = new UniversalTelegramBot(BOT_TOKEN, telegramClient);
    }
    return bot->sendMessage(CHAT_ID, pesan, "");
}

/* ============================================================
   [BARU v3] FUNGSI: SIMPAN SATU RIWAYAT DETEKSI KE GOOGLE SHEETS
   Kirim POST JSON {token, pesan} ke Apps Script Web App.
   Hari/tanggal/waktu DIHITUNG DI SISI SERVER (Apps Script),
   bukan di ESP32 - jadi tidak perlu RTC / NTP di firmware ini.
   ============================================================ */
bool simpanRiwayatSheets(String pesan) {
    if (WiFi.status() != WL_CONNECTED) return false;

    WiFiClientSecure client;
    client.setInsecure();   /* skip verifikasi sertifikat (cukup utk hobby project) */

    HTTPClient http;
    if (!http.begin(client, GOOGLE_SCRIPT_URL)) {
        Serial.println("[Sheets] http.begin() gagal.");
        return false;
    }
    /* Apps Script Web App merespons dengan redirect (302) sebelum
       memberi jawaban akhir - WAJIB diikuti agar dapat respons 200 */
    http.setFollowRedirects(HTTPC_FORCE_FOLLOW_REDIRECTS);
    http.addHeader("Content-Type", "application/json");

    StaticJsonDocument<256> doc;
    doc["token"] = GOOGLE_SCRIPT_TOKEN;
    doc["pesan"] = pesan;
    String body;
    serializeJson(doc, body);

    int httpCode = http.POST(body);
    bool ok = (httpCode == 200);
    if (!ok) {
        Serial.print("[Sheets] Gagal simpan riwayat, HTTP code: ");
        Serial.println(httpCode);
    }
    http.end();
    return ok;
}

/* ============================================================
   [BARU v3] FUNGSI: KIRIM RIWAYAT (TEKS) KE TELEGRAM
   GET ke Apps Script dengan mode=text, hasil JSON diparse lalu
   diformat ulang jadi pesan Telegram yang rapi.
   ============================================================ */
void kirimRiwayatTeks(int jumlah) {
    if (WiFi.status() != WL_CONNECTED) {
        kirimTelegram("WiFi tidak terhubung, riwayat tidak tersedia saat ini.");
        return;
    }

    WiFiClientSecure client;
    client.setInsecure();
    HTTPClient http;

    String url = String(GOOGLE_SCRIPT_URL) + "?token=" + GOOGLE_SCRIPT_TOKEN +
                 "&mode=text&jumlah=" + String(jumlah);

    if (!http.begin(client, url)) {
        kirimTelegram("Gagal menghubungi Google Sheets.");
        return;
    }
    http.setFollowRedirects(HTTPC_FORCE_FOLLOW_REDIRECTS);

    int httpCode = http.GET();
    if (httpCode != 200) {
        kirimTelegram("Gagal mengambil riwayat dari Google Sheets (HTTP " + String(httpCode) + ").");
        http.end();
        return;
    }

    String payload = http.getString();
    http.end();

    DynamicJsonDocument doc(4096);
    DeserializationError err = deserializeJson(doc, payload);
    if (err) {
        kirimTelegram("Gagal membaca data riwayat (format respons tidak valid).");
        return;
    }
    if (doc["status"] != "ok") {
        kirimTelegram("Google Sheets menolak permintaan (cek SECRET_TOKEN).");
        return;
    }

    JsonArray data = doc["data"].as<JsonArray>();
    String pesanGabungan = "RIWAYAT DETEKSI CHAINSAW\n";
    pesanGabungan += "========================================\n";

    int nomor = 1;
    for (JsonObject item : data) {
        String p = item["pesan"]   | "-";
        String h = item["hari"]    | "-";
        String t = item["tanggal"] | "-";
        String w = item["waktu"]   | "-";
        pesanGabungan += String(nomor) + ". " + p + "\n   " + h + ", " + t + " " + w + "\n";
        nomor++;
    }
    if (nomor == 1) {
        pesanGabungan += "(Belum ada data riwayat)\n";
    }

    kirimTelegram(pesanGabungan);
}

/* ============================================================
   [BARU v3] CALLBACK STREAMING UNTUK sendMultipartFormDataToTelegram()
   ============================================================ */
bool csvMoreDataAvailable() {
    return riwayatCSVIndex < (size_t)riwayatCSVBuffer.length();
}
uint8_t csvGetNextByte() {
    return (uint8_t)riwayatCSVBuffer[riwayatCSVIndex++];
}

/* ============================================================
   [BARU v3] FUNGSI: KIRIM RIWAYAT (FILE .CSV) KE TELEGRAM
   GET ke Apps Script dengan mode=csv - Apps Script yang menyusun
   teks CSV-nya langsung, ESP32 tinggal meneruskan apa adanya
   sebagai dokumen Telegram (tidak perlu parsing JSON di sini).
   ============================================================ */
void kirimRiwayatFile(int jumlah) {
    if (WiFi.status() != WL_CONNECTED || bot == nullptr) {
        Serial.println("[CSV] Batal: WiFi belum terhubung atau bot belum siap.");
        return;
    }

    WiFiClientSecure client;
    client.setInsecure();
    HTTPClient http;

    String url = String(GOOGLE_SCRIPT_URL) + "?token=" + GOOGLE_SCRIPT_TOKEN +
                 "&mode=csv&jumlah=" + String(jumlah);

    Serial.print("[CSV] GET -> ");
    Serial.println(url);

    if (!http.begin(client, url)) {
        Serial.println("[CSV] http.begin() gagal.");
        return;
    }
    http.setFollowRedirects(HTTPC_FORCE_FOLLOW_REDIRECTS);

    int httpCode = http.GET();
    Serial.print("[CSV] HTTP code: ");
    Serial.println(httpCode);

    if (httpCode != 200) {
        Serial.println("[CSV] Gagal ambil CSV dari Apps Script, batal kirim file.");
        http.end();
        return;   /* pesan teks sudah terkirim di kirimRiwayatTeks(), file dilewati saja */
    }

    riwayatCSVBuffer = http.getString();
    http.end();

    Serial.print("[CSV] Panjang data CSV diterima: ");
    Serial.print(riwayatCSVBuffer.length());
    Serial.println(" byte");
    Serial.print("[CSV] Free heap sebelum upload: ");
    Serial.println(ESP.getFreeHeap());

    if (riwayatCSVBuffer.length() == 0) {
        Serial.println("[CSV] Data CSV kosong, tidak ada file dikirim.");
        return;   /* tidak ada data, tidak perlu kirim file kosong */
    }

    riwayatCSVIndex = 0;
    /* [FIX] sendDocumentByBinary() tidak tersedia di semua versi library
       UniversalTelegramBot (hanya ada di beberapa fork). Fungsi generik
       sendMultipartFormDataToTelegram() di bawah ini SELALU ada di semua
       versi, jadi lebih aman dipakai langsung. GetNextBuffer/GetNextBufferLen
       diisi nullptr karena kita kirim data byte-per-byte, bukan per-blok. */
    String responsTelegram = bot->sendMultipartFormDataToTelegram(
        "sendDocument", "document", "riwayat_deteksi.csv", "text/csv",
        CHAT_ID, riwayatCSVBuffer.length(),
        csvMoreDataAvailable, csvGetNextByte, nullptr, nullptr);

    /* [FIX] Fungsi ini mengembalikan String berisi respons JSON mentah
       dari Telegram (BUKAN bool) - dicetak apa adanya di sini supaya
       kalau gagal, pesan error ASLI dari Telegram langsung kelihatan
       di Serial Monitor (mis. "Bad Request: ..."). */
    Serial.print("[CSV] Respons Telegram: ");
    Serial.println(responsTelegram);
}

/* ============================================================
   FUNGSI: SETUP I2S INMP441
   ============================================================ */
void setupI2S() {
    i2s_config_t cfg = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate          = SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = DMA_BUF_COUNT,
        .dma_buf_len          = DMA_BUF_LEN,
        .use_apll             = true,
        .tx_desc_auto_clear   = false,
        .fixed_mclk           = 0
    };

    i2s_pin_config_t pins = {
        .bck_io_num   = I2S_SCK_PIN,
        .ws_io_num    = I2S_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = I2S_SD_PIN
    };

    i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
    i2s_set_pin(I2S_PORT, &pins);
    i2s_zero_dma_buffer(I2S_PORT);
}
