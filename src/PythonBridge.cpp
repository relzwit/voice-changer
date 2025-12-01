// /src/PythonBridge.cpp

#include "PythonBridge.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 51235

PythonBridge::PythonBridge() {}
PythonBridge::~PythonBridge() { Disconnect(); }

void PythonBridge::Disconnect() {
    if (connected) {
        close(sock);
        connected = false;
    }
}

bool PythonBridge::Connect() {
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) return false;

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) return false;

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) return false;

    connected = true;
    return true;
}

std::vector<float> PythonBridge::ProcessAudio(const std::vector<float>& input, int pitch_semitones) {
    if (!connected) return {};

    // --- STEP 1: SEND HEADER (SAFE METHOD) ---
    // We send an array of 2 integers.
    // This avoids "Struct Padding" issues where C++ adds invisible bytes.
    int32_t header[2];
    header[0] = static_cast<int32_t>(input.size());
    header[1] = static_cast<int32_t>(pitch_semitones);

    // Send 8 bytes (2 ints * 4 bytes)
    if (send(sock, header, 2 * sizeof(int32_t), 0) < 0) {
        Disconnect(); return {};
    }

    // --- STEP 2: SEND AUDIO PAYLOAD ---
    if (send(sock, input.data(), input.size() * sizeof(float), 0) < 0) {
        Disconnect(); return {};
    }

    // --- STEP 3: WAIT FOR RESPONSE SIZE ---
    int32_t resp_size = 0;
    if (read(sock, &resp_size, 4) <= 0) {
        Disconnect(); return {};
    }

    if (resp_size == 0) return {};

    // --- STEP 4: RECEIVE AUDIO PAYLOAD ---
    std::vector<float> output(resp_size);
    size_t total = 0;
    size_t bytes = resp_size * sizeof(float);
    char* ptr = (char*)output.data();

    while (total < bytes) {
        int r = read(sock, ptr + total, bytes - total);
        if (r <= 0) break;
        total += r;
    }

    return output;
}