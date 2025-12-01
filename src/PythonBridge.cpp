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

    std::cout << "[BRIDGE] Connected to Python AI backend!" << std::endl;
    connected = true;
    return true;
}

std::vector<float> PythonBridge::ProcessAudio(const std::vector<float>& input, int pitch_semitones) {
    if (!connected) return {};

    // // 1. Send Header (8 bytes: Size + Pitch)
    struct { int32_t size; int32_t pitch; } header;
    header.size = static_cast<int32_t>(input.size());
    header.pitch = static_cast<int32_t>(pitch_semitones);

    if (send(sock, &header, sizeof(header), 0) < 0) {
        Disconnect(); return {};
    }

    // // 2. Send Audio
    if (send(sock, input.data(), input.size() * sizeof(float), 0) < 0) {
        Disconnect(); return {};
    }

    // // 3. Receive Size
    int32_t resp_size = 0;
    if (read(sock, &resp_size, 4) <= 0) {
        Disconnect(); return {};
    }

    if (resp_size == 0) return {};

    // // 4. Receive Audio
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