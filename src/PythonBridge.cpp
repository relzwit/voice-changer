#include "PythonBridge.h"
#include <iostream>
#include <cstring>
#include <vector>

#define PORT 5555 // // The dedicated communication port.

PythonBridge::PythonBridge() {}

PythonBridge::~PythonBridge() {
    if (connected) close(sock);
}

bool PythonBridge::Connect() {
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cerr << "[BRIDGE] Socket creation error" << std::endl;
        return false;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        std::cerr << "[BRIDGE] Invalid address" << std::endl;
        return false;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        return false;
    }

    std::cout << "[BRIDGE] Connected to Python Brain!" << std::endl;
    connected = true;
    return true;
}

std::vector<float> PythonBridge::ProcessAudio(const std::vector<float>& input, int pitch_semitones) {
    if (!connected) return {};

    // --- 1. PREPARE HEADER ---
    struct {
        int32_t sample_count;
        int32_t pitch;
    } header;

    header.sample_count = (int32_t)input.size();
    header.pitch = (int32_t)pitch_semitones;

    // --- 2. SEND HEADER (8 Bytes) ---
    if (send(sock, &header, sizeof(header), 0) < 0) {
        std::cerr << "[BRIDGE] Send Header Failed" << std::endl;
        close(sock); connected = false; return {};
    }

    // --- 3. SEND AUDIO ---
    if (send(sock, input.data(), input.size() * sizeof(float), 0) < 0) {
        std::cerr << "[BRIDGE] Send Audio Failed" << std::endl;
        close(sock); connected = false; return {};
    }

    // --- 4. RECEIVE RESPONSE SIZE ---
    int32_t response_size = 0;
    int valread = read(sock, &response_size, 4);
    if (valread <= 0 || response_size <= 0) {
        std::cerr << "[BRIDGE] Read Size Failed/Empty Response" << std::endl;
        close(sock); connected = false; return {};
    }

    // --- 5. RECEIVE AUDIO DATA (Loop for large data) ---
    std::vector<float> output(response_size);
    size_t total_read = 0;
    size_t bytes_to_read = response_size * sizeof(float);
    char* ptr = (char*)output.data();

    while (total_read < bytes_to_read) {
        valread = read(sock, ptr + total_read, bytes_to_read - total_read);
        if (valread <= 0) break;
        total_read += valread;
    }

    return output;
}