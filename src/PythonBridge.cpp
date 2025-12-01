#include "PythonBridge.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 51235 // // Port must match server.py exactly.

PythonBridge::PythonBridge() {}

PythonBridge::~PythonBridge() {
    Disconnect(); // // Ensure we close the socket on destruction.
}

void PythonBridge::Disconnect() {
    if (connected) {
        std::cout << "[BRIDGE DEBUG] Disconnecting socket." << std::endl;
        close(sock);
        connected = false;
    }
}

bool PythonBridge::Connect() {
    // // Create a TCP socket.
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cerr << "[BRIDGE] Socket creation error" << std::endl;
        return false;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    // // Set localhost IP.
    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        std::cerr << "[BRIDGE] Invalid address" << std::endl;
        return false;
    }

    // // Attempt connection.
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        return false;
    }

    std::cout << "[BRIDGE] Connected to Python AI backend!" << std::endl;
    connected = true;
    return true;
}

std::vector<float> PythonBridge::ProcessAudio(const std::vector<float>& input, int pitch_semitones) {
    if (!connected) {
        std::cerr << "[BRIDGE] Not connected to server." << std::endl;
        return {};
    }

    // --- 1. PREPARE HEADER ---
    // // Create a struct to hold the metadata.
    struct {
        int32_t sample_count;
        int32_t pitch;
    } header;

    header.sample_count = static_cast<int32_t>(input.size());
    header.pitch = static_cast<int32_t>(pitch_semitones);

    // --- 2. SEND HEADER ---
    // // Send the 8-byte header first.
    if (send(sock, &header, sizeof(header), 0) < 0) {
        std::cerr << "[BRIDGE] Send Header Failed" << std::endl;
        Disconnect();
        return {};
    }

    // --- 3. SEND AUDIO ---
    // // Send the raw float array.
    if (send(sock, input.data(), input.size() * sizeof(float), 0) < 0) {
        std::cerr << "[BRIDGE] Send Audio Failed" << std::endl;
        Disconnect();
        return {};
    }

    // --- 4. RECEIVE SIZE ---
    // // Wait for Python to tell us how big the result is.
    int32_t response_size = 0;
    int valread = read(sock, &response_size, 4);
    if (valread <= 0) {
        std::cerr << "[BRIDGE] Read Size Failed" << std::endl;
        Disconnect();
        return {};
    }

    if (response_size == 0) return {}; // // Server sent empty (error).

    // --- 5. RECEIVE AUDIO ---
    // // Create a vector to hold the result.
    std::vector<float> output(response_size);
    size_t total_read = 0;
    size_t bytes_to_read = static_cast<size_t>(response_size) * sizeof(float);
    char* ptr = reinterpret_cast<char*>(output.data());

    // // Loop to ensure we get every single byte.
    while (total_read < bytes_to_read) {
        int r = read(sock, ptr + total_read, bytes_to_read - total_read);
        if (r <= 0) break;
        total_read += static_cast<size_t>(r);
    }

    return output;
}