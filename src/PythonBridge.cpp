#include "PythonBridge.h"
#include <iostream>
#include <cstring>
#include <vector>

// These are Linux/Unix specific headers for networking.
// If you were on Windows, you would need <winsock2.h> instead.
#include <unistd.h>     // For close()
#include <arpa/inet.h>  // For socket(), connect(), htons(), inet_pton()

// The specific "door" number we will knock on.
// This MUST match the port defined in 'server.py'.
#define PORT 51235

PythonBridge::PythonBridge() {}

// Destructor: This runs when the program closes or the object is deleted.
// We strictly ensure we hang up the phone (Disconnect) to free the OS port.
PythonBridge::~PythonBridge() { Disconnect(); }

void PythonBridge::Disconnect() {
    if (connected) {
        // 'close' is a System Call. It tells the Linux Kernel to destroy the socket connection.
        close(sock);
        connected = false;
    }
}

bool PythonBridge::Connect() {
    // 1. Create the Socket
    // AF_INET      = IPv4 (Standard Internet Protocol)
    // SOCK_STREAM  = TCP (Reliable, ordered data. If we sent UDP, audio might arrive out of order.)
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) return false;

    // 2. Prepare the Address Structure
    // We need to tell the OS where we want to connect.
    serv_addr.sin_family = AF_INET;

    // htons = "Host TO Network Short"
    // Computers store numbers differently (Little Endian vs Big Endian).
    // The Internet Standard is Big Endian. This function flips the bytes of '51235'
    // so the network understands it, regardless of what CPU you have.
    serv_addr.sin_port = htons(PORT);

    // inet_pton = "Pointer TO Number"
    // Converts the human string "127.0.0.1" (Localhost) into raw binary bytes.
    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) return false;

    // 3. Attempt Connection
    // This blocks (waits) until the Python server accepts us or refuses.
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) return false;

    std::cout << "[BRIDGE] Connected to Python AI backend!" << std::endl;
    connected = true;
    return true;
}

std::vector<float> PythonBridge::ProcessAudio(const std::vector<float>& input, int pitch_semitones) {
    // Fail-safe: Don't try to talk if the line is dead.
    if (!connected) return {};

    // --- STEP 1: SEND HEADER (METADATA) ---
    // We need to tell Python two things before sending the audio:
    // 1. How much audio is coming? (So it knows when to stop reading)
    // 2. What pitch shift do we want?

    // We create a temporary struct to hold this data tightly packed.
    struct { int32_t size; int32_t pitch; } header;

    // input.size() returns the number of floats.
    // 'static_cast' forces it to be a standard 32-bit integer for consistency.
    header.size = static_cast<int32_t>(input.size());
    header.pitch = static_cast<int32_t>(pitch_semitones);

    // Send the header struct as raw bytes.
    // sizeof(header) is 8 bytes (4 bytes for size + 4 bytes for pitch).
    if (send(sock, &header, sizeof(header), 0) < 0) {
        Disconnect(); return {}; // If send fails, the server probably crashed.
    }

    // --- STEP 2: SEND AUDIO PAYLOAD ---
    // We send the actual audio data.
    // input.data() gives us a direct pointer to the raw memory array inside the vector.
    // We calculate total bytes as: (Number of Floats) * (4 bytes per float).
    if (send(sock, input.data(), input.size() * sizeof(float), 0) < 0) {
        Disconnect(); return {};
    }

    // --- STEP 3: WAIT FOR RESPONSE SIZE ---
    // Now we wait. Python is processing...
    // We expect Python to send back an Integer (4 bytes) telling us how big the result is.
    int32_t resp_size = 0;

    // 'read' will block execution here until data arrives.
    if (read(sock, &resp_size, 4) <= 0) {
        Disconnect(); return {};
    }

    if (resp_size == 0) return {}; // Python sent empty audio (something went wrong logic-wise).

    // --- STEP 4: RECEIVE AUDIO PAYLOAD ---
    // This is the tricky part of TCP networking.
    // TCP is a "Stream", not a "Packet" service.
    // If we expect 100kb of data, the OS might give us 10kb, then 50kb, then 40kb.
    // We MUST loop until we have received exactly the amount we expect.

    std::vector<float> output(resp_size); // Allocate memory for the incoming audio.
    size_t total = 0;
    size_t bytes = resp_size * sizeof(float); // Total bytes we expect.
    char* ptr = (char*)output.data();         // Pointer to where we are writing.

    while (total < bytes) {
        // Read into the buffer, offset by how much we've already read.
        int r = read(sock, ptr + total, bytes - total);

        // If r <= 0, the connection broke mid-transfer.
        if (r <= 0) break;

        total += r; // Tally up what we got.
    }

    return output;
}