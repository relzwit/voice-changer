#ifndef VOICE_CHANGER_PYTHONBRIDGE_H
#define VOICE_CHANGER_PYTHONBRIDGE_H

#include <vector>
#include <string>
#include <sys/socket.h>     // // Core networking functions.
#include <arpa/inet.h>      // // IP address handling.
#include <unistd.h>         // // Standard Unix utilities (close socket).

class PythonBridge {
private:
    int sock = -1;                     // // Socket file descriptor ID.
    struct sockaddr_in serv_addr;      // // Server address structure.
    bool connected = false;

public:
    PythonBridge();
    ~PythonBridge();

    bool Connect();

    // // Sends 16kHz audio data and pitch command to Python.
    std::vector<float> ProcessAudio(const std::vector<float>& input_audio, int pitch_semitones);
};

#endif //VOICE_CHANGER_PYTHONBRIDGE_H