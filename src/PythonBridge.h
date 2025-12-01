#ifndef VOICE_CHANGER_PYTHONBRIDGE_H
#define VOICE_CHANGER_PYTHONBRIDGE_H

#include <vector>
#include <string>
#include <sys/socket.h> // // Linux socket headers.
#include <arpa/inet.h>
#include <unistd.h>
#include <netinet/in.h>
#include <cstdint>

class PythonBridge {
private:
   int sock = -1;
   struct sockaddr_in serv_addr{};
   bool connected = false;

public:
   PythonBridge();
   ~PythonBridge();

   bool Connect();
   void Disconnect();

   // // Function to send audio and pitch data to Python and get processed audio back.
   std::vector<float> ProcessAudio(const std::vector<float>& input_audio, int pitch_semitones);
};

#endif //VOICE_CHANGER_PYTHONBRIDGE_H