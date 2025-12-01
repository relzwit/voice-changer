#ifndef VOICE_CHANGER_PYTHONBRIDGE_H
#define VOICE_CHANGER_PYTHONBRIDGE_H

#include <vector>
#include <netinet/in.h>

class PythonBridge {
private:
   int sock = -1;
   struct sockaddr_in serv_addr;
   bool connected = false;

public:
   PythonBridge();
   ~PythonBridge();
   bool Connect();
   void Disconnect();
   std::vector<float> ProcessAudio(const std::vector<float>& input, int pitch);
};

#endif