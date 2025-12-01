// /src/PythonBridge.h

// --- HEADER GUARDS ---
// These lines prevent this file from being pasted into your program twice.
// If you include "PythonBridge.h" in both main.cpp and another file,
// the compiler would error out saying "PythonBridge is defined twice" without this.
#ifndef VOICE_CHANGER_PYTHONBRIDGE_H
#define VOICE_CHANGER_PYTHONBRIDGE_H

#include <vector>        // Needed for std::vector (the list of audio samples).
#include <netinet/in.h>  // Needed for "struct sockaddr_in" (Linux networking structures).

// This class handles the TCP Client.
// It acts as a "Bridge" connecting our fast C++ audio engine
// to the smart (but slower) Python AI backend.
class PythonBridge {

// --- PRIVATE MEMBERS ---
// These variables are hidden. 'main.cpp' cannot see or touch them.
// This is "Encapsulation". We don't want the main loop accidentally
// messing with the raw socket ID.
private:
   // In Linux/Unix, a Socket is just a file.
   // This integer is the "File Descriptor" (ID) of that open connection.
   // We initialize it to -1 because 0 is a valid ID (stdin). -1 means "Disconnected".
   int sock = -1;

   // This is a C-style structure defined in <netinet/in.h>.
   // It holds the IP Address (127.0.0.1) and the Port (51235).
   // We store it here so we don't have to rebuild it every time we reconnect.
   struct sockaddr_in serv_addr;

   // A simple flag to track state.
   // We check this before trying to send data so we don't crash
   // by writing to a closed socket.
   bool connected = false;

// --- PUBLIC INTERFACE ---
// These are the buttons main.cpp is allowed to push.
public:
   // Constructor: Sets up the empty object.
   PythonBridge();

   // Destructor: Automatically calls Disconnect() when the program closes.
   // Ensures we don't leave "zombie" connections open in the OS.
   ~PythonBridge();

   // Tries to dial the Python server.
   // Returns 'true' if the handshake succeeded, 'false' if Python is offline.
   bool Connect();

   // Hangs up the phone and frees the system resources.
   void Disconnect();

   // The core function:
   // 1. Takes raw audio (input).
   // 2. Sends it to Python.
   // 3. Waits for Python to think.
   // 4. Returns the modified audio.
   std::vector<float> ProcessAudio(const std::vector<float>& input, int pitch);
};

#endif