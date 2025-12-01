# // Sets the minimum version of CMake required to build this project.
cmake_minimum_required(VERSION 3.20)

# // Names the project "voice-changer".
project(voice-changer)

# // Sets the C++ standard to 17 (Modern C++ features).
set(CMAKE_CXX_STANDARD 17)

# // --- FIND DEPENDENCIES ---
# // Search for the 'miniaudio.h' file path (for audio I/O).
find_path(MINIAUDIO_INCLUDE_DIRS "miniaudio.h" REQUIRED)
# // Search for the 'pkg-config' tool (helper to find system libraries on Linux).
find_package(PkgConfig REQUIRED)
# // Use pkg-config to find the 'rnnoise' library (AI Noise Suppression).
pkg_check_modules(RNNOISE REQUIRED rnnoise)

# // --- DEFINE APP ---
# // Creates the executable 'voice_changer_app' from these specific source files.
add_executable(voice_changer_app
        src/main.cpp
        src/PythonBridge.cpp
        src/PythonBridge.h
        src/Denoise.h
        src/Utils.h
)

# // --- INCLUDE DIRECTORIES ---
# // Tells the compiler where to look for header (.h) files.
target_include_directories(voice_changer_app PRIVATE ${MINIAUDIO_INCLUDE_DIRS})
target_include_directories(voice_changer_app PRIVATE ${RNNOISE_INCLUDE_DIRS})
target_include_directories(voice_changer_app PRIVATE src)

# // --- LINK LIBRARIES ---
# // Tells the linker to attach the compiled RNNoise library to our app.
target_link_libraries(voice_changer_app PRIVATE ${RNNOISE_LIBRARIES})