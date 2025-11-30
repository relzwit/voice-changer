cmake_minimum_required(VERSION 3.20)
project(voice-changer) # // Defines the name of the project.

set(CMAKE_CXX_STANDARD 17) # // Sets the C++ standard to 17 (standard for modern C++ projects).

# 1. FIND DEPENDENCIES
find_path(MINIAUDIO_INCLUDE_DIRS "miniaudio.h" REQUIRED) # // Finds the miniaudio header file.
find_package(PkgConfig REQUIRED) # // Finds pkg-config for library discovery (standard Linux tool).
pkg_check_modules(RNNOISE REQUIRED rnnoise) # // Checks if the RNNoise library is installed.

# 2. DEFINE APP
# // Defines the final executable and lists every source file that needs to be compiled.
add_executable(voice_changer_app
        src/main.cpp
        src/PythonBridge.cpp
        src/PythonBridge.h
        src/Denoise.h
        src/Utils.h
)

# 3. INCLUDES (Tells the compiler WHERE to look for header files)
target_include_directories(voice_changer_app PRIVATE ${MINIAUDIO_INCLUDE_DIRS})
target_include_directories(voice_changer_app PRIVATE ${RNNOISE_INCLUDE_DIRS})
target_include_directories(voice_changer_app PRIVATE src)

# 4. LINK LIBRARIES (Tells the linker WHAT compiled libraries to use)
# // We link against RNNoise (the cleaner) and the default C++ libraries.
target_link_libraries(voice_changer_app PRIVATE ${RNNOISE_LIBRARIES})