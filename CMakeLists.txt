cmake_minimum_required(VERSION 3.20)
project(voice-changer)

set(CMAKE_CXX_STANDARD 17)

# 1. FIND DEPENDENCIES
find_path(MINIAUDIO_INCLUDE_DIRS "miniaudio.h" REQUIRED)
find_package(onnxruntime REQUIRED)

# Find the Noise Suppression library
# We use pkg-config to find it on Fedora reliably
find_package(PkgConfig REQUIRED)
pkg_check_modules(RNNOISE REQUIRED rnnoise)

# 2. DEFINE THE APP
add_executable(voice_changer_app
        src/main.cpp
        src/RVCEngine.cpp
        src/RVCEngine.h
        src/Utils.h
)

# 3. INCLUDE DIRECTORIES
target_include_directories(voice_changer_app PRIVATE ${MINIAUDIO_INCLUDE_DIRS})
target_include_directories(voice_changer_app PRIVATE ${RNNOISE_INCLUDE_DIRS})
target_include_directories(voice_changer_app PRIVATE src)

# 4. LINK LIBRARIES
target_link_libraries(voice_changer_app PRIVATE
        onnxruntime::onnxruntime
        ${RNNOISE_LIBRARIES}
)