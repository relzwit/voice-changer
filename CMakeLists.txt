# --- CMAKE CONFIGURATION ---

# We require CMake 3.20 or newer.
# WHY: This version introduced better support for 'presets' and improved PkgConfig handling.
# It is a safe baseline for modern C++ projects on Linux and Windows.
cmake_minimum_required(VERSION 3.20)

# The name of our solution.
project(voice-changer)

# --- COMPILER SETTINGS ---

# We force the compiler to use the C++17 Standard.
# WHY: Your code uses modern features like std::atomic, std::thread, and std::future.
# While C++11 introduced these, C++17 fixes many edge cases and is the current
# industry standard for new "Modern C++" projects.
set(CMAKE_CXX_STANDARD 17)

# --- DEPENDENCY: MINIAUDIO ---

# Miniaudio is a "Header-Only" library. It doesn't have a compiled .lib file.
# We just need to find where the file "miniaudio.h" lives on your hard drive.
# REQUIRED means: "Stop the build immediately if this file is missing."
find_path(MINIAUDIO_INCLUDE_DIRS "miniaudio.h" REQUIRED)

# --- DEPENDENCY: RNNOISE ---

# RNNoise is a compiled C library. It is often installed via package managers (apt, pacman, vcpkg).
# Standard CMake 'find_package' often fails for C libraries.
# WHY: We use PkgConfig (Package Config) instead. It reads the .pc files provided
# by Linux package managers to find the exact include paths and library files.
find_package(PkgConfig REQUIRED)

# This command asks PkgConfig: "Do you know where 'rnnoise' is?"
# It creates variables like ${RNNOISE_INCLUDE_DIRS} and ${RNNOISE_LIBRARIES}.
pkg_check_modules(RNNOISE REQUIRED rnnoise)

# --- EXECUTABLE DEFINITION ---

# Here we list every single file that belongs to our project.
# WHY: Listing headers (.h) here isn't strictly necessary for the compiler,
# but it is REQUIRED for IDEs (like CLion or VS Code) to display them in the project tree.
add_executable(voice_changer_app
        src/main.cpp
        src/PythonBridge.cpp
        src/PythonBridge.h
#        src/Denoise.h
        src/Utils.h
)

# --- INCLUDES ---

# This tells the compiler: "When you see #include <miniaudio.h>, look in these folders."
# PRIVATE means: "Only use these paths for compiling THIS executable, don't share them."
target_include_directories(voice_changer_app PRIVATE
        ${MINIAUDIO_INCLUDE_DIRS}
        ${RNNOISE_INCLUDE_DIRS}
        src  # Allows us to write #include "Utils.h" instead of #include "src/Utils.h"
)

# --- LINKING ---

# This tells the Linker: "Bind the compiled RNNoise binary code into our executable."
# If we skip this, the code will compile, but fail at the last second with "Undefined Reference" errors.
# Note: We don't link Miniaudio because it is header-only (compiled inside main.cpp).
target_link_libraries(voice_changer_app PRIVATE ${RNNOISE_LIBRARIES})