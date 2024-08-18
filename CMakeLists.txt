cmake_minimum_required(VERSION 3.29.6)
project(Vulkan C)

# Set the C standard to C99
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

# Set the path to GLFW (Ensure the correct path based on your GLFW installation)
set(GLFW_DIR "C:/glfw-3.4.bin.WIN64")

# Include directories for GLFW
include_directories(${GLFW_DIR}/include)

# Link directories for GLFW (Ensure you're using the correct MSVC-compatible library)
link_directories(${GLFW_DIR}/lib-vc2022)

# Set the path to Vulkan SDK (Update this to your Vulkan SDK path)
set(VULKAN_SDK "C:/VulkanSDK/1.3.290.0")

# Include directories for Vulkan
include_directories(${VULKAN_SDK}/Include)

# Link directories for Vulkan
link_directories(${VULKAN_SDK}/Lib)

# Set the path to Assimp (Ensure these paths are correct for Assimp built with MSVC)
set(ASSIMP_DIR "C:/assimp")
include_directories(${ASSIMP_DIR}/include)
include_directories(${ASSIMP_DIR}/build/include)
#include_directories(${ASSIMP_DIR}/build/bin/Release)
include_directories(${ASSIMP_DIR}/build/lib/Release)

link_directories(${ASSIMP_DIR}/build/lib/Release) # Ensure this path contains assimp.lib
#link_directories(${ASSIMP_DIR}/build/bin/Release) # Ensure this path contains assimp.lib


# Add the executable
add_executable(Vulkan main.c)

# Define the relative paths to the shader files
set(SHADERS_PATH ${CMAKE_SOURCE_DIR}/shaders)

# Define the relative paths to the texture files
set(TEXTURES_PATH ${CMAKE_SOURCE_DIR}/textures)

# Define the relative paths to the model files
set(MODELS_PATH ${CMAKE_SOURCE_DIR}/models)

# Pass the directories paths as preprocessor macros
add_definitions(-DSHADERS_PATH=\"${SHADERS_PATH}\")
add_definitions(-DTEXTURES_PATH=\"${TEXTURES_PATH}\")
add_definitions(-DMODELS_PATH=\"${MODELS_PATH}\")

# Specify the required libraries for linking
target_link_libraries(Vulkan PRIVATE
        glfw3.lib        # Ensure you're using .lib files for MSVC
        vulkan-1.lib     # Ensure you're using .lib files for MSVC
        assimp-vc143-mt.lib  # Assimp library name might differ based on your build (ensure you have this .lib)
)

# Include stb_image.h if needed (Ensure the path to stb_image.h is correct)
target_include_directories(Vulkan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Set optimization and debug flags specifically for MSVC
if (MSVC)
    target_compile_options(Vulkan PRIVATE /W4) # Warning level 4 for MSVC
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(Vulkan PRIVATE /Od /RTC1)  # Disable optimizations, enable runtime checks for debug
    else()
        target_compile_options(Vulkan PRIVATE /O2)  # Optimize for speed in release builds
    endif()
else()
    target_compile_options(Vulkan PRIVATE -O2)  # Standard optimization for non-MSVC compilers
endif()
