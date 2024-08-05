cmake_minimum_required(VERSION 3.28)
project(Vulkan C)

set(CMAKE_C_STANDARD 23)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED True)




# Add the executable
add_executable(Vulkan main.c)

# Define the relative paths to the shader files
set(SHADERS_PATH ${CMAKE_SOURCE_DIR}/shaders)

# Define the relative paths to the texture files
set(TEXTURES_PATH ${CMAKE_SOURCE_DIR}/textures)

# Define the relative paths to the model files
set(MODELS_PATH ${CMAKE_SOURCE_DIR}/models)


# Pass the directors paths as preprocessor macros
add_definitions(-DSHADERS_PATH=\"${SHADERS_PATH}\")
add_definitions(-DTEXTURES_PATH=\"${TEXTURES_PATH}\")
add_definitions(-DMODELS_PATH=\"${MODELS_PATH}\")

# Specify the required libraries for linking
target_link_libraries(Vulkan PRIVATE
        glfw
        vulkan
        dl
        pthread
        X11
        Xxf86vm
        Xrandr
        Xi
        m    )

target_include_directories(Vulkan PRIVATE ${ASSIMP_INCLUDE_DIRS})

# include stb_image.h
target_include_directories(Vulkan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Set optimization flags
target_compile_options(Vulkan PRIVATE -O2)
