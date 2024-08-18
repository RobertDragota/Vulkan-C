
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "cglm/cglm.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdalign.h>
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

//#define max(a,b) ((a>b)?a:b)
/*Utility Header*/

char* readFile
 (
   const char* filename,
   size_t* size
 );

uint32_t clamp
(
   uint32_t value,
   uint32_t min,
   uint32_t max
);

/*Utility Header*/

/*Utility Implementation*/

char* readFile
(
    const char* filename,
    size_t* size
)
{
    FILE* file =NULL;
    fopen_s( &file, filename, "rb" );

    if ( file == NULL )
    {
        perror( "Failed to open file" );
        exit( EXIT_FAILURE );
    }

    fseek( file, 0, SEEK_END );
    *size = ftell( file );
    rewind( file );

    char* buffer = (char*) malloc( *size * sizeof( char ) );
    if ( buffer == NULL )
    {
        perror( "Failed to allocate memory" );
        fclose( file );
        exit( EXIT_FAILURE );
    }

    if( fread( buffer, sizeof( char ), *size, file ) < 0)
    {
        perror( "Failed to read file" );
        fclose( file );
        exit( EXIT_FAILURE );
    }
    fclose( file );

    return ( buffer );
}

uint32_t clamp
(
   uint32_t value,
   uint32_t min,
   uint32_t max
)
{
    if ( value < min )
    {
        return ( min );
    }
    else if ( value > max )
    {
        return ( max );
    }
    else
    {
        return ( value );
    }
}

/*Utility Implementation*/


/* Vulkan Utility Header*/

#ifdef NDEBUG
     const bool enableValidationLayers=false;
#else
     const bool enableValidationLayers=true;
#endif

#define SET_UNUSED (uint32_t)-1

  const uint32_t WIDTH=800;
  const uint32_t HEIGHT=600;

 uint32_t currentFrame=0;

 const char* MODEL_PATH=MODELS_PATH"/viking_room.obj";
 const char* TEXTURE_PATH=TEXTURES_PATH"/viking_room.png";

 const int MAX_FRAMES_IN_FLIGHT=2;

 const char * validationLayers[]=
{

    "VK_LAYER_KHRONOS_validation"

};

 const char* deviceExtensions[]=
{

    VK_KHR_SWAPCHAIN_EXTENSION_NAME

};

typedef struct Vertex_ {
   vec3 pos;
   vec3 color;
   vec2 texCoord;
} Vertex;

typedef struct Mesh_{

    Vertex* vertices_array;
    uint32_t* indices_arry;

    uint32_t vertices_count;
    uint32_t index_count;

}Mesh;

Mesh* meshes;

uint32_t mesh_size;

 const VkFormat candidates[]=
{

    VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT

};

size_t image_size;
static size_t vertices_size;
static size_t indices_size;


typedef struct QueueFamilyIndices_{
    uint32_t graphicsFamily;
    uint32_t presentFamily;
    bool is_graphic_family_found;
    bool is_present_family_found;

}QueueFamilyIndices;

typedef struct SwapChainSupportDetails_{
    VkSurfaceCapabilitiesKHR capabilities;
    VkSurfaceFormatKHR* formats;
    VkPresentModeKHR* presentModes;
}SwapChainSupportDetails;

typedef struct  UniformBufferObjects_{

    __declspec(align(16)) mat4 model;
    __declspec(align(16)) mat4 view;
    __declspec(align(16)) mat4 proj;

}UniformBufferObjects;

typedef struct  VulkObj_{

    GLFWwindow* window;
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapChain;
    VkImage* swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    VkImageView* swapChainImageViews;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkRenderPass renderPass;
    VkPipeline graphicsPipeline;
    VkFramebuffer* swapChainFramebuffers;
    VkCommandPool commandPool;
    VkCommandBuffer* commandBuffers;
    VkSemaphore* imageAvailableSemaphores;
    VkSemaphore* renderFinishedSemaphores;
    VkFence* inFlightFences;
    bool framebufferResized;
    Vertex* vertices;
    uint32_t* indices;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    VkBuffer* uniformBuffers;
    VkDeviceMemory* uniformBuffersMemory;
    void** uniformBuffersMapped;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet* descriptorSets;
    uint32_t mipLevels;
    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;
    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;
    VkSampleCountFlagBits msaaSmples;
    VkImage colorImage;
    VkDeviceMemory colorImageMemory;
    VkImageView colorImageView;

}VulkObj;

VulkObj Vulk= {.msaaSmples=VK_SAMPLE_COUNT_1_BIT};

bool isComplete
(
    QueueFamilyIndices *indices
);

double get_current_time_seconds
(
    void
);
bool isDeviceSuitable
(
    VkPhysicalDevice device
);
bool checkDeviceExensionSupport
(
    VkPhysicalDevice device
);

bool hasStencilComponent
(
    VkFormat format
);

bool checkValidationlayersSupport
(
    void
);

uint32_t findMemoryType
(
    uint32_t typeFilter,
    VkMemoryPropertyFlags properties
);




QueueFamilyIndices findQueueFamilies
(
    VkPhysicalDevice device
);

SwapChainSupportDetails querySwapChainSupport
(
    VkPhysicalDevice device,
    uint32_t* format_size,
    uint32_t* present_size
);

VkSurfaceFormatKHR chooseSwapSurfaceFormat
(
    const VkSurfaceFormatKHR* availableFormats,
    size_t size
);

VkExtent2D chooseSwapExtent
(
    const VkSurfaceCapabilitiesKHR* capabilities
);

VkFormat findDepthFormat
(
    void
);

VkFormat findSupportedFormat
(
    const VkFormat candidates[],
    size_t candidates_size,
    VkImageTiling tiling,
    VkFormatFeatureFlags features
);

VkCommandBuffer beginSingleTimeCommands
(
    void
);

void copyBufferToImage
(
    VkBuffer buffer,
    VkImage image,
    uint32_t width,
    uint32_t height
);

void endSingleTimeCommands
(
    VkCommandBuffer commandBuffer
);

void transitionImageLayout
(
    VkImage image,
    VkFormat format,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    uint32_t mipLevels
);



void updateUniformBuffer
(
    uint32_t currentImage
);

void copyBuffer
(
    VkBuffer srcBuffer,
    VkBuffer dstBuffer,
    VkDeviceSize size
);

void framebufferResizeCallback
(
    GLFWwindow* window,
    int width,
    int height
);



/* Vulkan Utility Header*/


/**Vulkan Core Header*/

VkVertexInputAttributeDescription* getAttributeDescriptions
(
    void
);

VkVertexInputBindingDescription getBindingDescription
(
    void
);




void mainLoop
(
    void
);

void initWindow
(
    void
);

 void initVulkan
 (
     void
 );

void drawFrame
(
    void
);

 void creatInstance
(
    void
);

 void pickPhysicalDevice
(
    void
);

 void createRenderPass
(
    void
);

 void createSwapChain
(
    void
);

 void recreateSwapChain
(
    void
);

 void createImageViews
(
    void
);

 void createFramebuffers
(
    void
);

 void createLogicalDevice
(
    void
);

 void createSurface
(
    void
);

 void createGraphicsPipeline
(
    void
);

 void createCommandPool
(
    void
);

 void createCommandBuffer
(
    void
);

 void createSyncObjects
(
    void
);

 void createVertexBuffer
(
    uint32_t index
);

 void createIndexBuffer
(
    uint32_t index
);

 void createDescriptorSetLayout
(
    void
);

 void createDescriptorPool
(
    void
);

 void createDescriptorSets
(
    void
);

 void createTextureImage
(
    void
);

 void createTextureSampler
(
    void
);

 void createDepthResources
(
    void
);

 void createUniformBuffers
(
    void
);

 void createTextureImageView
(
    void
);

 void cleanupSwapChain
(
    void
);

 void cleanup
(
    void
);

 VkShaderModule createShaderModule
(
    const char* code,
    size_t code_size
);

 void createBuffer
(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkBuffer* buffer, VkDeviceMemory* bufferMemory
);

 void createImage
(
    uint32_t width,
    uint32_t height,
    uint32_t mipLevels,
    VkSampleCountFlagBits numSamples,
    VkFormat format,
    VkImageTiling tiling,
    VkImageUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkImage* image,
    VkDeviceMemory* imageMemory
);

 void recordCommandBuffer
(
    VkCommandBuffer commandBuffer,
    uint32_t imageIndex
);

 VkImageView createImageView
(
  VkImage image,
  VkFormat format,
  VkImageAspectFlags aspectFlags,
  uint32_t mipLevels
);

 VkPresentModeKHR chooseSwapPresentMode
(
    const VkPresentModeKHR *availablePresentModes,
    size_t size
);

void loadModel
(
    const char* model_path
);

void processMesh
(
    const struct aiMesh* assimp_mesh,
    Mesh* mesh
);

void processNode
(
    const struct aiNode* node,
    const struct aiScene* scene,
    Mesh* mesh
);

void generateMipmaps
(
    VkImage image,
    VkFormat imageFormat,
    int32_t texWidth,
    int32_t texHeight,
    uint32_t mipLevels
);

VkSampleCountFlagBits getMaxUsableSampleCount
(
    void
);

void createColorResources
(
    void
);
/*Vulkan Core Header*/


/*Vulkan Utility Implementation*/
double get_current_time_seconds
(
    void
)
{
    return sin(-glfwGetTime())*sin(glfwGetTime());
}

bool isComplete
(
    QueueFamilyIndices *indices
)
{
    return ( indices->is_graphic_family_found && indices->is_present_family_found );
}


bool isDeviceSuitable
(
    VkPhysicalDevice device
)
{
    bool extensionsSupport = checkDeviceExensionSupport( device );
    QueueFamilyIndices indices = findQueueFamilies( device );
    bool swapChainAdequate = false;
    SwapChainSupportDetails swapChainSupport = {0};
    uint32_t format_size = 0;
    uint32_t present_size = 0;

    if ( extensionsSupport )
    {
        swapChainSupport = querySwapChainSupport( device, &format_size, &present_size );
        swapChainAdequate = ( format_size != 0 ) && ( present_size != 0 );
    }

    VkPhysicalDeviceFeatures supportedFeatures = {0};
    vkGetPhysicalDeviceFeatures( device, &supportedFeatures );

    return ( isComplete( &indices ) &&
           extensionsSupport &&
           swapChainAdequate &&
           supportedFeatures.samplerAnisotropy );
}


bool checkDeviceExensionSupport
(
    VkPhysicalDevice device
)
{
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties( device, NULL, &extensionCount, NULL );

    VkExtensionProperties* availableExtensions = NULL;
    availableExtensions = (VkExtensionProperties*)malloc( sizeof( VkExtensionProperties ) * extensionCount );

    if ( availableExtensions == NULL )
    {
        perror( "failed to allocate memory for availableExtensions!" );
        return ( false );
    }

    vkEnumerateDeviceExtensionProperties( device, NULL, &extensionCount, availableExtensions );

    size_t requiredExtension = (size_t)1;
    size_t remainingExtension = requiredExtension;

    for ( size_t i = 0; i < requiredExtension; ++i )
    {
        for ( size_t j = 0; j < extensionCount; ++j )
        {
            if ( strcmp( deviceExtensions[i], availableExtensions[j].extensionName ) == 0 )
            {
                remainingExtension--;
            }
        }
    }

    free( availableExtensions );
    return ( remainingExtension == 0 );
}


bool hasStencilComponent
(
    VkFormat format
)
{
    return ( format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT );
}


bool checkValidationlayersSupport
(
    void
)
{
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties( &layerCount, NULL );

    VkLayerProperties* availableLayers = NULL;
    availableLayers = (VkLayerProperties*)calloc( layerCount, sizeof( VkLayerProperties ) );

    if ( availableLayers == NULL )
    {
        perror( "failed to allocate memory for availableLayers!" );
        return ( false );
    }

    vkEnumerateInstanceLayerProperties( &layerCount, availableLayers );

    size_t validationLayers_size = (size_t)1;

    for ( size_t i = 0; i < validationLayers_size; ++i )
    {
        bool layerFound = false;

        for ( size_t j = 0; j < (size_t)layerCount; ++j )
        {
            if ( strcmp( validationLayers[i], availableLayers[j].layerName ) == 0 )
            {
                layerFound = true;
            }
        }

        if ( !layerFound )
        {
            free( availableLayers );
            return false;
        }
    }

    free( availableLayers );
    return ( true );
}

uint32_t findMemoryType
(
    uint32_t typeFilter,
    VkMemoryPropertyFlags properties
)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties( Vulk.physicalDevice, &memProperties );

    for ( size_t i = 0; i < memProperties.memoryTypeCount; ++i )
    {
        if ( ( typeFilter & (1 << i) ) &&
             ( ( memProperties.memoryTypes[i].propertyFlags & properties ) == properties ) )
        {
            return ( uint32_t )i;
        }
    }

    perror( "failed to find suitable memory type!" );
    exit( EXIT_FAILURE );
}

QueueFamilyIndices findQueueFamilies
(
    VkPhysicalDevice device
)
{
    QueueFamilyIndices indices = { SET_UNUSED };
    indices.is_graphic_family_found = false;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties( device, &queueFamilyCount, NULL );
    VkQueueFamilyProperties* queueFamilies = NULL;
    queueFamilies = ( VkQueueFamilyProperties* )malloc( sizeof( VkQueueFamilyProperties ) * queueFamilyCount );

    if ( queueFamilies == NULL )
    {
        perror( "failed to allocate memory for queueFamilies!" );
        exit( EXIT_FAILURE );
    }

    vkGetPhysicalDeviceQueueFamilyProperties( device, &queueFamilyCount, queueFamilies );

    for ( uint32_t j = 0; j < queueFamilyCount; ++j )
    {
        if ( ( queueFamilies[j].queueFlags & VK_QUEUE_GRAPHICS_BIT ) && !indices.is_graphic_family_found )
        {
            indices.graphicsFamily = j;
            indices.is_graphic_family_found = true;
        }

        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR( device, j, Vulk.surface, &presentSupport );
        if ( presentSupport )
        {
            indices.presentFamily = j;
            indices.is_present_family_found = true;
        }
    }

    free( queueFamilies );
    return ( indices );
}


SwapChainSupportDetails querySwapChainSupport
(
    VkPhysicalDevice device,
    uint32_t* format_size,
    uint32_t* present_size
)
{
    SwapChainSupportDetails details = {0};

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR( device, Vulk.surface, &details.capabilities );

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR( device, Vulk.surface, &formatCount, NULL );
    if ( formatCount != 0 )
    {
        details.formats = ( VkSurfaceFormatKHR* )calloc( formatCount, sizeof( VkSurfaceFormatKHR ) );
        vkGetPhysicalDeviceSurfaceFormatsKHR( device, Vulk.surface, &formatCount, details.formats );
    }

    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR( device, Vulk.surface, &presentModeCount, NULL );
    if ( presentModeCount != 0 )
    {
        details.presentModes = ( VkPresentModeKHR* )calloc( presentModeCount, sizeof( VkPresentModeKHR ) );
        vkGetPhysicalDeviceSurfacePresentModesKHR( device, Vulk.surface, &presentModeCount, details.presentModes );
    }

    *format_size = formatCount;
    *present_size = presentModeCount;
    return ( details );
}


VkSurfaceFormatKHR chooseSwapSurfaceFormat
(
    const VkSurfaceFormatKHR* availableFormats,
    size_t size
)
{
    for ( size_t i = 0; i < size; ++i )
    {
        if ( availableFormats[i].format == VK_FORMAT_B8G8R8A8_SRGB &&
             availableFormats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR )
        {
            return ( availableFormats[i] );
        }
    }

    return ( availableFormats[0] );
}


VkExtent2D chooseSwapExtent
(
    const VkSurfaceCapabilitiesKHR* capabilities
)
{
    if ( capabilities->currentExtent.width != 0xFFFFFFFF )
    {
        return capabilities->currentExtent;
    }
    else
    {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize( Vulk.window, &width, &height );

        VkExtent2D actualExtent =
        {
            (uint32_t)(width),
            (uint32_t)(height)
        };

        actualExtent.width = clamp( actualExtent.width, capabilities->minImageExtent.width, capabilities->maxImageExtent.width );
        actualExtent.height = clamp( actualExtent.height, capabilities->minImageExtent.height, capabilities->maxImageExtent.height );

        return ( actualExtent );
    }
}


VkFormat findDepthFormat
(
    void
)
{
    return ( findSupportedFormat
    (
        candidates,
        3,
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    ) );
}


VkFormat findSupportedFormat
(
    const VkFormat _candidates[],
    size_t candidates_size,
    VkImageTiling tiling,
    VkFormatFeatureFlags features
)
{
    for (size_t i = 0; i < candidates_size; ++i)
    {
        VkFormatProperties props = {0};
        vkGetPhysicalDeviceFormatProperties( Vulk.physicalDevice, _candidates[i], &props );

        if ( tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features )
        {
            return ( _candidates[i] );
        }
        else if ( tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features )
        {
            return ( _candidates[i] );
        }
    }

    perror("failed to find supported format!");
    exit( EXIT_FAILURE );
}


VkCommandBuffer beginSingleTimeCommands
(
    void
)
{
    VkCommandBufferAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = Vulk.commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = {0};
    vkAllocateCommandBuffers( Vulk.device, &allocInfo, &commandBuffer );

    VkCommandBufferBeginInfo beginInfo = {0};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer( commandBuffer, &beginInfo );

    return ( commandBuffer );
}


void copyBufferToImage
(
    VkBuffer buffer,
    VkImage image,
    uint32_t width,
    uint32_t height
)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferImageCopy region = {0};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset.x = 0;
    region.imageOffset.y = 0;
    region.imageOffset.z = 0;

    region.imageExtent.width = width;
    region.imageExtent.height = height;
    region.imageExtent.depth = 1;

    vkCmdCopyBufferToImage
    (
        commandBuffer,
        buffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region
    );

    endSingleTimeCommands( commandBuffer );
}


void endSingleTimeCommands
(
    VkCommandBuffer commandBuffer
)
{
    vkEndCommandBuffer( commandBuffer );

    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit( Vulk.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE );

    vkQueueWaitIdle( Vulk.graphicsQueue );

    vkFreeCommandBuffers( Vulk.device, Vulk.commandPool, 1, &commandBuffer );
}

void transitionImageLayout
(
    VkImage image,
    VkFormat format,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    uint32_t mipLevels
)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier = {0};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = 0;

    VkPipelineStageFlags sourceStage = {0};
    VkPipelineStageFlags destinationStage = {0};

    if ( oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL )
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if ( oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL )
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else
    {
        perror("unsupported layout transition!");
        exit( EXIT_FAILURE );
    }

    vkCmdPipelineBarrier
    (
        commandBuffer,
        sourceStage,
        destinationStage,
        0,
        0,
        NULL,
        0,
        NULL,
        1,
        &barrier
    );

    endSingleTimeCommands( commandBuffer );
}

 vec3 cameraPos = {2.0f, 2.0f, 2.0f}; // cameraPos
 vec3 cameraFront = {0.0f, -1.0f, -1.0f};
 vec3 cameraUp = {0.0f, 0.0f, 1.0f}; // cameraUp
 vec3 center = {0.0f, 0.0f, 0.0f};
 vec3 z_axis = {0.0f, 0.0f, 1.0f};
 vec3 y_axis = {0.0f, 1.0f, 0.0f};
 vec3 x_axis = {1.0f, 0.0f, 0.0f};
void updateUniformBuffer
(
    uint32_t currentImage
)
{
    static double startTime = 0.0;

    if (startTime == 0.0)
    {
        startTime = get_current_time_seconds();
    }

    double currentTime = get_current_time_seconds();
    float time = (float)(currentTime - startTime);

    static UniformBufferObjects ubo = {.model = GLM_MAT4_IDENTITY_INIT};
    //glm_mat4_identity( ubo.model );

    float cameraSpeed = 0.005f;
    vec3 tempVec;
    // Forward movement
    if (glfwGetKey(Vulk.window, GLFW_KEY_W) == GLFW_PRESS)
    {
        glm_vec3_scale(cameraFront, cameraSpeed, tempVec);
        glm_vec3_add(cameraPos, tempVec, cameraPos);
    }

    // Backward movement
    if (glfwGetKey(Vulk.window, GLFW_KEY_S) == GLFW_PRESS)
    {
        glm_vec3_scale(cameraFront, cameraSpeed, tempVec);
        glm_vec3_sub(cameraPos, tempVec, cameraPos);
    }

    // Left movement (Strafe left)
    if (glfwGetKey(Vulk.window, GLFW_KEY_A) == GLFW_PRESS)
    {
        glm_vec3_cross(cameraFront, cameraUp, tempVec); // Calculate the right vector
        glm_normalize(tempVec);
        glm_vec3_scale(tempVec, cameraSpeed, tempVec); // Scale it by the camera speed
        glm_vec3_sub(cameraPos, tempVec, cameraPos);   // Subtract it from the camera position
    }

    // Right movement (Strafe right)
    if (glfwGetKey(Vulk.window, GLFW_KEY_D) == GLFW_PRESS)
    {
        glm_vec3_cross(cameraFront, cameraUp, tempVec); // Calculate the right vector
        glm_normalize(tempVec);
        glm_vec3_scale(tempVec, cameraSpeed, tempVec); // Scale it by the camera speed
        glm_vec3_add(cameraPos, tempVec, cameraPos);   // Add it to the camera position
    }

    if ( glfwGetKey(Vulk.window , GLFW_KEY_Z) == GLFW_PRESS )
    {
        glm_rotate( ubo.model,  glm_rad(1.0f), z_axis );
    }
    if ( glfwGetKey(Vulk.window , GLFW_KEY_Y) == GLFW_PRESS )
    {
        glm_rotate( ubo.model,  glm_rad(1.0f), y_axis );
    }
    if ( glfwGetKey(Vulk.window , GLFW_KEY_X) == GLFW_PRESS )
    {
        glm_rotate( ubo.model,  glm_rad(1.0f), x_axis );
    }
    glm_vec3_add(cameraPos,cameraFront,center);
    glm_lookat( cameraPos, center, cameraUp, ubo.view );
    glm_perspective( 20.0f, Vulk.swapChainExtent.width / (float)Vulk.swapChainExtent.height, 0.1f, 10.0f, ubo.proj );

    ubo.proj[1][1] *= -1;

    memcpy( Vulk.uniformBuffersMapped[currentImage], &ubo, sizeof(ubo) );
}


void copyBuffer
(
    VkBuffer srcBuffer,
    VkBuffer dstBuffer,
    VkDeviceSize size
)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferCopy copyRegion = {0};
    copyRegion.size = size;

    vkCmdCopyBuffer( commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion );

    endSingleTimeCommands( commandBuffer );
}

void framebufferResizeCallback
(
    GLFWwindow* window,
    int width,
    int height
)
{

    VulkObj * app = ( VulkObj* )glfwGetWindowUserPointer( window );
    app->framebufferResized = true;

}
/*Vulkan Utility Implementation*/

/*Vulkan Core Implementation*/
void createColorResources
(
    void
)
{
    VkFormat colorFormat = Vulk.swapChainImageFormat;

    createImage( Vulk.swapChainExtent.width, Vulk.swapChainExtent.height, 1, Vulk.msaaSmples , colorFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &Vulk.colorImage, &Vulk.colorImageMemory);
    Vulk.colorImageView = createImageView( Vulk.colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
}
VkSampleCountFlagBits getMaxUsableSampleCount
(
    void
)
{
    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(Vulk.physicalDevice, &physicalDeviceProperties);

    VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;

    if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
    if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
    if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
    if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
    if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
    if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

    return VK_SAMPLE_COUNT_1_BIT;

}

void generateMipmaps
(
    VkImage image,
    VkFormat imageFormat,
    int32_t texWidth,
    int32_t texHeight,
    uint32_t mipLevels
)
{
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(Vulk.physicalDevice,imageFormat,&formatProperties);

    if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
    {
        perror("texture image format does not support linear blitting");
        exit(EXIT_FAILURE);
    }
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier = {0};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;
   // endSingleTimeCommands(commandBuffer);

    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;

    for(uint32_t i = 1 ; i < Vulk.mipLevels ; ++i)
    {
           barrier.subresourceRange.baseMipLevel = i - 1;
           barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
           barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
           barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
           barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                            0, NULL,
                            0, NULL,
                            1, &barrier);

        VkImageBlit blit = {0};
        blit.srcOffsets[0].x = 0;
        blit.srcOffsets[0].y = 0;
        blit.srcOffsets[0].z = 0;
        blit.srcOffsets[1].x = mipWidth;
        blit.srcOffsets[1].y = mipHeight;
        blit.srcOffsets[1].z = 1;
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[0].x = 0;
        blit.dstOffsets[0].y = 0;
        blit.dstOffsets[0].z = 0;
        blit.dstOffsets[1].x = mipWidth>1 ? mipWidth/2 : 1;
        blit.dstOffsets[1].y = mipHeight>1 ? mipHeight/2 : 1;
        blit.dstOffsets[1].z = 1;
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;

        vkCmdBlitImage(commandBuffer,
                        image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1, &blit,
                        VK_FILTER_LINEAR);

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                0, NULL,
                0, NULL,
                1, &barrier);
            if(mipWidth > 1) mipWidth /= 2;
            if(mipHeight > 1) mipHeight /= 2;


    }
      barrier.subresourceRange.baseMipLevel = mipLevels - 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(commandBuffer,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                 0, NULL,
                 0, NULL,
                 1, &barrier);

    endSingleTimeCommands(commandBuffer);
}

void loadModel
(
    const char* model_path
)
{
    const struct aiScene* scene = aiImportFile( model_path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace );

    if ( scene == NULL )
    {
        perror( "failed to load model!" );
        exit( EXIT_FAILURE );
    }
    meshes = (Mesh*)calloc(scene->mNumMeshes,sizeof(Mesh));
    mesh_size = scene->mNumMeshes;
    processNode( scene->mRootNode, scene,meshes );
    aiReleaseImport( scene );
};

void processNode
(
    const struct aiNode* node,
    const struct aiScene* scene,
    Mesh* mesh
)
{

    // Process all meshes in this node
    for (unsigned int i = 0; i < node->mNumMeshes; i++) {
        struct aiMesh* assimp_mesh = scene->mMeshes[node->mMeshes[i]];
        processMesh(assimp_mesh,&mesh[i]);
    }

    // Recursively process each child node
    for (unsigned int i = 0; i < node->mNumChildren; i++) {
        processNode(node->mChildren[i], scene, meshes);
    }

}

void processMesh
(
    const struct aiMesh* assimp_mesh,
    Mesh* mesh
)
{
    // Print the number of vertices and faces
    printf("Mesh has %d vertices and %d faces\n", assimp_mesh->mNumVertices, assimp_mesh->mNumFaces);
    memset( mesh, 0, sizeof(Mesh) );
    mesh->vertices_count=assimp_mesh->mNumVertices;
    mesh->vertices_array=(Vertex*)calloc( mesh->vertices_count , sizeof( Vertex ) );



    indices_size=3*assimp_mesh->mNumFaces;
    vertices_size=assimp_mesh->mNumVertices;



    // Iterate through vertices
    for (unsigned int i = 0; i < assimp_mesh->mNumVertices; i++) {
        //Vertex vertex={0};
        mesh->vertices_array[i].pos[0] = assimp_mesh->mVertices[i].x;
        mesh->vertices_array[i].pos[1] = assimp_mesh->mVertices[i].y;
        mesh->vertices_array[i].pos[2] = assimp_mesh->mVertices[i].z;

        // Print texture coordinates if available
        if (assimp_mesh->mTextureCoords[0]) {
            mesh->vertices_array[i].texCoord[0] = assimp_mesh->mTextureCoords[0][i].x;
            mesh->vertices_array[i].texCoord[1] =assimp_mesh->mTextureCoords[0][i].y;
        }else {

            mesh->vertices_array[i].texCoord[0] = 0.0f;
            mesh->vertices_array[i].texCoord[1] = 0.0f;
        }
        mesh->vertices_array[i].color[0] = 1.0f;
        mesh->vertices_array[i].color[1] = 1.0f;
        mesh->vertices_array[i].color[2] = 1.0f;

        //printf("\n");
        //Vulk.vertices[i]=vertex;
    }

    mesh->index_count = assimp_mesh->mNumFaces * assimp_mesh->mFaces[ 0 ].mNumIndices;
    mesh->indices_arry = (uint32_t*)calloc( mesh->index_count , sizeof( uint32_t ) );
    // Iterate through faces (triangles)

    for (unsigned int i = 0; i < assimp_mesh->mNumFaces; i++) {
        struct aiFace face = assimp_mesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; j++) {
           // printf("%d ", face.mIndices[j]);
            mesh->indices_arry[3*i + j]=face.mIndices[j];
        }
       // putchar(10);

    }

}



 VkVertexInputBindingDescription getBindingDescription
(
    void
)
{
    VkVertexInputBindingDescription bindingDescription = {0};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof( Vertex );
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    return ( bindingDescription );
}

 VkVertexInputAttributeDescription* getAttributeDescriptions
(
    void
)
{
        VkVertexInputAttributeDescription* attributedescriptions = (VkVertexInputAttributeDescription*)calloc( 3, sizeof( VkVertexInputAttributeDescription ) );

        attributedescriptions[0].binding = 0;
        attributedescriptions[0].location = 0;
        attributedescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributedescriptions[0].offset = offsetof( Vertex, pos );

        attributedescriptions[1].binding = 0;
        attributedescriptions[1].location = 1;
        attributedescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributedescriptions[1].offset = offsetof(  Vertex, color );

        attributedescriptions[2].binding = 0;
        attributedescriptions[2].location = 2;
        attributedescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributedescriptions[2].offset = offsetof(  Vertex, texCoord );
        return ( attributedescriptions );
}


 void initVulkan
 (
     void
 )
 {
    creatInstance();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createColorResources();
    createDepthResources();
    createFramebuffers();
    createCommandPool();
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    loadModel(MODEL_PATH);

    for(uint32_t i = 0; i < mesh_size ; ++i){
    createVertexBuffer(i);
    createIndexBuffer(i);
    }

    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffer();
    createSyncObjects();
}


void mainLoop
(
    void
)
{

    while( !glfwWindowShouldClose( Vulk.window ) )
    {
        glfwPollEvents();
        drawFrame();
    }
}

void initWindow
(
     void
)
{
    glfwInit();
    glfwWindowHint(GLFW_RESIZABLE,GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    Vulk.window=glfwCreateWindow( WIDTH,HEIGHT, "VULKAN", NULL, NULL );
    glfwSetWindowUserPointer( Vulk.window, &Vulk );
    glfwSetFramebufferSizeCallback( Vulk.window, framebufferResizeCallback );
}

void creatInstance
(
    void
)
{


     if( enableValidationLayers && !checkValidationlayersSupport() )
     {

        perror("validation layers requested, but not available!");
        exit( EXIT_FAILURE );

     }

        VkApplicationInfo appInfo = {0};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1,0,0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1,0,0);
        appInfo.apiVersion = VK_API_VERSION_1_0;


        VkInstanceCreateInfo createInfo = {0};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;


        size_t validationLayers_size = (size_t)1;

        if( enableValidationLayers )
        {

        createInfo.enabledLayerCount = (uint32_t)(validationLayers_size);
        createInfo.ppEnabledExtensionNames=validationLayers;

        }
        else
        {

        createInfo.enabledLayerCount = 0;

        }

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions( &glfwExtensionCount );
        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;
        createInfo.enabledLayerCount = 0;

        if( vkCreateInstance( &createInfo, NULL, &Vulk.instance ) != VK_SUCCESS )
        {

            perror("failed to create instance!");
            exit( EXIT_FAILURE );

        }

 }


void createSurface()
{


    if ( glfwCreateWindowSurface(Vulk.instance, Vulk.window, NULL, &Vulk.surface) != VK_SUCCESS )
    {
        perror("failed to create window surface!");
        exit( EXIT_FAILURE );
    }
}


void pickPhysicalDevice
(
    void
)
{

     Vulk.physicalDevice = VK_NULL_HANDLE;
     uint32_t deviceCount = 0;

     vkEnumeratePhysicalDevices( Vulk.instance, &deviceCount, NULL );

     if( deviceCount==0 )
     {

     perror("failed to find GPUs with Vulkan support!");
     exit( EXIT_FAILURE );

     }

      VkPhysicalDevice* devices = NULL;
      devices = (VkPhysicalDevice*)malloc( sizeof(VkPhysicalDevice)*deviceCount );

     if (devices == NULL)
     {

      perror("failed to allocate memory for devices!");
      exit( EXIT_FAILURE );

     }

     vkEnumeratePhysicalDevices( Vulk.instance, &deviceCount, devices );

     for(size_t i = 0;i < deviceCount; ++i)
     {
           if( isDeviceSuitable( devices[i] ) )
           {

               Vulk.physicalDevice = devices[i];
               Vulk.msaaSmples = getMaxUsableSampleCount();
               break;

           }

     }
     if( Vulk.physicalDevice == VK_NULL_HANDLE )
     {

       perror("failed to find a suitable GPU!");
       exit( EXIT_FAILURE );

     }
     free( devices );
 }


void createRenderPass
(
    void
)
{

    VkAttachmentDescription depthAttachment = {0};
    depthAttachment.format = findDepthFormat();
    depthAttachment.samples = Vulk.msaaSmples;
    depthAttachment.loadOp =  VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription colorAttachment = {0};
    colorAttachment.format = Vulk.swapChainImageFormat;
    colorAttachment.samples = Vulk.msaaSmples;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription colorAttachmentResolve = {0};
    colorAttachmentResolve.format = Vulk.swapChainImageFormat;
    colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachmentResolve.initialLayout =VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;


    VkAttachmentReference colorAttachmentResolveRef = {0};
    colorAttachmentResolveRef.attachment = 2;
    colorAttachmentResolveRef.layout =VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef = {0};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentRef = {0};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {0};
    subpass.pipelineBindPoint =  VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    subpass.pResolveAttachments = &colorAttachmentResolveRef;

    VkAttachmentDescription* attachments = (VkAttachmentDescription*)calloc( 3, sizeof( VkAttachmentDescription ) );
    attachments[0] = colorAttachment;
    attachments[1] = depthAttachment;
    attachments[2] = colorAttachmentResolve;

    size_t attachments_size = 3;

    VkRenderPassCreateInfo renderPassInfo = {0};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = (uint32_t)attachments_size;
    renderPassInfo.pAttachments = attachments;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;


    VkSubpassDependency dependency = {0};
    dependency.dstSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;
    if( vkCreateRenderPass( Vulk.device, &renderPassInfo, NULL, &Vulk.renderPass ) != VK_SUCCESS )
    {
      perror("failed to create render pass!");
      exit( EXIT_FAILURE );
    }

}

void createImageViews
(
    void
)
{

    Vulk.swapChainImageViews = (VkImageView*)calloc( image_size, sizeof( VkImageView ) );
    for(size_t i = 0; i < image_size; ++i)
    {

      Vulk.swapChainImageViews[i] = createImageView( Vulk.swapChainImages[i], Vulk.swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1 );

    }

}

void createFramebuffers
(
    void
)
{
    Vulk.swapChainFramebuffers = (VkFramebuffer*)calloc( image_size,sizeof( VkFramebuffer ) );

    for(size_t i = 0;i < image_size ;++i)
    {

        VkImageView attachments[]=
        {

            Vulk.colorImageView,
            Vulk.depthImageView,
            Vulk.swapChainImageViews[i]

        };
        size_t attachment_size = 3;
        VkFramebufferCreateInfo framebufferInfo = {0};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = Vulk.renderPass;
        framebufferInfo.attachmentCount = (uint32_t)attachment_size;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = Vulk.swapChainExtent.width;
        framebufferInfo.height = Vulk.swapChainExtent.height;
        framebufferInfo.layers = 1;

        if( vkCreateFramebuffer( Vulk.device, &framebufferInfo, NULL, &Vulk.swapChainFramebuffers[i] ) != VK_SUCCESS)
        {

            perror("failed to create framebuffer!");
            exit( EXIT_FAILURE );

        }

    }

}


 void createLogicalDevice
(
void
)
{
     QueueFamilyIndices indicies = findQueueFamilies(Vulk.physicalDevice);

    VkDeviceQueueCreateInfo* queueCreateInfos;
    queueCreateInfos=(VkDeviceQueueCreateInfo*)calloc( 2, sizeof( VkDeviceQueueCreateInfo ) );

    if( queueCreateInfos == NULL )
    {
       perror("failed to allocate memory for queueCreateInfos ");
       exit( EXIT_FAILURE );
    }

    float queuePriority = 1.0f;

    VkDeviceQueueCreateInfo queueCreateInfo_graphics = {0};
    queueCreateInfo_graphics.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo_graphics.queueFamilyIndex = indicies.graphicsFamily;
    queueCreateInfo_graphics.queueCount = 1;
    queueCreateInfo_graphics.pQueuePriorities = &queuePriority;


    VkDeviceQueueCreateInfo queueCreateInfo_present = {0};
    queueCreateInfo_present.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo_present.queueFamilyIndex = indicies.presentFamily;
    queueCreateInfo_present.queueCount = 1;
    queueCreateInfo_present.pQueuePriorities = &queuePriority;


    queueCreateInfos[0] = queueCreateInfo_graphics;
    queueCreateInfos[1] = queueCreateInfo_present;

    size_t queueCreateInfos_size = 2;
    VkPhysicalDeviceFeatures deviceFeatures = {0};
    deviceFeatures.samplerAnisotropy = VK_TRUE;
    deviceFeatures.sampleRateShading = VK_TRUE;
    uint32_t deviceExtensions_size = 1;
    VkDeviceCreateInfo createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = (uint32_t)queueCreateInfos_size;
    createInfo.pQueueCreateInfos = queueCreateInfos;
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = deviceExtensions_size;
    createInfo.ppEnabledExtensionNames = deviceExtensions;

    size_t validationLayers_size = 1;
    if( enableValidationLayers )
    {

        createInfo.enabledLayerCount = (uint32_t)validationLayers_size;
        createInfo.ppEnabledLayerNames = validationLayers;

    }
    else
    {

        createInfo.enabledLayerCount = 0;

    }
    if( vkCreateDevice( Vulk.physicalDevice, &createInfo, NULL, &Vulk.device ) != VK_SUCCESS )
    {
      perror("failed to create logical device!");
      exit( EXIT_FAILURE );
    }

    vkGetDeviceQueue( Vulk.device, indicies.graphicsFamily, 0, &Vulk.graphicsQueue );
    vkGetDeviceQueue( Vulk.device, indicies.presentFamily, 0, &Vulk.presentQueue );

 }


 void createGraphicsPipeline
(
void
)
{
    size_t vert_size;
    size_t frag_size;
    char* vertShaderCode = readFile( SHADERS_PATH"/vert.spv", &vert_size );
    char* fragShaderCode = readFile( SHADERS_PATH"/frag.spv", &frag_size );


    VkShaderModule vertShaderModule = createShaderModule( vertShaderCode, vert_size);
    VkShaderModule fragShaderModule = createShaderModule( fragShaderCode, frag_size);


    VkPipelineShaderStageCreateInfo vertShaderStageInfo = {0};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";


    VkPipelineShaderStageCreateInfo fragShaderStageInfo = {0};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";


    VkPipelineShaderStageCreateInfo shaderStages[] =
    {
        vertShaderStageInfo,
        fragShaderStageInfo
    };


    VkDynamicState* dynamicStates = ( VkDynamicState* )malloc( 2*sizeof( VkDynamicState ) );
    if( dynamicStates == NULL )
    {
      perror("failed to allocate memory for dynamicStates!");
      exit( EXIT_FAILURE );
    }
    dynamicStates[0] = VK_DYNAMIC_STATE_VIEWPORT;
    dynamicStates[1] = VK_DYNAMIC_STATE_SCISSOR;


    size_t dynamicState_size = 2;
    VkPipelineDynamicStateCreateInfo dynamicState = {0};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = (uint32_t)dynamicState_size;
    dynamicState.pDynamicStates = dynamicStates;


    VkVertexInputBindingDescription bindingDescription = getBindingDescription();
    VkVertexInputAttributeDescription* attributeDescriptions = getAttributeDescriptions();
    uint32_t attr_size = 3;


    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {0};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = attr_size;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions;


    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {0};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;


    VkViewport viewport = {0};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)Vulk.swapChainExtent.width;
    viewport.height = (float)Vulk.swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;


    VkRect2D scissor = {0};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent = Vulk.swapChainExtent;


    VkPipelineViewportStateCreateInfo viewportState = {0};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;


    VkPipelineRasterizationStateCreateInfo rasterizer = {0};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasClamp = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0f;


    VkPipelineMultisampleStateCreateInfo multisampling = {0};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_TRUE;
    multisampling.rasterizationSamples = Vulk.msaaSmples;
    multisampling.minSampleShading = .2f;
    multisampling.pSampleMask = NULL;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;


    VkPipelineDepthStencilStateCreateInfo depthStencil = {0};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f;
    depthStencil.maxDepthBounds = 1.0f;
    depthStencil.stencilTestEnable = VK_FALSE;


    VkPipelineColorBlendAttachmentState colorBlendAttachment = {0};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending = {0};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    VkPipelineLayoutCreateInfo pipelinelayoutInfo = {0};
    pipelinelayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelinelayoutInfo.setLayoutCount = 1;
    pipelinelayoutInfo.pSetLayouts = &Vulk.descriptorSetLayout;
    pipelinelayoutInfo.pushConstantRangeCount =  0;
    pipelinelayoutInfo.pPushConstantRanges = NULL;

    if( vkCreatePipelineLayout( Vulk.device, &pipelinelayoutInfo, NULL, &Vulk.pipelineLayout ) != VK_SUCCESS )
    {

        perror("failed to create pipeline layout!");
        exit( EXIT_FAILURE );

    }

    VkGraphicsPipelineCreateInfo pipelineInfo = {0};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = NULL;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.layout = Vulk.pipelineLayout;
    pipelineInfo.renderPass = Vulk.renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    if( vkCreateGraphicsPipelines( Vulk.device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &Vulk.graphicsPipeline ) != VK_SUCCESS )
    {

        perror("failed to create graphics pipeline!");
        exit( EXIT_FAILURE );

    }
     vkDestroyShaderModule( Vulk.device, fragShaderModule, NULL );
     vkDestroyShaderModule( Vulk.device, vertShaderModule, NULL );

}

 void createCommandPool
(
    void
)
{
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(Vulk.physicalDevice);

    VkCommandPoolCreateInfo poolInfo = {0};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;

    if( vkCreateCommandPool( Vulk.device, &poolInfo, NULL, &Vulk.commandPool ) != VK_SUCCESS)
    {

        perror("failed to create command pool!");
        exit( EXIT_FAILURE );

    }

}

void createCommandBuffer
(
void
)
{
    Vulk.commandBuffers = (VkCommandBuffer*)calloc( MAX_FRAMES_IN_FLIGHT, sizeof( VkCommandBuffer ) );

    if( Vulk.commandBuffers == NULL )
    {
        perror("failed to allocate memory for commandBuffers ");
        exit( EXIT_FAILURE );
    }

    VkCommandBufferAllocateInfo allocInfo = {0};

    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = Vulk.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

    if( vkAllocateCommandBuffers( Vulk.device, &allocInfo, Vulk.commandBuffers ) != VK_SUCCESS )
    {

        perror("failed to create command buffers!");
        exit( EXIT_FAILURE );

    }
}


void createSyncObjects
(
void
)
{
    Vulk.imageAvailableSemaphores = (VkSemaphore*)calloc( MAX_FRAMES_IN_FLIGHT, sizeof( VkSemaphore ) );
    Vulk.renderFinishedSemaphores = (VkSemaphore*)calloc( MAX_FRAMES_IN_FLIGHT, sizeof( VkSemaphore ) );
    Vulk.inFlightFences = (VkFence*)calloc( MAX_FRAMES_IN_FLIGHT, sizeof( VkFence ) );

    if( Vulk.imageAvailableSemaphores == NULL || Vulk.renderFinishedSemaphores == NULL || Vulk.inFlightFences == NULL )
    {
        perror("failed to allocate memory for sync objects");
        exit( EXIT_FAILURE );
    }

    VkSemaphoreCreateInfo semaphoreInfo = {0};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo = {0};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for( size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i )
    {
        if( vkCreateSemaphore( Vulk.device, &semaphoreInfo, NULL, &Vulk.imageAvailableSemaphores[i] ) != VK_SUCCESS ||
            vkCreateSemaphore( Vulk.device, &semaphoreInfo, NULL, &Vulk.renderFinishedSemaphores[i] ) != VK_SUCCESS ||
            vkCreateFence( Vulk.device, &fenceInfo, NULL, &Vulk.inFlightFences[i] ) != VK_SUCCESS )
        {
            perror("failed to create sync objects!");
            exit( EXIT_FAILURE );
        }
    }
}


void createVertexBuffer
(
    uint32_t index
)
{

    VkDeviceSize bufferSize = sizeof(meshes[index].vertices_array[0]) * meshes[index].vertices_count;

    VkBuffer stagingBuffer = {0};
    VkDeviceMemory stagingBufferMemory = {0};

    createBuffer( bufferSize,
                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  &stagingBuffer,
                  &stagingBufferMemory );

    void* data = NULL;
    vkMapMemory( Vulk.device, stagingBufferMemory, 0, bufferSize, 0, &data );
    memcpy( data, meshes[index].vertices_array, (size_t)bufferSize );
    vkUnmapMemory( Vulk.device, stagingBufferMemory );

    createBuffer( bufferSize,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  &Vulk.vertexBuffer,
                  &Vulk.vertexBufferMemory );

    copyBuffer( stagingBuffer, Vulk.vertexBuffer, bufferSize );

    vkDestroyBuffer( Vulk.device, stagingBuffer, NULL );
    vkFreeMemory( Vulk.device, stagingBufferMemory, NULL );
}


 void createIndexBuffer
(
    uint32_t index
)
{

    VkDeviceSize bufferSize = sizeof(meshes[index].indices_arry[0]) * meshes[index].index_count;

    VkBuffer stagingBuffer = {0};
    VkDeviceMemory stagingBufferMemory = {0};

    createBuffer( bufferSize,
                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  &stagingBuffer,
                  &stagingBufferMemory );

    void *data = NULL;
    vkMapMemory( Vulk.device, stagingBufferMemory, 0, bufferSize, 0, &data );
    memcpy( data, meshes[index].indices_arry, (size_t)bufferSize );
    vkUnmapMemory( Vulk.device, stagingBufferMemory );

    createBuffer( bufferSize,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  &Vulk.indexBuffer,
                  &Vulk.indexBufferMemory );

    copyBuffer( stagingBuffer, Vulk.indexBuffer, bufferSize );

    vkDestroyBuffer( Vulk.device, stagingBuffer, NULL );
    vkFreeMemory( Vulk.device, stagingBufferMemory, NULL );
}


void createDescriptorSetLayout
(
void
)
{
    VkDescriptorSetLayoutBinding uboLayoutBinding = {0};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uboLayoutBinding.pImmutableSamplers = NULL;

    VkDescriptorSetLayoutBinding samplerLayoutBinding = {0};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = NULL;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding* bindings = (VkDescriptorSetLayoutBinding*)calloc( 2, sizeof( VkDescriptorSetLayoutBinding ) );

    if( bindings == NULL )
    {
        perror("failed to allocate memory for bindings");
        exit( EXIT_FAILURE );
    }

    bindings[0] = uboLayoutBinding;
    bindings[1] = samplerLayoutBinding;

    size_t binding_size = 2;
    VkDescriptorSetLayoutCreateInfo layoutInfo = {0};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = (uint32_t)binding_size;
    layoutInfo.pBindings = bindings;

    if( vkCreateDescriptorSetLayout( Vulk.device, &layoutInfo, NULL, &Vulk.descriptorSetLayout ) != VK_SUCCESS )
    {
        perror("failed to create descriptor set layout");
        exit( EXIT_FAILURE );
    }

    free( bindings );
}


void createDescriptorPool
(
void
)
{
    VkDescriptorPoolSize* poolSizes = (VkDescriptorPoolSize*)calloc( 2, sizeof( VkDescriptorPoolSize ) );

    if( poolSizes == NULL )
    {
        perror("failed to allocate memory for poolSizes");
        exit( EXIT_FAILURE );
    }

    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = (uint32_t)(MAX_FRAMES_IN_FLIGHT);
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = (uint32_t)(MAX_FRAMES_IN_FLIGHT);

    size_t pool_size = 2;
    VkDescriptorPoolCreateInfo poolInfo = {0};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = (uint32_t)pool_size;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = (uint32_t)(MAX_FRAMES_IN_FLIGHT);

    if( vkCreateDescriptorPool( Vulk.device, &poolInfo, NULL, &Vulk.descriptorPool ) != VK_SUCCESS )
    {
        perror("failed to create descriptor pool!");
        exit( EXIT_FAILURE );
    }

    free( poolSizes );
}


 void createDescriptorSets
(
void
)
{
    VkDescriptorSetLayout* layouts = (VkDescriptorSetLayout*)calloc( MAX_FRAMES_IN_FLIGHT, sizeof( VkDescriptorSetLayout ) );

    if( layouts == NULL )
    {
        perror("failed to allocate memory for layouts");
        exit( EXIT_FAILURE );
    }

    for( size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i )
    {
        layouts[i] = Vulk.descriptorSetLayout;
    }

    VkDescriptorSetAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = Vulk.descriptorPool;
    allocInfo.descriptorSetCount = (uint32_t)MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts = layouts;

    Vulk.descriptorSets = (VkDescriptorSet*)calloc( MAX_FRAMES_IN_FLIGHT, sizeof( VkDescriptorSet ) );

    if( Vulk.descriptorSets == NULL )
    {
        perror("failed to allocate memory for descriptor sets");
        exit( EXIT_FAILURE );
    }

    if( vkAllocateDescriptorSets( Vulk.device, &allocInfo, Vulk.descriptorSets ) != VK_SUCCESS )
    {
        perror("failed to allocate descriptor sets!");
        exit( EXIT_FAILURE );
    }

    for( size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i )
    {
        VkDescriptorBufferInfo bufferInfo = {0};
        bufferInfo.buffer = Vulk.uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof( UniformBufferObjects );

        VkDescriptorImageInfo imageInfo = {0};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = Vulk.textureImageView;
        imageInfo.sampler = Vulk.textureSampler;

        VkWriteDescriptorSet* descriptorWrites = (VkWriteDescriptorSet*)calloc( 2, sizeof( VkWriteDescriptorSet ) );

        if( descriptorWrites == NULL )
        {
            perror("failed to allocate memory for descriptorWrites");
            exit( EXIT_FAILURE );
        }

        size_t descriptors_size = 2;

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = Vulk.descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = Vulk.descriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo;

        vkUpdateDescriptorSets( Vulk.device, (uint32_t)descriptors_size, descriptorWrites, 0, NULL );

        free( descriptorWrites );
    }

    free( layouts );
}


 void createTextureImage
(
void
)
{
    int texWidth = 0, texHeight = 0, texChannels = 0;
    stbi_uc* pixels = stbi_load( TEXTURE_PATH, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha );
    Vulk.mipLevels = (uint32_t)floor(log2(max(texWidth,texHeight))) + 1 ;

    VkDeviceSize imageSize = texWidth * texHeight * 4;

    if( !pixels )
    {
        perror("failed to load texture image!");
        exit( EXIT_FAILURE );
    }

    VkBuffer stagingBuffer = {0};
    VkDeviceMemory stagingBufferMemory = {0};

    createBuffer( imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBuffer, &stagingBufferMemory );

    void* data;
    vkMapMemory( Vulk.device, stagingBufferMemory, 0, imageSize, 0, &data );
    memcpy( data, pixels, (size_t)imageSize );
    vkUnmapMemory( Vulk.device, stagingBufferMemory );

    stbi_image_free( pixels );

    createImage( texWidth, texHeight, Vulk.mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &Vulk.textureImage, &Vulk.textureImageMemory );

    transitionImageLayout( Vulk.textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, Vulk.mipLevels );
    copyBufferToImage( stagingBuffer, Vulk.textureImage, (uint32_t)texWidth, (uint32_t)texHeight );
   // transitionImageLayout( Vulk.textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, Vulk.mipLevels );

    vkDestroyBuffer( Vulk.device, stagingBuffer, NULL );
    vkFreeMemory( Vulk.device, stagingBufferMemory, NULL );
    generateMipmaps(Vulk.textureImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, Vulk.mipLevels);
}


 void createTextureSampler
(
    void
)
{

    VkSamplerCreateInfo samplerInfo = {0};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    VkPhysicalDeviceProperties properties = {0};
    vkGetPhysicalDeviceProperties( Vulk.physicalDevice, &properties );
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;

    if( vkCreateSampler( Vulk.device, &samplerInfo, NULL, &Vulk.textureSampler ) != VK_SUCCESS )
    {
        perror("faild to create texture sampler!");
        exit( EXIT_FAILURE );
    }

}

void createDepthResources
(
    void
)
{
        VkFormat depthFormat = findDepthFormat();
        createImage( Vulk.swapChainExtent.width, Vulk.swapChainExtent.height, 1, Vulk.msaaSmples, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &Vulk.depthImage, &Vulk.depthImageMemory);
        Vulk.depthImageView = createImageView( Vulk.depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT,1 );
}

void createUniformBuffers
(
    void
)
{
    VkDeviceSize bufferSize = sizeof( UniformBufferObjects);
    Vulk.uniformBuffers = (VkBuffer*)calloc( MAX_FRAMES_IN_FLIGHT, sizeof( VkBuffer ) );
    Vulk.uniformBuffersMemory = (VkDeviceMemory*)calloc( MAX_FRAMES_IN_FLIGHT, sizeof( VkDeviceMemory ) );
    Vulk.uniformBuffersMapped = (void**)calloc( MAX_FRAMES_IN_FLIGHT, sizeof( void* ) );

    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {

        createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &Vulk.uniformBuffers[i], &Vulk.uniformBuffersMemory[i] );
        vkMapMemory( Vulk.device, Vulk.uniformBuffersMemory[i], 0, bufferSize, 0, &Vulk.uniformBuffersMapped[i] );

    }
}

 void createTextureImageView
(
    void
)
{

     Vulk.textureImageView=createImageView(Vulk.textureImage,VK_FORMAT_R8G8B8A8_SRGB,VK_IMAGE_ASPECT_COLOR_BIT, Vulk.mipLevels);

}

 void cleanupSwapChain
(
    void
)
{

    vkDestroyImageView( Vulk.device, Vulk.colorImageView, NULL );
    vkDestroyImage( Vulk.device, Vulk.colorImage, NULL );
    vkFreeMemory( Vulk.device, Vulk.colorImageMemory, NULL );

    vkDestroyImageView( Vulk.device, Vulk.depthImageView, NULL );
    vkDestroyImage( Vulk.device, Vulk.depthImage, NULL );
    vkFreeMemory( Vulk.device, Vulk.depthImageMemory, NULL );

    for(size_t i = 0;i < image_size ; ++i)
    {

         vkDestroyFramebuffer( Vulk.device, Vulk.swapChainFramebuffers[i], NULL );

    }
    for(size_t i = 0; i < image_size; ++i)
    {

        vkDestroyImageView( Vulk.device, Vulk.swapChainImageViews[i], NULL );

    }
    vkDestroySwapchainKHR( Vulk.device, Vulk.swapChain, NULL );

}

 void cleanup
(
    void
)
{
    cleanupSwapChain();

    vkDestroySampler(Vulk.device, Vulk.textureSampler, NULL);
    vkDestroyImageView(Vulk.device, Vulk.textureImageView, NULL);
    vkDestroyImage(Vulk.device, Vulk.textureImage, NULL);
    vkFreeMemory(Vulk.device, Vulk.textureImageMemory, NULL);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vkDestroyBuffer(Vulk.device, Vulk.uniformBuffers[i], NULL);
        vkFreeMemory(Vulk.device, Vulk.uniformBuffersMemory[i], NULL);
    }

    vkDestroyDescriptorPool(Vulk.device, Vulk.descriptorPool, NULL);
    vkDestroyDescriptorSetLayout(Vulk.device, Vulk.descriptorSetLayout, NULL);

    vkDestroyBuffer(Vulk.device, Vulk.indexBuffer, NULL);
    vkFreeMemory(Vulk.device, Vulk.indexBufferMemory, NULL);

    vkDestroyBuffer(Vulk.device, Vulk.vertexBuffer, NULL);
    vkFreeMemory(Vulk.device, Vulk.vertexBufferMemory, NULL);

    vkDestroyPipeline(Vulk.device, Vulk.graphicsPipeline, NULL);
    vkDestroyPipelineLayout(Vulk.device, Vulk.pipelineLayout, NULL);
    vkDestroyRenderPass(Vulk.device, Vulk.renderPass, NULL);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vkDestroySemaphore(Vulk.device, Vulk.imageAvailableSemaphores[i], NULL);
        vkDestroySemaphore(Vulk.device, Vulk.renderFinishedSemaphores[i], NULL);
        vkDestroyFence(Vulk.device, Vulk.inFlightFences[i], NULL);
    }

    vkDestroyCommandPool(Vulk.device, Vulk.commandPool, NULL);
    vkDestroyDevice(Vulk.device, NULL);
    vkDestroySurfaceKHR(Vulk.instance, Vulk.surface, NULL);
    vkDestroyInstance(Vulk.instance, NULL);

    glfwDestroyWindow(Vulk.window);
    glfwTerminate();
}


 VkShaderModule createShaderModule
(
    const char* code,
    size_t code_size
)
{
    VkShaderModuleCreateInfo createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code_size;
    createInfo.pCode = (const uint32_t*)code;

    VkShaderModule shaderModule;
    if ( vkCreateShaderModule( Vulk.device, &createInfo, NULL, &shaderModule) != VK_SUCCESS )
    {
        perror("failed to create shader module!");
        exit( EXIT_FAILURE );
    }

    free( (char*)code );
    return ( shaderModule );
}


 void createBuffer
(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkBuffer* buffer,
    VkDeviceMemory* bufferMemory
)
{
    VkBufferCreateInfo bufferInfo = {0};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if ( vkCreateBuffer( Vulk.device, &bufferInfo, NULL, buffer ) != VK_SUCCESS )
    {
        perror( "failed to create vertex buffer!" );
        exit( EXIT_FAILURE );
    }

    VkMemoryRequirements memRequirements = {0};
    vkGetBufferMemoryRequirements( Vulk.device, *buffer, &memRequirements );

    VkMemoryAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType( memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT );

    if ( vkAllocateMemory( Vulk.device, &allocInfo, NULL, bufferMemory ) != VK_SUCCESS )
    {
        perror( "failed to allocate vertex buffer memory!" );
        exit( EXIT_FAILURE );
    }

    vkBindBufferMemory( Vulk.device, *buffer, *bufferMemory, 0 );
}


 void createImage
(
    uint32_t width,
    uint32_t height,
    uint32_t mipLevels,
    VkSampleCountFlagBits numSamples,
    VkFormat format,
    VkImageTiling tiling,
    VkImageUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkImage* image,
    VkDeviceMemory* imageMemory
)
{
    VkImageCreateInfo imageInfo = {0};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = (uint32_t)(width);
    imageInfo.extent.height = (uint32_t)(height);
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.samples =numSamples;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    imageInfo.flags = 0;

    if ( vkCreateImage( Vulk.device, &imageInfo, NULL, image ) != VK_SUCCESS )
    {
        perror( "failed to create image!" );
        exit( EXIT_FAILURE );
    }

    VkMemoryRequirements memRequirements = {0};
    vkGetImageMemoryRequirements( Vulk.device, *image, &memRequirements );

    VkMemoryAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType( memRequirements.memoryTypeBits, properties );

    if ( vkAllocateMemory( Vulk.device, &allocInfo, NULL, imageMemory ) != VK_SUCCESS )
    {
        perror( "failed to allocate image memory!" );
        exit( EXIT_FAILURE );
    }

    vkBindImageMemory( Vulk.device, *image, *imageMemory, 0 );
}


 void recordCommandBuffer
(
    VkCommandBuffer commandBuffer,
    uint32_t imageIndex
)
{


    VkCommandBufferBeginInfo beginInfo = {0};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;
    beginInfo.pInheritanceInfo = NULL;

    if ( vkBeginCommandBuffer( commandBuffer, &beginInfo ) != VK_SUCCESS )
    {
        perror( "failed to begin recording command buffer!" );
        exit( EXIT_FAILURE );
    }

    VkRenderPassBeginInfo renderPassInfo = {0};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = Vulk.renderPass;
    renderPassInfo.framebuffer = Vulk.swapChainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset.x = 0;
    renderPassInfo.renderArea.offset.y = 0;
    renderPassInfo.renderArea.extent = Vulk.swapChainExtent;

    VkClearValue* clearValues = (VkClearValue*)calloc( 2, sizeof(VkClearValue) );
    VkClearColorValue clearColorValue = { .float32 = { 0.0f, 0.0f, 0.0f, 0.1f } };
    clearValues[0].color = clearColorValue;
    VkClearDepthStencilValue clearDepthValue = { 1.0f, 0 };
    clearValues[1].depthStencil = clearDepthValue;
    size_t clearValues_size = 2;
    renderPassInfo.clearValueCount = (uint32_t)clearValues_size;
    renderPassInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass( commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE );
    vkCmdBindPipeline( commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, Vulk.graphicsPipeline );

    VkBuffer vertexBuffers[] = { Vulk.vertexBuffer };
    VkDeviceSize offsets[] = { 0 };


    vkCmdBindVertexBuffers( commandBuffer, 0, 1, vertexBuffers, offsets );
    vkCmdBindIndexBuffer( commandBuffer, Vulk.indexBuffer, 0, VK_INDEX_TYPE_UINT32 );
    vkCmdBindDescriptorSets( commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, Vulk.pipelineLayout, 0, 1, &Vulk.descriptorSets[currentFrame], 0, NULL );
    vkCmdDrawIndexed( commandBuffer, (uint32_t)indices_size, 1, 0, 0, 0 );

    VkViewport viewport = {0};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)(Vulk.swapChainExtent.width);
    viewport.height = (float)(Vulk.swapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport( commandBuffer, 0, 1, &viewport );

    VkRect2D scissor = {0};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent = Vulk.swapChainExtent;
    vkCmdSetScissor( commandBuffer, 0, 1, &scissor );

    // vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    vkCmdEndRenderPass( commandBuffer );

    if ( vkEndCommandBuffer( commandBuffer ) != VK_SUCCESS )
    {
        perror( "failed to record command buffer" );
        exit( EXIT_FAILURE );
    }
}

VkImageView createImageView
(
    VkImage image,
    VkFormat format,
    VkImageAspectFlags aspectFlags,
    uint32_t mipLevels
)
{


    VkImageViewCreateInfo viewInfo = {0};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView = {0};
    if ( vkCreateImageView( Vulk.device, &viewInfo, NULL, &imageView ) != VK_SUCCESS )
    {
        perror( "failed to create texture image view!" );
        exit( EXIT_FAILURE );
    }

    return( imageView );
}



 void createSwapChain
(
    void
)
{
    uint32_t format_size = 0, present_size = 0;
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport( Vulk.physicalDevice, &format_size, &present_size );
    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat( swapChainSupport.formats, (size_t)format_size );
    VkPresentModeKHR presentMode = chooseSwapPresentMode( swapChainSupport.presentModes, (size_t)present_size );
    VkExtent2D extent = chooseSwapExtent( &swapChainSupport.capabilities );
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

    if ( swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount )
    {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = Vulk.surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices = findQueueFamilies( Vulk.physicalDevice );
    uint32_t queueFamilyIndices[] = { indices.graphicsFamily, indices.presentFamily };

    if ( indices.graphicsFamily != indices.presentFamily )
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = NULL;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if ( vkCreateSwapchainKHR( Vulk.device, &createInfo, NULL, &Vulk.swapChain ) != VK_SUCCESS )
    {
        perror( "failed to create swap chain!" );
        exit( EXIT_FAILURE );
    }

    vkGetSwapchainImagesKHR( Vulk.device, Vulk.swapChain, &imageCount, NULL );
    Vulk.swapChainImages = (VkImage*)calloc( imageCount, sizeof(VkImage) );
    image_size = (size_t)imageCount;
    vkGetSwapchainImagesKHR( Vulk.device, Vulk.swapChain, &imageCount, Vulk.swapChainImages );

    Vulk.swapChainImageFormat = surfaceFormat.format;
    Vulk.swapChainExtent = extent;
}


 void recreateSwapChain
(
    void
)
{
    int width = 0, height = 0;

    while ( width == 0 || height == 0 )
    {
        glfwGetFramebufferSize( Vulk.window, &width, &height );
        glfwWaitEvents();
    }

    vkDeviceWaitIdle( Vulk.device );

    cleanupSwapChain();
    createSwapChain();
    createImageViews();
    createColorResources();
    createDepthResources();
    createFramebuffers();
}


 void drawFrame
(
    void
)
{
    vkWaitForFences ( Vulk.device, 1, &Vulk.inFlightFences[currentFrame], VK_TRUE, UINT64_MAX );
    uint32_t imageIndex = 0;
    VkResult result = vkAcquireNextImageKHR( Vulk.device, Vulk.swapChain, UINT64_MAX, Vulk.imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex );

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreateSwapChain();
        exit( EXIT_FAILURE );
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        perror("failed to acquire swap chain image!");
        // exit(EXIT_FAILURE);
    }

    vkResetFences( Vulk.device, 1, &Vulk.inFlightFences[currentFrame] );

    vkResetCommandBuffer( Vulk.commandBuffers[currentFrame], 0 );

    recordCommandBuffer( Vulk.commandBuffers[currentFrame], imageIndex );

    updateUniformBuffer( currentFrame );

    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = { Vulk.imageAvailableSemaphores[currentFrame] };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &Vulk.commandBuffers[currentFrame];

    VkSemaphore signalSemaphores[] = { Vulk.renderFinishedSemaphores[currentFrame] };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if ( vkQueueSubmit( Vulk.graphicsQueue, 1, &submitInfo, Vulk.inFlightFences[currentFrame] ) != VK_SUCCESS )
    {
        perror("failed to submit draw command buffer!");
        exit( EXIT_FAILURE );
    }

    VkPresentInfoKHR presentInfo = {0};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = { Vulk.swapChain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = NULL;

    result = vkQueuePresentKHR(Vulk.presentQueue,&presentInfo);

    if ( result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || Vulk.framebufferResized )
    {
        Vulk.framebufferResized = false;
        recreateSwapChain();
    }
    else if (result != VK_SUCCESS)
    {
        perror("failed to present swap chain image!");
        exit( EXIT_FAILURE );
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

VkPresentModeKHR chooseSwapPresentMode
(
    const VkPresentModeKHR *availablePresentModes,
    size_t size
)
{
    for (size_t i = 0; i < size; ++i)
    {
        if (availablePresentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            return ( availablePresentModes[i] );
        }
    }

    return ( VK_PRESENT_MODE_FIFO_KHR );
}


/*Vulkan Core Implementation*/
 void run (){
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}


int main() {

   run();

   return 0;
}