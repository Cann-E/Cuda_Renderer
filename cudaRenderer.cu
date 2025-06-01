#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>


#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32
#define THREADS_PER_BLOCK (BLOCK_WIDTH * BLOCK_HEIGHT)
#define BLOCK_SIZE THREADS_PER_BLOCK

#define TILE_WIDTH (BLOCK_WIDTH)
#define TILE_HEIGHT (BLOCK_HEIGHT)

#define MAX_CIRCLE_COUNT 10000
#define MAX_CIRCLES MAX_CIRCLE_COUNT

#define CIRCLES_PER_THREAD 200
#define CIRCLES_PER_THREAD_MAX CIRCLES_PER_THREAD

#define SCAN_BLOCK_DIM THREADS_PER_BLOCK

#include "exclusiveScan.cu_inl"
#include "circleBoxTest.cu_inl"

bool gUsePixelKernel = true;




////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

//Rand100k and Biglittle perfomance problem I tried chatgpt to see if it can help me to increase the performance for those 2 but no success:((((

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}





CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
	isFastGPU = true;


        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU)
    {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA GTX 480, 670 or 780.\n");
        printf("---------------------------------------------------------\n");
    }
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void CudaRenderer::clearImage() {
    // Set up 2D grid for per-pixel clearing
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y
    );

    // Use gradient background for snow scenes
    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        // White background for all other scenes
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 0.f);
    }

    cudaDeviceSynchronize();
}


// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}


// // kernelRenderCircles -- (CUDA device code)
// // This kernel renders circles to the image. Each thread renders a circle by checking if the pixel lies within the circle.
// // However, there is no synchronization to ensure the correct order of updates, which could lead to incorrect image results.

// __global__ void kernelRenderCircles() {
//     int threadId = threadIdx.x;   // Get the thread ID within the block
//     int blockId = blockIdx.x;     // Get the block ID
//     int blockSize = blockDim.x;   // Get the size of each block (number of threads in a block)

//     int imageWidth = cuConstRendererParams.imageWidth;  // Image width
//     int imageHeight = cuConstRendererParams.imageHeight; // Image height
//     float invWidth = 1.f / imageWidth;   // Inverse of the image width
//     float invHeight = 1.f / imageHeight; // Inverse of the image height

//     // Shared memory to store circle positions, radii, and indices for up to 128 circles per block
//     __shared__ float3 sharedPos[128];
//     __shared__ float  sharedRad[128];
//     __shared__ int    sharedIdx[128];

//     int totalCircles = cuConstRendererParams.numCircles;  // Total number of circles

//     // Loop through the circles in chunks of 128 per block
//     for (int tileStart = 0; tileStart < totalCircles; tileStart += 128) {
//         int circleIdx = tileStart + threadId;  // Calculate which circle each thread is responsible for

//         // Load circle data into shared memory if the circle index is valid
//         if (circleIdx < totalCircles) {
//             sharedPos[threadId] = *(float3*)&cuConstRendererParams.position[3 * circleIdx];
//             sharedRad[threadId] = cuConstRendererParams.radius[circleIdx];
//             sharedIdx[threadId] = circleIdx;
//         }
//         __syncthreads(); // Synchronize all threads within the block

//         // Loop through each circle in the block (up to 128 circles)
//         #pragma unroll 8  // Unroll the loop for better performance
//         for (int i = 0; i < 128; ++i) {
//             if (tileStart + i >= totalCircles) break;  // If the circle index exceeds the total number of circles, exit the loop

//             // Get circle properties from shared memory
//             float3 p = sharedPos[i];
//             float rad = sharedRad[i];
//             int circleIndex = sharedIdx[i];

//             // If the circle is outside the image bounds, skip it
//             if (p.x + rad < 0.f || p.x - rad > 1.f || p.y + rad < 0.f || p.y - rad > 1.f)
//                 continue;

//             // Calculate the bounding box of the circle in pixel coordinates
//             int minX = max(0, (int)((p.x - rad) * imageWidth));
//             int maxX = min(imageWidth - 1, (int)((p.x + rad) * imageWidth));
//             int minY = max(0, (int)((p.y - rad) * imageHeight));
//             int maxY = min(imageHeight - 1, (int)((p.y + rad) * imageHeight));

//             bool isSnow = (cuConstRendererParams.sceneName == SNOWFLAKES ||
//                            cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME);  // Check if the scene is "snowflakes"

//             // Loop through the pixel grid within the circle's bounding box
//             for (int y = minY + threadId % 4; y <= maxY; y += 4) {  // Loop over pixels in the Y-direction
//                 for (int x = minX + threadId / 4; x <= maxX; x += blockSize / 4) {  // Loop over pixels in the X-direction
//                     // Calculate the center of the pixel
//                     float2 pixelCenter = make_float2((x + 0.5f) * invWidth, (y + 0.5f) * invHeight);

//                     // Calculate the distance from the pixel center to the circle center
//                     float dx = p.x - pixelCenter.x;
//                     float dy = p.y - pixelCenter.y;
//                     float dist2 = dx * dx + dy * dy;

//                     // If the pixel is outside the circle, skip it
//                     if (dist2 > rad * rad) continue;

//                     float3 rgb;
//                     float alpha;

//                     // Apply snowflake color and alpha if in snow scene, otherwise use regular circle color
//                     if (isSnow) {
//                         const float kCircleMaxAlpha = 0.5f;
//                         const float falloffScale = 4.f;
//                         float normDist = sqrtf(dist2) / rad;  // Normalize the distance
//                         rgb = lookupColor(normDist);  // Get color based on the normalized distance

//                         // Calculate the maximum alpha based on distance and falloff
//                         float maxAlpha = 0.6f + 0.4f * (1.f - p.z);
//                         maxAlpha = kCircleMaxAlpha * fminf(fmaxf(maxAlpha, 0.f), 1.f);
//                         alpha = maxAlpha * expf(-falloffScale * normDist * normDist);  // Apply falloff to alpha
//                     } else {
//                         rgb = *(float3*)&cuConstRendererParams.color[3 * circleIndex];  // Get the circle's color
//                         alpha = 0.5f;  // Set the default alpha for non-snow scenes
//                     }

//                     // Perform alpha blending with the current pixel color
//                     int pixelIdx = y * imageWidth + x;
//                     float4* imagePtr = ((float4*)cuConstRendererParams.imageData) + pixelIdx;

//                     float4 curr = *imagePtr;  // Get the current pixel color
//                     float oneMinusAlpha = 1.f - alpha;  // Calculate the complementary alpha
//                     float4 out;  // The output pixel color after blending
//                     out.x = alpha * rgb.x + oneMinusAlpha * curr.x;
//                     out.y = alpha * rgb.y + oneMinusAlpha * curr.y;
//                     out.z = alpha * rgb.z + oneMinusAlpha * curr.z;
//                     out.w = alpha + curr.w;  // Update the alpha channel

//                     // Store the resulting color back into the image data
//                     *imagePtr = out;
//                 }
//             }
//         }
//         __syncthreads();  // Synchronize all threads before processing the next set of circles
//     }
// }




// //-NEW------------------------------------------------
// #define COARSEN_X 2
// #define COARSEN_Y 2

// __global__ void kernelRenderPixels() {
//     int baseX = (blockIdx.x * blockDim.x + threadIdx.x) * COARSEN_X;
//     int baseY = (blockIdx.y * blockDim.y + threadIdx.y) * COARSEN_Y;

//     int width = cuConstRendererParams.imageWidth;
//     int height = cuConstRendererParams.imageHeight;

//     if (baseX >= width || baseY >= height) return;

//     float invWidth = 1.f / width;
//     float invHeight = 1.f / height;

//     for (int dy = 0; dy < COARSEN_Y; dy++) {
//         for (int dx = 0; dx < COARSEN_X; dx++) {
//             int x = baseX + dx;
//             int y = baseY + dy;
//             if (x >= width || y >= height) continue;

//             int pixelIndex = y * width + x;
//             float2 pixelCenter = make_float2((x + 0.5f) * invWidth, (y + 0.5f) * invHeight);

//             float4 pixelColor = ((float4*)cuConstRendererParams.imageData)[pixelIndex];

//             for (int i = 0; i < cuConstRendererParams.numCircles; ++i) {
//                 int idx3 = i * 3;
//                 float3 p = *(float3*)&cuConstRendererParams.position[idx3];
//                 float rad = cuConstRendererParams.radius[i];

//                 if (fabsf(p.x - pixelCenter.x) > rad || fabsf(p.y - pixelCenter.y) > rad)
//                     continue;

//                 float dx = p.x - pixelCenter.x;
//                 float dy = p.y - pixelCenter.y;
//                 float dist2 = dx * dx + dy * dy;
//                 if (dist2 > rad * rad)
//                     continue;

//                 float alpha;
//                 float3 rgb;

//                 bool isSnow = (cuConstRendererParams.sceneName == SNOWFLAKES ||
//                                cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME);

//                 if (isSnow) {
//                     const float kCircleMaxAlpha = 0.5f;
//                     const float falloffScale = 4.f;
//                     float normDist = sqrtf(dist2) / rad;
//                     rgb = lookupColor(normDist);
//                     float maxAlpha = 0.6f + 0.4f * (1.f - p.z);
//                     maxAlpha = kCircleMaxAlpha * fminf(fmaxf(maxAlpha, 0.f), 1.f);
//                     alpha = maxAlpha * expf(-falloffScale * normDist * normDist);
//                 } else {
//                     rgb = *(float3*)&cuConstRendererParams.color[idx3];
//                     alpha = 0.5f;
//                 }

//                 float oneMinusAlpha = 1.f - alpha;
//                 pixelColor.x = alpha * rgb.x + oneMinusAlpha * pixelColor.x;
//                 pixelColor.y = alpha * rgb.y + oneMinusAlpha * pixelColor.y;
//                 pixelColor.z = alpha * rgb.z + oneMinusAlpha * pixelColor.z;
//                 pixelColor.w = alpha + pixelColor.w;
//             }

//             ((float4*)cuConstRendererParams.imageData)[pixelIndex] = pixelColor;
//         }
//     }
// }




// //NEW-ENDING------------------------------------------


// ////////////////////////////////////////////////////////////////////////////////////////

// void CudaRenderer::render() {
//     dim3 blockDim(16, 16);
// dim3 gridDim(
//     (image->width + blockDim.x * COARSEN_X - 1) / (blockDim.x * COARSEN_X),
//     (image->height + blockDim.y * COARSEN_Y - 1) / (blockDim.y * COARSEN_Y)
// );


//     if (gUsePixelKernel) {
//         printf("Rendering with kernelRenderPixels\n");
//         kernelRenderPixels<<<gridDim, blockDim>>>();
//     } else {
//         printf("Rendering with kernelRenderCircles\n");
//         kernelRenderCircles<<<gridDim, blockDim>>>();
//     }

//     cudaDeviceSynchronize();
// }

//---------------------------------------------------------------------------------------

// #define TILE_WIDTH  16
// #define TILE_HEIGHT 16

// // Count overlaps for tiles. Each thread counts the overlaps for one circle.
// __global__ void kernelCountTileOverlaps(int numCircles, int imageWidth, int imageHeight,
//     const float* position, const float* radius,
//     int tileCountX, int* d_tileCount) {
// int c = blockIdx.x * blockDim.x + threadIdx.x;
// if (c >= numCircles) return;
// int idx3 = 3 * c;

// // Compute the bounding box of the circle in pixel space.
// float cx = position[idx3] * imageWidth;
// float cy = position[idx3+1] * imageHeight;
// float radPx = radius[c] * imageWidth;

// int minX = max(0, (int)floorf(cx - radPx));
// int maxX = min(imageWidth - 1, (int)ceilf(cx + radPx));
// int minY = max(0, (int)floorf(cy - radPx));
// int maxY = min(imageHeight - 1, (int)ceilf(cy + radPx));

// // Determine the tile range for this circle.
// int tMinX = minX / TILE_WIDTH;
// int tMaxX = maxX / TILE_WIDTH;
// int tMinY = minY / TILE_HEIGHT;
// int tMaxY = maxY / TILE_HEIGHT;

// // Update the tile overlap counts.
// for (int ty = tMinY; ty <= tMaxY; ty++) {
// for (int tx = tMinX; tx <= tMaxX; tx++) {
// int tileIndex = ty * tileCountX + tx;
// atomicAdd(&d_tileCount[tileIndex], 1);
// }
// }
// }

// __global__ void kernelFillTileCircleIndices(int numCircles, int imageWidth, int imageHeight,
//     const float* position, const float* radius,
//     int tileCountX, int* d_tileFill, int* d_tileStart,
//     int* d_tileCircleIndices) {
// int c = blockIdx.x * blockDim.x + threadIdx.x;
// if (c >= numCircles) return;

// int idx3 = 3 * c;
// float cx = position[idx3] * imageWidth;
// float cy = position[idx3 + 1] * imageHeight;
// float radPx = radius[c] * imageWidth;

// int minX = max(0, (int)floorf(cx - radPx));
// int maxX = min(imageWidth - 1, (int)ceilf(cx + radPx));
// int minY = max(0, (int)floorf(cy - radPx));
// int maxY = min(imageHeight - 1, (int)ceilf(cy + radPx));

// int tMinX = minX / TILE_WIDTH;
// int tMaxX = maxX / TILE_WIDTH;
// int tMinY = minY / TILE_HEIGHT;
// int tMaxY = maxY / TILE_HEIGHT;

// for (int ty = tMinY; ty <= tMaxY; ty++) {
// for (int tx = tMinX; tx <= tMaxX; tx++) {
// int tileIndex = ty * tileCountX + tx;
// // Use d_tileFill as a per-tile counter.
// int pos = atomicAdd(&d_tileFill[tileIndex], 1);
// d_tileCircleIndices[d_tileStart[tileIndex] + pos] = c;
// }
// }
// }

// __global__ void kernelRenderPixelsPerTile(int imageWidth, int imageHeight, int tileCountX,
    
// int* d_tileStart, int* d_tileFill, int* d_tileCircleIndices,
// float* imageData) {
// int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
// int pixelY = blockIdx.y * blockDim.y + threadIdx.y;
// if (pixelX >= imageWidth || pixelY >= imageHeight) return;

// int pixelIndex = pixelY * imageWidth + pixelX;
// int offset = 4 * pixelIndex;

// float4 pixelColor = make_float4(imageData[offset], imageData[offset + 1], imageData[offset + 2], imageData[offset + 3]);

// // Normalized coordinates for the pixel center.
// float invWidth = 1.f / imageWidth;
// float invHeight = 1.f / imageHeight;
// float2 pixelCenter = make_float2(invWidth * (pixelX + 0.5f), invHeight * (pixelY + 0.5f));

// // Get tile index for this pixel
// int tx = pixelX / TILE_WIDTH;
// int ty = pixelY / TILE_HEIGHT;
// int tileIndex = ty * tileCountX + tx;
// int count = d_tileFill[tileIndex];  // number of circles for this tile
// int start = d_tileStart[tileIndex];

// for (int i = 0; i < count; i++) {
// int circleIdx = d_tileCircleIndices[start + i];
// int idx3 = 3 * circleIdx;

// float3 circlePos = make_float3(cuConstRendererParams.position[idx3], cuConstRendererParams.position[idx3 + 1], cuConstRendererParams.position[idx3 + 2]);
// float rad = cuConstRendererParams.radius[circleIdx];

// // Quick bounding-box check in normalized coordinates.
// float minX = circlePos.x - rad;
// float maxX = circlePos.x + rad;
// float minY = circlePos.y - rad;
// float maxY = circlePos.y + rad;

// if (pixelCenter.x < minX || pixelCenter.x > maxX || pixelCenter.y < minY || pixelCenter.y > maxY)
// continue;

// float diffX = circlePos.x - pixelCenter.x;
// float diffY = circlePos.y - pixelCenter.y;
// float distSq = diffX * diffX + diffY * diffY;
// if (distSq > rad * rad)
// continue;

// float3 circleColor;
// float alpha;
// if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
// const float kCircleMaxAlpha = 0.5f;
// const float falloffScale = 4.f;
// float normDist = sqrtf(distSq) / rad;
// circleColor = lookupColor(normDist);
// float maxAlpha = kCircleMaxAlpha * fmaxf(fminf(0.6f + 0.4f * (1.f - circlePos.z), 1.f), 0.f);
// alpha = maxAlpha * expf(-falloffScale * normDist * normDist);
// } else {
// circleColor = make_float3(cuConstRendererParams.color[3 * circleIdx], cuConstRendererParams.color[3 * circleIdx + 1], cuConstRendererParams.color[3 * circleIdx + 2]);
// alpha = 0.5f;
// }

// float oneMinusAlpha = 1.0f - alpha;
// pixelColor.x = alpha * circleColor.x + oneMinusAlpha * pixelColor.x;
// pixelColor.y = alpha * circleColor.y + oneMinusAlpha * pixelColor.y;
// pixelColor.z = alpha * circleColor.z + oneMinusAlpha * pixelColor.z;
// pixelColor.w += alpha;
// }

// imageData[offset] = pixelColor.x;
// imageData[offset + 1] = pixelColor.y;
// imageData[offset + 2] = pixelColor.z;
// imageData[offset + 3] = pixelColor.w;
// }

// // Kernel to sort circle indices within each tile (using insertion sort).
// __global__ void kernelSortTileLists(int totalTiles, int *d_tileStart, int *d_tileFill, int *d_tileCircleIndices) {
//     int tileIndex = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tileIndex >= totalTiles) return;

//     int start = d_tileStart[tileIndex];
//     int count = d_tileFill[tileIndex];

//     // Insertion sort for small lists (sorting circle indices in each tile).
//     for (int i = 1; i < count; i++) {
//         int key = d_tileCircleIndices[start + i];
//         int j = i - 1;

//         // Shift elements of d_tileCircleIndices[] that are greater than key
//         // to one position ahead of their current position
//         while (j >= 0 && d_tileCircleIndices[start + j] > key) {
//             d_tileCircleIndices[start + j + 1] = d_tileCircleIndices[start + j];
//             j = j - 1;
//         }
//         d_tileCircleIndices[start + j + 1] = key;
//     }
// }


// void CudaRenderer::render() {
//     // 1. Determine tile counts.
//     int tileCountX = (image->width + TILE_WIDTH - 1) / TILE_WIDTH;
//     int tileCountY = (image->height + TILE_HEIGHT - 1) / TILE_HEIGHT;
//     int totalTiles = tileCountX * tileCountY;
//     int totalTileCircleIndices = 0;

//     // Declare device pointers for the per-tile data.
//     int *d_tileCount = NULL;
//     int *d_tileStart = NULL;
//     int *d_tileFill = NULL;
//     int *d_tileCircleIndices = NULL;

//     // 2. Allocate device memory for tile structure arrays.
//     cudaMalloc(&d_tileCount, sizeof(int) * totalTiles);
//     cudaMalloc(&d_tileStart, sizeof(int) * totalTiles);
//     cudaMalloc(&d_tileFill, sizeof(int) * totalTiles);
//     cudaMemset(d_tileCount, 0, sizeof(int) * totalTiles);
//     cudaMemset(d_tileFill, 0, sizeof(int) * totalTiles);

//     // 3. Launch kernelCountTileOverlaps (one thread per circle).
//     int blockSize = 256;
//     int gridSize = (numCircles + blockSize - 1) / blockSize;
//     kernelCountTileOverlaps<<<gridSize, blockSize>>>(numCircles, image->width, image->height,
//                                                      cudaDevicePosition, cudaDeviceRadius,
//                                                      tileCountX, d_tileCount);
//     cudaDeviceSynchronize();

//     // 4. Copy d_tileCount from GPU to host to compute exclusive scan.
//     std::vector<int> h_tileCount(totalTiles);
//     std::vector<int> h_tileStart(totalTiles, 0);
//     cudaMemcpy(h_tileCount.data(), d_tileCount, sizeof(int) * totalTiles, cudaMemcpyDeviceToHost);
//     int totalIndices = 0;
//     for (int i = 0; i < totalTiles; i++) {
//         h_tileStart[i] = totalIndices;
//         totalIndices += h_tileCount[i];
//     }
//     totalTileCircleIndices = totalIndices;
//     cudaMemcpy(d_tileStart, h_tileStart.data(), sizeof(int) * totalTiles, cudaMemcpyHostToDevice);

//     // 5. Allocate d_tileCircleIndices.
//     cudaMalloc(&d_tileCircleIndices, sizeof(int) * totalTileCircleIndices);

//     // 6. Launch kernelFillTileCircleIndices.
//     kernelFillTileCircleIndices<<<gridSize, blockSize>>>(numCircles, image->width, image->height,
//         cudaDevicePosition, cudaDeviceRadius,
//         tileCountX, d_tileFill, d_tileStart, d_tileCircleIndices);
//     cudaDeviceSynchronize();

//     // 7. Launch kernelSortTileLists (one thread per tile).
//     int sortBlockSize = 256;
//     int sortGridSize = (totalTiles + sortBlockSize - 1) / sortBlockSize;
//     kernelSortTileLists<<<sortGridSize, sortBlockSize>>>(totalTiles, d_tileStart, d_tileFill, d_tileCircleIndices);
//     cudaDeviceSynchronize();

//     // 8. Launch the rendering kernel.
//     dim3 blockDim(16, 16);
//     dim3 gridDim((image->width + blockDim.x - 1) / blockDim.x,
//                  (image->height + blockDim.y - 1) / blockDim.y);
//     kernelRenderPixelsPerTile<<<gridDim, blockDim>>>(image->width, image->height, tileCountX,
//                                                      d_tileStart, d_tileFill, d_tileCircleIndices,
//                                                      cudaDeviceImageData);
//     cudaDeviceSynchronize();

//     // 9. Free tile data structure arrays if not needed in the next frame.
//     cudaFree(d_tileCount);
//     cudaFree(d_tileStart);
//     cudaFree(d_tileFill);
//     cudaFree(d_tileCircleIndices);
// }



__device__ bool isCircleInTile(float x, float y, float r, float l, float rgt, float bot, float top) {
    float left = x - r;
    float right = x + r;
    float bottom = y - r;
    float up = y + r;
    return !(right < l || left > rgt || up < bot || bottom > top);
}

__global__ void optimizedCircleRenderer() {
    // Calculate thread ID within block
    int tIdx = threadIdx.y * BLOCK_WIDTH + threadIdx.x;

    int w = cuConstRendererParams.imageWidth;
    int h = cuConstRendererParams.imageHeight;

    float invW = 1.0f / w;
    float invH = 1.0f / h;

    __shared__ uint circleCountPerThread[BLOCK_SIZE];
    __shared__ uint finalCircleList[MAX_CIRCLES];
    __shared__ uint scanOutput[BLOCK_SIZE];

    int tileX = blockIdx.x * TILE_WIDTH;
    int tileY = blockIdx.y * TILE_HEIGHT;
    int tileEndX = min(tileX + TILE_WIDTH - 1, w - 1);
    int tileEndY = min(tileY + TILE_HEIGHT - 1, h - 1);

    float normL = (tileX - 1) * invW;
    float normR = (tileEndX + 1) * invW;
    float normB = (tileY - 1) * invH;
    float normT = (tileEndY + 1) * invH;

    int circlesPerT = (cuConstRendererParams.numCircles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int begin = tIdx * circlesPerT;
    int end = min(begin + circlesPerT, cuConstRendererParams.numCircles);

    uint localList[CIRCLES_PER_THREAD_MAX];
    int myCount = 0;

    bool isSnow = cuConstRendererParams.sceneName == SNOWFLAKES ||
                  cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME;

    // Each thread filters circles overlapping the tile
    for (int i = begin; i < end; ++i) {
        float3 pos = *((float3*)&cuConstRendererParams.position[i * 3]);
        float r = cuConstRendererParams.radius[i];
        if (isCircleInTile(pos.x, pos.y, r, normL, normR, normB, normT)) {
            localList[myCount++] = i;
        }
    }

    circleCountPerThread[tIdx] = myCount;
    __syncthreads();

    // Exclusive scan to prep for merging into shared list
    sharedMemExclusiveScan(tIdx, circleCountPerThread, scanOutput, finalCircleList, BLOCK_SIZE);
    __syncthreads();

    int writeAt = scanOutput[tIdx];
    for (int i = 0; i < myCount; ++i) {
        finalCircleList[writeAt + i] = localList[i];
    }
    __syncthreads();

    int totalCircles = scanOutput[BLOCK_SIZE - 1] + circleCountPerThread[BLOCK_SIZE - 1];
    int x = tileX + threadIdx.x;
    int y = tileY + threadIdx.y;

    if (x >= w || y >= h) return;

    float2 center = make_float2((x + 0.5f) * invW, (y + 0.5f) * invH);
    float4 pix = isSnow
        ? make_float4(0.4f + 0.45f * (float(h - y) / h), 0.4f + 0.45f * (float(h - y) / h), 0.4f + 0.45f * (float(h - y) / h), 1.0f)
        : make_float4(1.f, 1.f, 1.f, 1.f);

    for (int i = 0; i < totalCircles; ++i) {
        int c = finalCircleList[i];
        float3 pos = *((float3*)&cuConstRendererParams.position[c * 3]);
        float rad = cuConstRendererParams.radius[c];

        float dx = pos.x - center.x;
        float dy = pos.y - center.y;
        float distSq = dx * dx + dy * dy;

        if (distSq > rad * rad) continue;

        float3 color;
        float alpha;

        if (isSnow) {
            float normD = sqrtf(distSq) / rad;
            color = lookupColor(normD); // FIXED
            float maxA = 0.6f + 0.4f * (1.f - pos.z);
            maxA = fminf(1.f, fmaxf(0.f, maxA)); // FIXED clamp
            alpha = 0.5f * maxA * __expf(-4.f * normD * normD);
        } else {
            color = *((float3*)&cuConstRendererParams.color[c * 3]);
            alpha = 0.5f;
        }

        float invAlpha = 1.f - alpha;
        pix.x = alpha * color.x + invAlpha * pix.x;
        pix.y = alpha * color.y + invAlpha * pix.y;
        pix.z = alpha * color.z + invAlpha * pix.z;
        pix.w += alpha;
    }

    int outIdx = 4 * (y * w + x);
    *((float4*)&cuConstRendererParams.imageData[outIdx]) = pix;
}


void CudaRenderer::render() {
    dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 blocksPerGrid(
        (image->width + TILE_WIDTH - 1) / TILE_WIDTH,
        (image->height + TILE_HEIGHT - 1) / TILE_HEIGHT
    );

    optimizedCircleRenderer<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();
}




//Summary:
// I implemented a tile based CUDA renderer that processes only overlapping circles per tile and renders each pixel using a sorted list of circles to keep correct blending order.
//  I used shared memory, bounding box checks, and prefix sums on CPU to reduce overhead.

// I tried using Thrust (thrust::sort, exclusive_scan) to do GPU-side processing but ran into errors with CUDA 12.6 and couldn’t resolve them. .

// I struggled most with rand100k and biglittle. Despite trying multiple optimizations, their render times stayed high and couldn't be improved much.

// Brief Write up
// kernelRenderCircles(later not used)commented out in the code
// This kernel renders circles to the image. Each thread is responsible for checking if a pixel lies within a circle. 
// It uses shared memory to store circle data for efficiency and processes the circles in chunks. The function applies alpha blending, especially for snow scenes, and updates the pixel color accordingly. 
// However, there is no synchronization, so updates may not be in the correct order, which could lead to incorrect results.

// kernelRenderPixels(later not used)commented out in the code
// This kernel renders the image using a technique called "coarsening," where multiple pixels are processed together. It splits the image into small blocks of pixels (based on the COARSEN_X and COARSEN_Y values) and checks if any circles overlap with these pixels.
//  It applies the appropriate color and alpha blending, similar to kernelRenderCircles, but processes pixels in groups for improved efficiency.


// kernelCountTileOverlaps(later not used)commented out in the code
// This kernel computes the number of overlaps for each tile by iterating over all circles and checking if each circle overlaps with a tile's bounding box. For each circle, it calculates its pixel bounding box, determines the range of tiles that it intersects,
//  and updates the tile overlap count using atomic operations. This helps in efficiently distributing the work for rendering in later steps.

// kernelFillTileCircleIndices(later not used)commented out in the code
// This kernel fills the tile with the indices of the circles that intersect the tile. It performs a similar operation to the previous kernel, but here, for each overlap, the circle's index is saved into a shared array for later processing. 
// The indices are stored in a specific position within the tile's circle list, and the number of circles in each tile is updated atomically.

// kernelRenderPixelsPerTile(later not used)commented out in the code
// This kernel is responsible for rendering the circles to the image. For each pixel, the kernel checks whether it lies within any circle by calculating the distance from the pixel center to the circle's center. 
// If the pixel lies within the circle, it performs alpha blending with the circle's color. It processes the image in tiles for efficiency and handles snowflake specific coloring and alpha calculation when applicable.

// kernelSortTileLists(later not used)commented out in the code
// This kernel sorts the circle indices for each tile. It uses the insertion sort algorithm to arrange the circles by their indices within each tile. Sorting the circle indices allows the rendering process to process circles in a predictable order, 
// which can help in optimizing performance or achieving correct rendering results by ensuring each circle is drawn in a specific order.

// optimizedCircleRenderer 
// This kernel filters circles at the block level: each thread finds circles that might intersect its tile, 
// and writes them into a shared  list using a parallel exclusive scan. Then, each thread renders 
// its pixel using only the filtered circles. The function uses shared memory heavily, avoids unnecessary 
// checks, and unrolls loops for speed. Works especially well when only a small fraction of circles affect each tile.

// render 
// Sets up the grid and block dimensions, then launches `optimizedCircleRenderer()`.
// This version no longer uses the tile-counting, circle-filling, or sorting kernels—it does everything in a single optimized step.

// render (later changed)commented out in the code
// This is the main function responsible for managing the rendering process. It performs several steps:

// Tile Count Calculation: It calculates how many tiles are needed to cover the entire image.

// Memory Allocation: It allocates memory on the device  for the tile structure arrays d_tileCount, d_tileStart, d_tileFill, d_tileCircleIndices.

// Kernel Launches: It launches three main kernels kernelCountTileOverlaps, kernelFillTileCircleIndices, kernelSortTileLists to count the overlaps, fill the circle indices in each tile, and sort the indices for rendering.

// Rendering: Finally, it launches the kernelRenderPixelsPerTile to render the final image by drawing the circles onto the image.

