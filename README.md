# 🚀 Cuda_Renderer

An optimized CUDA-based renderer for animated scenes composed of 2D circles. Supports multiple effects including snowflakes, bouncing balls, fireworks, and hypnotic pulsations. Implements tile-based rendering with shared memory and efficient circle filtering per tile.

## 🧠 Description

This project demonstrates advanced CUDA rendering techniques by efficiently drawing and animating thousands of 2D circles on the screen using GPU acceleration. It applies animation logic and alpha blending for realistic motion and visual effects across different scenes. The renderer went through multiple optimization iterations to improve performance, especially on complex scenes like `rand100k` and `biglittle`.

## 🔧 Tech Stack

- C++
- CUDA
- Thrust (attempted but not used due to CUDA 12.6 issues)
- Makefile-based build system

## ✨ Key Features

- ✅ Tile-based CUDA circle rendering with per-tile filtering  
- ✅ Shared memory for fast access and reduced global memory load  
- ✅ Animated scenes: Snowflakes, Fireworks, Bouncing Balls, Hypnosis  
- ✅ Circle overlap filtering using bounding box checks  
- ✅ Final kernel `optimizedCircleRenderer` for max performance  
- ✅ Custom alpha blending and color lookup based on distance  

## 📂 Folder Structure

```
.
├── Makefile
├── main.cpp
├── cudaRenderer.cu / .h
├── image.h
├── noise.cpp / .h / .cu_inl
├── exclusiveScan.cu_inl
├── lookupColor.cu_inl
├── render (binary)
├── render_soln (reference binary)
├── checker.pl (testing script)
├── benchmark.cpp (perf comparison)
└── sceneLoader.cpp / .h
```

## 🛠️ Setup & Usage

### ✅ Requirements

- CUDA Toolkit 12.x  
- NVIDIA GPU with Compute Capability 3.5+  
- Linux/macOS environment with `make`  

### ⚙️ Build & Run

```bash
make
./render
```

It will load the default scene and start rendering.

To switch scenes (e.g., `SNOWFLAKES`, `FIREWORKS`, `BOUNCING_BALLS`, `HYPNOSIS`), edit `main.cpp` accordingly.

## 📸 Visual Output

Output is shown in a graphical window or saved as an image depending on how the `render` function is configured. Use `.ppm` output if rendering to file.

## 👥 Contributors

- Can Ercan (@cann-e)

