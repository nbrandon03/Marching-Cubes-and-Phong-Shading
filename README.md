# 3D Isosurface Visualization — Marching Cubes & Phong Shading

A C++ and OpenGL application that generates, renders, and exports 3D isosurfaces from mathematical scalar fields using the Marching Cubes algorithm. Features interactive camera controls, Phong-style lighting, incremental mesh generation, and PLY mesh export.

## Features

- **Marching Cubes Algorithm:** Generates triangle meshes from sampled 3D scalar fields.
- **Interactive 3D Rendering:** Displays generated surfaces using OpenGL.
- **Phong-Style Shading:** Implements ambient, diffuse, and specular lighting for surface illumination.
- **Incremental Mesh Generation:** Processes approximately 300 cubes per frame, allowing the mesh to grow progressively during rendering.
- **Camera Controls:** Rotate and zoom around the generated surface.
- **3D Reference Geometry:** Displays a bounding box and RGB coordinate axes.
- **PLY Export:** Saves generated meshes with vertex normals and triangle face indices for use in external 3D viewers.
- **Numerical Edge-Case Handling:** Applies a small epsilon bias to scalar field samples near the isovalue to prevent missing triangles.

## Supported Scalar Fields

The application supports two mathematical scalar fields:

| Field | Scalar Function | Isovalue |
|---|---|---|
| Field 1 | \(y - \sin(x)\cos(z)\) | 0 |
| Field 2 | \(x^2 - y^2 - z^2 - z\) | -1.5 |

The Marching Cubes algorithm samples each field across a 3D grid and interpolates intersections along cube edges to construct the resulting isosurface.

## Technologies

- **Language:** C++17
- **Graphics API:** OpenGL 3.3+
- **Windowing and Input:** GLFW
- **OpenGL Extension Loading:** GLEW
- **Mathematics:** GLM
- **Mesh Format:** ASCII PLY

## Implementation Highlights

### Marching Cubes and Mesh Generation

The implementation evaluates scalar field values at grid points, identifies cube configurations using a lookup table, and generates triangles along intersected edges. Mesh generation is incremental, processing a portion of the grid each frame rather than computing the entire surface before rendering.

### OpenGL Rendering

Generated geometry is stored in Vertex Array Objects (VAOs) and Vertex Buffer Objects (VBOs). Dynamic GPU buffers are updated as new triangles are generated, and the mesh is rendered using `glDrawArrays()`.

The shader implements Phong-style lighting with ambient, diffuse, and specular components. The specular shininess exponent is set to 64.

### Surface Normals

Triangle normals are calculated and normalized to support lighting calculations. A helper function, `compute_normals()`, is also provided for batch normal computation.

### Numerical Robustness and Rendering Fixes

- **Isovalue ambiguity:** Introduced `biasSampleNearIso()` to handle samples that fall exactly on the isovalue and reduce missing-triangle artifacts.
- **Axis visibility:** Adjusted depth comparison during axis rendering so coordinate axes remain visible when overlapping the bounding box.
- **Depth consistency:** Maintained depth testing during line rendering to preserve appropriate occlusion.

## Getting Started

### Prerequisites

Install a C++ compiler and the required OpenGL development libraries.

**Ubuntu / Debian / WSL**

```bash
sudo apt update
sudo apt install build-essential libglew-dev libglfw3-dev libglm-dev libgl1-mesa-dev
```

### Compilation

Compile the application from the directory containing `main.cpp` and `TriTable.hpp`:

```bash
g++ main.cpp -o a5 -std=c++17 -lglfw -lGLEW -lGL
```

### Running the Application

Run the default scalar field:

```bash
./a5
```

Run the second scalar field:

```bash
./a5 field2
```

The application opens an interactive window and generates the selected isosurface incrementally.

## Controls

| Input | Action |
|---|---|
| Left mouse button + drag | Rotate the camera around the origin |
| Up arrow | Move the camera closer |
| Down arrow | Move the camera farther away |
| ESC | Exit and export the generated mesh |

## Mesh Export

When the application closes, the generated mesh is exported as an ASCII PLY file:

- `field1.ply` — generated using the default scalar field.
- `field2.ply` — generated when the application is launched with the `field2` argument.

The exported files contain vertex positions, vertex normals, and triangle face indices, making them suitable for inspection in compatible 3D visualization software.

## Repository Contents

| File | Description |
|---|---|
| `main.cpp` | Marching Cubes implementation, OpenGL rendering, shaders, camera controls, and PLY export |
| `TriTable.hpp` | Lookup table used to determine triangle configurations for each cube case |
| `field1.ply` | Example mesh exported from the first scalar field |
| `field2.ply` | Example mesh exported from the second scalar field |

## Project Context

Developed as part of CS3388 — Computer Graphics, Assignment 5. The project explores procedural mesh generation, real-time 3D rendering, lighting, and numerical edge-case handling in C++ and OpenGL.
