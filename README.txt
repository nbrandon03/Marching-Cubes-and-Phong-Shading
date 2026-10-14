CS3388 - Assignment 5
Marching Cubes and Phong Shading

Files Included:
- main.cpp
A C++ OpenGL program that implements marching cubes on scalar fields, renders
the result with a Phong-like shader, supports camera interaction, and exports
the generated mesh to PLY. Approximately 800 lines.

- TriTable.hpp
Provided lookup table used by marching cubes to determine triangle
configurations for each cube case.

- Generated PLY Files
- field1.ply
- field2.ply

Description:
This program generates and renders triangle meshes using marching cubes
applied to scalar fields. The mesh is constructed by evaluating the scalar
field over a 3D grid and extracting an isosurface based on a given isovalue.

Two scalar fields are supported:
y - sin(x)*cos(z), isovalue 0
x^2 - y^2 - z^2 - z, isovalue -1.5

The generated mesh is rendered using OpenGL with a Phong-like lighting model
that includes ambient, diffuse, and specular components. The program writes
the generated mesh data (vertices and normals) to a valid PLY file.

A bounding box and coordinate axes are drawn to visualize the marching volume
and orientation in 3D space.

Design:
Marching cubes sampling and edge interpolation are implemented in main.cpp.
The code supports incremental mesh generation where a chunk of cubes is
processed per frame, so the mesh visibly grows during rendering rather than
blocking on full computation upfront.

In incremental mode, runtime state (grid position, field parameters, vertices)
is tracked and advanced via processMarchingChunk(). Each frame, approximately
300 cubes are processed. Newly generated vertices and normals are uploaded to
dynamic GPU buffers (GL_DYNAMIC_DRAW) so changes appear immediately.

Normals are generated per triangle and normalized, with the same triangle
normal repeated for all three vertices to enable smooth shading. The code
also includes a compute_normals() helper for batch workflows.

The mesh is stored in vertex buffers and rendered using Vertex Array Objects
(VAOs) and Vertex Buffer Objects (VBOs). Rendering is done using glDrawArrays().

A Phong-like shader is implemented:
Vertex shader uses uniforms MVP, V, and LightDir
Fragment shader uses uniform modelColor
Ambient color is (0.2, 0.2, 0.2)
Specular color is (1, 1, 1)
Shininess is 64

Technical Improvements:

(1) Pinhole artifacts in field1:
Early testing revealed missing triangles ("pinholes") in field1. The root cause
was ambiguous cube configurations when the scalar field evaluated to exactly
the isovalue at grid points. The solution adds biasSampleNearIso() which shifts
field samples by epsilon (1e-6) when they land exactly on the isovalue. This
resolves the ambiguity while preserving surface smoothness.

(2) Axes visibility:
The RGB axes (red X, green Y, blue Z) were initially invisible because they share
exact edge geometry with the bounding box. With GL_LESS depth testing, equal-depth
axis fragments were discarded. The fix uses GL_LEQUAL for axis drawing to allow
equal-depth passes through, restoring visibility while maintaining proper occlusion
from the mesh.

(3) Depth test consistency:
Initial code unnecessarily disabled depth testing before line drawing. This was
removed to ensure consistent occlusion throughout rendering.

Bonus Implemented:
The mesh grows incrementally during rendering instead of showing only after full
marching is complete. Each frame advances about 300 cubes, appending geometry to
GPU buffers. Users see smooth growth over a few seconds rather than a blockade wait.

PLY Output:
When the window closes, the generated mesh data is written to:
- field1.ply (default run)
- field2.ply (when launched with argument: field2)

Each file contains vertices with per-vertex normals and triangle face indices in
ASCII PLY format, suitable for import into 3D viewers and analysis tools.

Controls:
- Left click + drag: rotate camera around origin
- Up arrow: move camera closer
- Down arrow: move camera farther
- ESC: quit and export PLY file

How to Compile:
g++ main.cpp -o a5 -std=c++17 -lglfw -lGLEW -lGL

Required libraries (install if missing):
- GLFW 3: windowing and input handling
- GLEW: OpenGL extension loader
- GLM: linear algebra
- OpenGL 3.3+: core graphics API

On Linux/WSL (Ubuntu/Debian):
sudo apt install libglew-dev libglfw3-dev libglm-dev libgl1-mesa-dev

On macOS with Homebrew:
brew install glew glfw glm
Then compile with: g++ main.cpp -o a5 -std=c++17 -lglfw -lGLEW -framework OpenGL

If compilation fails with "cannot find" errors, verify all dev packages are
installed and check your library paths. The command above handles most systems.

How to Run:
./a5
Runs the program using the first scalar field (field1)

./a5 field2
Runs the program using the second scalar field

Example:
./a5
./a5 field2

Code Quality Notes:
- Inline comments explain key sections and algorithms
- Functions named descriptively (processMarchingChunk, biasSampleNearIso, etc.)
- No compiler warnings under C++17
- Robust handling of edge cases (epsilon bias, degenerate triangles)