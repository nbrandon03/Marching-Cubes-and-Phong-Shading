#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include <string>
#include <cmath>
#include <algorithm>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "TriTable.hpp"

// Basic mesh data structure used for CPU-side generation and PLY export.
struct Mesh {
    std::vector<float> vertices;
    std::vector<float> normals;
};

// GPU buffer handles and counts for rendering.
struct GLMesh {
    GLuint vao = 0;
    GLuint vboPositions = 0;
    GLuint vboNormals = 0;
    GLsizei vertexCount = 0;
};

// GPU buffer handles and counts for rendering line sets (bounding box, axes).
struct GLLineSet {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLsizei vertexCount = 0;
};

// Bonus:
// Instead of generating the entire mesh up-front, we keep marching progress here
// and advance a small chunk each frame so the surface appears to grow live.
struct MarchingRuntime {
    std::function<float(float, float, float)> field;

    float isoValue = 0.0f;
    float minBound = -3.0f;
    float maxBound = 3.0f;
    float stepSize = 0.15f;

    int xCount = 0;
    int yCount = 0;
    int zCount = 0;
    int totalCubes = 0;
    int nextCube = 0;

    bool finished = false;
    bool gpuDirty = false;

    std::vector<float> vertices;
    std::vector<float> normals;
};

// Precompute the mesh on the CPU using marching cubses, returning a flat list of triangle vertices.
struct AppState {
    GLFWwindow* window = nullptr;

    float minBound = -3.0f;
    float maxBound = 3.0f;
    float stepSize = 0.15f;
    float isoValue = 0.0f;

    float r = 0.0f;
    float theta = 0.0f;
    float phi = 0.0f;

    bool dragging = false;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;

    GLuint meshProgram = 0;
    GLuint lineProgram = 0;

    GLMesh mesh;
    GLLineSet box;
    GLLineSet axes;

    glm::vec3 meshColor = glm::vec3(0.0f, 0.85f, 0.85f); // Base color for the mesh; used as a simple diffuse term in the shader.
    glm::vec3 lightDir = glm::normalize(glm::vec3(-1.0f, 1.0f, 1.0f)); // World-space direction to the light for simple diffuse shading.
};

// Global application state.
static AppState gApp;

// Cube corner offsets from the base corner, used for iterating over cube vertices.
static const int cornerOffset[8][3] = {
    {0, 0, 0},
    {1, 0, 0},
    {1, 1, 0},
    {0, 1, 0},
    {0, 0, 1},
    {1, 0, 1},
    {1, 1, 1},
    {0, 1, 1}
};

// For each of the 12 edges of the cube, the indices of the two corners it connects.
static const int edgeToCorners[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},
    {4, 5}, {5, 6}, {6, 7}, {7, 4},
    {0, 4}, {1, 5}, {2, 6}, {3, 7}
};

// Scalar field definitions.
float scalarField1(float x, float y, float z) {
    return y - std::sin(x) * std::cos(z);
}

// Scalr field2 definition, more complex with a quadratic component and a linear bias in z, producing a different shape.
float scalarField2(float x, float y, float z) {
    return x * x - y * y - z * z - z;
}

// Linear interpolation of isosurface intersection along an edge, with biasing to handle exact hits and avoid ambiguities.
glm::vec3 interpolateVertex(const glm::vec3& p1, const glm::vec3& p2,
                            float v1, float v2, float isoValue) {
    const float eps = 1e-6f; 

    // Bias logic to handle isovalue hits and nearhits
    if (std::fabs(isoValue - v1) < eps) return p1;
    if (std::fabs(isoValue - v2) < eps) return p2;
    if (std::fabs(v1 - v2) < eps) return p1;

    float t = (isoValue - v1) / (v2 - v1); // Compute interpolation factor along the edge
    t = std::clamp(t, 0.0f, 1.0f); // Ensure t is within [0,1] to avoid extrapolation
    return p1 + t * (p2 - p1); // Return the interpolated position of the isosurface intersection along the edge
}

float biasSampleNearIso(float value, float isoValue) {
    // Small fix:
    // exact iso hits can produce ambiguous cube cases and tiny pinholes.
    // bias values extremely close to the isovalue to one side consistently.
    const float eps = 1e-6f;
    if (std::fabs(value - isoValue) <= eps) {
        return isoValue - eps;
    }
    return value;
}

// Main marching cubes function: iterate over the scalar field in a grid, determine cube configurations, and emit triangles.
int gridCellCount(float minValue, float maxValue, float stepSize) {
    int count = 0;
    for (float value = minValue; value < maxValue; value += stepSize) {
        ++count;
    }
    return count;
}

void appendTriangleWithNormal(const glm::vec3& v0,
                              const glm::vec3& v1,
                              const glm::vec3& v2,
                              std::vector<float>& vertices,
                              std::vector<float>& normals) {
    // Helper used by the bonus path:
    // append one triangle and immediately append its per-vertex normals,
    // so GPU buffers can be refreshed directly without a full recompute.
    vertices.push_back(v0.x); // v0, v1, v2 order chosen to produce consisten winding and outward facing normals with given cross product order
    vertices.push_back(v0.y);
    vertices.push_back(v0.z);

    vertices.push_back(v1.x);
    vertices.push_back(v1.y);
    vertices.push_back(v1.z);

    vertices.push_back(v2.x);
    vertices.push_back(v2.y);
    vertices.push_back(v2.z);

    glm::vec3 edge1 = v1 - v0; 
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 normal = glm::cross(edge1, edge2);

    // Handle degenerate triangles by assigning an arbitrary normal. This prevents NaNs in the output which can cause rendering issues.
    if (glm::length(normal) < 1e-8f) {
        normal = glm::vec3(0.0f, 1.0f, 0.0f);
    } else {
        normal = glm::normalize(normal);
    }

    // Append the same normal for each vertex of the triangle.
    for (int i = 0; i < 3; ++i) {
        normals.push_back(normal.x);
        normals.push_back(normal.y);
        normals.push_back(normal.z);
    }
}

// Bonus: create runtime object to keep marching state and advance it incrementally each frame.
MarchingRuntime createMarchingRuntime(std::function<float(float, float, float)> field,
                                      float isoValue,
                                      float minBound,
                                      float maxBound,
                                      float stepSize) {
    // Precompute grid dimensions once, and convert marching into a linear list
    // of cubes [0..totalCubes). This makes chunked progression simple.
    MarchingRuntime runtime;
    runtime.field = std::move(field);
    runtime.isoValue = isoValue;
    runtime.minBound = minBound;
    runtime.maxBound = maxBound;
    runtime.stepSize = stepSize;

    runtime.xCount = gridCellCount(minBound, maxBound, stepSize);
    runtime.yCount = runtime.xCount;
    runtime.zCount = runtime.xCount;
    runtime.totalCubes = runtime.xCount * runtime.yCount * runtime.zCount; // Total number of cubes to process in the marching algorithm

    return runtime;
}

void processMarchingChunk(MarchingRuntime& runtime, int cubesPerFrame) {
    // Bonus: march only a subset of cubes each frame.
    // This keeps rendering interactive while geometry is still being generated.
    if (runtime.finished) return;

    bool addedAnyTriangles = false;
    for (int processed = 0; processed < cubesPerFrame && runtime.nextCube < runtime.totalCubes; ++processed) {
        // Convert linear cube id -> (x,y,z) cell index.
        int cubeIndexLinear = runtime.nextCube++;

        // Compute the 3D position of the cube's base corner from its (x,y,z) indices and the grid parameters.
        int yzCount = runtime.yCount * runtime.zCount;
        int xIdx = cubeIndexLinear / yzCount;
        int rem = cubeIndexLinear % yzCount;
        int yIdx = rem / runtime.zCount;
        int zIdx = rem % runtime.zCount; // COmpute the 3D grid indcies for current cube

        // Compute the world-space position of the cube's base corner from the grid indices and parameters.
        float x = runtime.minBound + static_cast<float>(xIdx) * runtime.stepSize;
        float y = runtime.minBound + static_cast<float>(yIdx) * runtime.stepSize;
        float z = runtime.minBound + static_cast<float>(zIdx) * runtime.stepSize;

        glm::vec3 cubeCorners[8];
        float cubeValues[8];

        // For each 8 corners of cube, compute their world space pos and sample the scalar field
        for (int i = 0; i < 8; ++i) {
            cubeCorners[i] = glm::vec3(
                x + static_cast<float>(cornerOffset[i][0]) * runtime.stepSize,
                y + static_cast<float>(cornerOffset[i][1]) * runtime.stepSize,
                z + static_cast<float>(cornerOffset[i][2]) * runtime.stepSize
            );

                // Bias samples near the isovalue to avoid ambiguous cases and ensure consistent topology.
                cubeValues[i] = biasSampleNearIso(
                    runtime.field(cubeCorners[i].x, cubeCorners[i].y, cubeCorners[i].z),
                    runtime.isoValue
                );
        }

        // Build lookup-table cube configuration mask.
        int cubeIndex = 0;
        for (int i = 0; i < 8; ++i) {
                    if (cubeValues[i] <= runtime.isoValue) {
                cubeIndex |= (1 << i);
            }
        }

        // Skip empty/full cubes that produce no geometry.
        if (cubeIndex == 0 || cubeIndex == 255) {
            continue;
        }

        // Compute the intersection vertex for each edge using linear interpolation.
        glm::vec3 edgeVertices[12];
        for (int edge = 0; edge < 12; ++edge) {
            int c0 = edgeToCorners[edge][0];
            int c1 = edgeToCorners[edge][1];

            edgeVertices[edge] = interpolateVertex(
                cubeCorners[c0],
                cubeCorners[c1],
                cubeValues[c0],
                cubeValues[c1],
                runtime.isoValue
            );
        }

        // Emit all triangles for this cube and append normals immediately.
        for (int i = 0; marching_cubes_lut[cubeIndex][i] != -1; i += 3) {
            int e0 = marching_cubes_lut[cubeIndex][i];
            int e1 = marching_cubes_lut[cubeIndex][i + 1];
            int e2 = marching_cubes_lut[cubeIndex][i + 2];

            glm::vec3 v0 = edgeVertices[e0];
            glm::vec3 v1 = edgeVertices[e1];
            glm::vec3 v2 = edgeVertices[e2];

            appendTriangleWithNormal(v0, v2, v1, runtime.vertices, runtime.normals);
            addedAnyTriangles = true;
        }
    }

    if (runtime.nextCube >= runtime.totalCubes) {
        runtime.finished = true;
    }

    if (addedAnyTriangles) {
        runtime.gpuDirty = true;
    }
}

// Marhcing cubes implementation: itereate over the scalar field in a grid, determine configs, and emit triangles
std::vector<float> marching_cubes(
    std::function<float(float, float, float)> f,
    float isovalue,
    float minValue,
    float maxValue,
    float stepsize) {

    // Vector to hold the output vertices of the generated mesh
    std::vector<float> outVertices;

    // Iterate over the 3D grid defined by the min and max bounds and step size
    for (float x = minValue; x < maxValue; x += stepsize) {
        for (float y = minValue; y < maxValue; y += stepsize) {
            for (float z = minValue; z < maxValue; z += stepsize) {
                glm::vec3 cubeCorners[8];
                float cubeValues[8];

                for (int i = 0; i < 8; ++i) {
                    cubeCorners[i] = glm::vec3(
                        x + cornerOffset[i][0] * stepsize,
                        y + cornerOffset[i][1] * stepsize,
                        z + cornerOffset[i][2] * stepsize
                    );

                        cubeValues[i] = biasSampleNearIso(
                            f(cubeCorners[i].x, cubeCorners[i].y, cubeCorners[i].z),
                            isovalue
                        );
                }
                
                // Build the cube configuration index for the marching cubes lookup table
                int cubeIndex = 0;
                for (int i = 0; i < 8; ++i) {
                        if (cubeValues[i] <= isovalue) {
                        cubeIndex |= (1 << i);
                    }
                }

                if (cubeIndex == 0 || cubeIndex == 255) {
                    continue;
                }

                // Compute the intersection vertex for each edge using linear interpolation
                glm::vec3 edgeVertices[12];
                for (int edge = 0; edge < 12; ++edge) {
                    int c0 = edgeToCorners[edge][0];
                    int c1 = edgeToCorners[edge][1];

                    edgeVertices[edge] = interpolateVertex(
                        cubeCorners[c0],
                        cubeCorners[c1],
                        cubeValues[c0],
                        cubeValues[c1],
                        isovalue
                    );
                }

                // Emit all triangles for this cube based on the marching cubes lookup table
                for (int i = 0; marching_cubes_lut[cubeIndex][i] != -1; i += 3) {
                    int e0 = marching_cubes_lut[cubeIndex][i];
                    int e1 = marching_cubes_lut[cubeIndex][i + 1];
                    int e2 = marching_cubes_lut[cubeIndex][i + 2];

                    glm::vec3 v0 = edgeVertices[e0];
                    glm::vec3 v1 = edgeVertices[e1];
                    glm::vec3 v2 = edgeVertices[e2];

                    outVertices.push_back(v0.x); // v0, v2, v1 order chosen for counter-clockwise winding and outward facing normals with given cross product order
                    outVertices.push_back(v0.y);
                    outVertices.push_back(v0.z);

                    outVertices.push_back(v2.x);
                    outVertices.push_back(v2.y);
                    outVertices.push_back(v2.z);

                    outVertices.push_back(v1.x);
                    outVertices.push_back(v1.y);
                    outVertices.push_back(v1.z);
                }
            }
        }
    }

    return outVertices;
}

// Compute normal vectors for each triangle in the mesh by taking the cross product of two edges, returning a flat list of normals corresponding to the input vertices.
std::vector<float> compute_normals(const std::vector<float>& vertices) {
    std::vector<float> normals;
    normals.reserve(vertices.size());

    // Iterate over the vertices in groups of 9 (3 vertices per triangle, each with x,y,z), compute the normal for each triangle, and append it to the normals list.
    for (size_t i = 0; i + 8 < vertices.size(); i += 9) {
        glm::vec3 v0(vertices[i],     vertices[i + 1], vertices[i + 2]);
        glm::vec3 v1(vertices[i + 3], vertices[i + 4], vertices[i + 5]);
        glm::vec3 v2(vertices[i + 6], vertices[i + 7], vertices[i + 8]);

        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        glm::vec3 normal = glm::cross(edge1, edge2);

        // Handle degenerate triangles by assigning an arbitrary normal. This prevents NaNs in the output which can cause rendering issues.
        if (glm::length(normal) < 1e-8f) {
            normal = glm::vec3(0.0f, 1.0f, 0.0f);
        } else {
            normal = glm::normalize(normal);
        }

        // Append the same normal for each vertex of the triangle.
        for (int repeat = 0; repeat < 3; ++repeat) {
            normals.push_back(normal.x);
            normals.push_back(normal.y);
            normals.push_back(normal.z);
        }
    }

    return normals;
}

// Write the generated mesh to a PLY file with vertex positions and normals, and simple triangular face definitions.
bool writePLY(const std::vector<float>& vertices,
              const std::vector<float>& normals,
              const std::string& fileName) {
    if (vertices.size() != normals.size() || vertices.size() % 9 != 0) {
        std::cerr << "PLY write failed: invalid vertex/normal data.\n";
        return false;
    }

    std::ofstream out(fileName);
    if (!out.is_open()) {
        std::cerr << "PLY write failed: could not open " << fileName << "\n";
        return false;
    }

    const size_t vertexCount = vertices.size() / 3;
    const size_t faceCount = vertexCount / 3;

    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << vertexCount << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property float nx\n";
    out << "property float ny\n";
    out << "property float nz\n";
    out << "element face " << faceCount << "\n";
    out << "property list uchar int vertex_indices\n";
    out << "end_header\n";

    for (size_t i = 0; i < vertexCount; ++i) {
        out << vertices[3 * i] << ' '
            << vertices[3 * i + 1] << ' '
            << vertices[3 * i + 2] << ' '
            << normals[3 * i] << ' '
            << normals[3 * i + 1] << ' '
            << normals[3 * i + 2] << '\n';
    }

    for (size_t i = 0; i < faceCount; ++i) {
        out << "3 " << (3 * i) << ' ' << (3 * i + 1) << ' ' << (3 * i + 2) << '\n';
    }

    std::cout << "Wrote " << fileName << " with " << faceCount << " triangles.\n";
    return true;
}

// Compile a shader of the given type (vertex or fragment) from source code, and return the shader handle. Print compilation errors if any.
GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint logLen = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
        std::string log(logLen, '\0');
        glGetShaderInfoLog(shader, logLen, nullptr, log.data());
        std::cerr << "Shader compile error:\n" << log << std::endl;
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

// Create a shader program by compiling the given vertex and fragment shader sources, attaching them to a program, and linking it. Print any compilation or linking errors.
GLuint createProgram(const char* vsSrc, const char* fsSrc) {
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
    if (!vs || !fs) return 0;

    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint logLen = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLen);
        std::string log(logLen, '\0');
        glGetProgramInfoLog(program, logLen, nullptr, log.data());
        std::cerr << "Program link error:\n" << log << std::endl;
        glDeleteProgram(program);
        program = 0;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

// Simple vertex and fragment shader sources for rendering the mesh with basic diffuse and specular lighting, and for rendering lines with a solid color.
const char* meshVertexShader = R"GLSL(
#version 330 core
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

uniform mat4 MVP;
uniform mat4 V;
uniform vec3 LightDir;

out vec3 vNormal;
out vec3 vLightDir;
out vec3 vViewDir;

void main() {
    vec4 posView = V * vec4(inPosition, 1.0);
    vec3 normalView = normalize(mat3(V) * inNormal);
    vec3 lightView = normalize(mat3(V) * LightDir);

    vNormal = normalView;
    vLightDir = lightView;
    vViewDir = normalize(-posView.xyz);

    gl_Position = MVP * vec4(inPosition, 1.0);
}
)GLSL";

// Fragment shader for the mesh, implementing a simple Blinn-Phong lighting model with ambient, diffuse, and specular components.
const char* meshFragmentShader = R"GLSL(
#version 330 core
in vec3 vNormal;
in vec3 vLightDir;
in vec3 vViewDir;

uniform vec3 modelColor;

out vec4 fragColor;

void main() {
    vec3 N = normalize(vNormal);
    if (!gl_FrontFacing) {
        N = -N;
    }
    vec3 L = normalize(vLightDir);
    vec3 V = normalize(vViewDir);
    vec3 R = reflect(-L, N);

    vec3 ambient = vec3(0.2, 0.2, 0.2);
    vec3 diffuse = max(dot(N, L), 0.0) * modelColor;

    float spec = 0.0;
    if (dot(N, L) > 0.0) {
        spec = pow(max(dot(R, V), 0.0), 64.0);
    }

    vec3 specular = spec * vec3(1.0, 1.0, 1.0);
    vec3 color = ambient + diffuse + specular;

    fragColor = vec4(color, 1.0);
}
)GLSL";

// Simple vertex shader for rendering lines, transforming vertex positions by the MVP matrix.
const char* lineVertexShader = R"GLSL(
#version 330 core
layout(location = 0) in vec3 inPosition;
uniform mat4 MVP;
void main() {
    gl_Position = MVP * vec4(inPosition, 1.0);
}
)GLSL";

// Simple fragment shader for rendering lines with a uniform color.
const char* lineFragmentShader = R"GLSL(
#version 330 core
uniform vec3 lineColor;
out vec4 fragColor;
void main() {
    fragColor = vec4(lineColor, 1.0);
}
)GLSL";

// Create GPU buffers for a line set from a flat list of vertex positions, and return the buffer handles and vertex count for rendering.
GLLineSet createLineSet(const std::vector<float>& data) {
    GLLineSet lineSet;
    lineSet.vertexCount = static_cast<GLsizei>(data.size() / 3);

    // Generate and bind a Vertex Array Object (VAO) and a Vertex Buffer Object (VBO) for the line set, upload the vertex data to the GPU, and set up the vertex attribute pointer for position.
    glGenVertexArrays(1, &lineSet.vao);
    glGenBuffers(1, &lineSet.vbo);

    glBindVertexArray(lineSet.vao);
    glBindBuffer(GL_ARRAY_BUFFER, lineSet.vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(data.size() * sizeof(float)),
                 data.data(),
                 GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glBindVertexArray(0);
    return lineSet;
}

// Create GPU buffers for the mesh from flat lists of vertex positions and normals, and return the buffer handles and vertex count for rendering.
GLMesh createMeshBuffers(const std::vector<float>& vertices,
                         const std::vector<float>& normals) {
    GLMesh mesh;
    mesh.vertexCount = static_cast<GLsizei>(vertices.size() / 3);

    // VAO and VBO setup for the mesh
    glGenVertexArrays(1, &mesh.vao);
    glGenBuffers(1, &mesh.vboPositions);
    glGenBuffers(1, &mesh.vboNormals);

    glBindVertexArray(mesh.vao);

    // Upload vertex positions to the GPU and set up the vertex attribute pointer for position.
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vboPositions);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(vertices.size() * sizeof(float)),
                 vertices.data(),
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, mesh.vboNormals);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(normals.size() * sizeof(float)),
                 normals.data(),
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glBindVertexArray(0);
    return mesh;
}

// Update GPU buffers for mesh vertex pos and normals wiht new data
void updateMeshBuffers(GLMesh& mesh,
                       const std::vector<float>& vertices,
                       const std::vector<float>& normals) {
    // Upload latest generated geometry each frame the CPU data changed.
    // Using GL_DYNAMIC_DRAW because the mesh grows over time.
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vboPositions);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(vertices.size() * sizeof(float)),
                 vertices.empty() ? nullptr : vertices.data(),
                 GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, mesh.vboNormals);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(normals.size() * sizeof(float)),
                 normals.empty() ? nullptr : normals.data(),
                 GL_DYNAMIC_DRAW);

    mesh.vertexCount = static_cast<GLsizei>(vertices.size() / 3);
}

// Build line vertex data for a bounding box defined by the given min and max values, returning a flat list of line vertex positions for rendering.
std::vector<float> buildBoundingBoxLines(float minValue, float maxValue) {
    const glm::vec3 p000(minValue, minValue, minValue);
    const glm::vec3 p100(maxValue, minValue, minValue);
    const glm::vec3 p110(maxValue, maxValue, minValue);
    const glm::vec3 p010(minValue, maxValue, minValue);
    const glm::vec3 p001(minValue, minValue, maxValue);
    const glm::vec3 p101(maxValue, minValue, maxValue);
    const glm::vec3 p111(maxValue, maxValue, maxValue);
    const glm::vec3 p011(minValue, maxValue, maxValue);

    // Define the 12 edges of the cube as pairs of corner vertices.
    const glm::vec3 edges[][2] = {
        {p000, p100}, {p100, p110}, {p110, p010}, {p010, p000},
        {p001, p101}, {p101, p111}, {p111, p011}, {p011, p001},
        {p000, p001}, {p100, p101}, {p110, p111}, {p010, p011}
    };

    // Flatten the edge vertex pairs into a single list of vertex positions for rendering as lines.
    std::vector<float> lines;
    for (const auto& edge : edges) {
        lines.push_back(edge[0].x); lines.push_back(edge[0].y); lines.push_back(edge[0].z);
        lines.push_back(edge[1].x); lines.push_back(edge[1].y); lines.push_back(edge[1].z);
    }
    return lines;
}

// Build line vertex data for the three axis lines from the min and max values.
std::vector<float> buildAxesLines(float minValue, float maxValue) {
    float len = maxValue - minValue;
    glm::vec3 start(minValue, minValue, minValue);
    glm::vec3 xEnd = start + glm::vec3(len, 0.0f, 0.0f);
    glm::vec3 yEnd = start + glm::vec3(0.0f, len, 0.0f);
    glm::vec3 zEnd = start + glm::vec3(0.0f, 0.0f, len);

    // Flatten the axis lines into a single list of vertex positions for rendering as lines.
    std::vector<float> lines = {
        start.x, start.y, start.z, xEnd.x, xEnd.y, xEnd.z,
        start.x, start.y, start.z, yEnd.x, yEnd.y, yEnd.z,
        start.x, start.y, start.z, zEnd.x, zEnd.y, zEnd.z
    };

    return lines;
}

// Compute camera pos in world space from spherical coords (r, theta, phi) for orbiting around origin
glm::vec3 cameraPositionFromSpherical(float r, float theta, float phi) {
    float sinPhi = std::sin(phi);
    return glm::vec3(
        r * sinPhi * std::cos(theta),
        r * std::cos(phi),
        r * sinPhi * std::sin(theta)
    );
}

// GLFW input callbacks for mouse and keyboard interaction to control camera orbiting and zooming, and to close the window on escape key.
void mouseButtonCallback(GLFWwindow*, int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            gApp.dragging = true;
            glfwGetCursorPos(gApp.window, &gApp.lastMouseX, &gApp.lastMouseY);
        } else if (action == GLFW_RELEASE) {
            gApp.dragging = false;
        }
    }
}

// Update camera angles base on mouse movement when dragging.
void cursorPositionCallback(GLFWwindow*, double xpos, double ypos) {
    if (!gApp.dragging) return;

    double dx = xpos - gApp.lastMouseX;
    double dy = ypos - gApp.lastMouseY;
    gApp.lastMouseX = xpos;
    gApp.lastMouseY = ypos;

    const float sensitivity = 0.008f;
    gApp.theta += static_cast<float>(dx) * sensitivity;
    gApp.phi += static_cast<float>(dy) * sensitivity;

    const float eps = 0.05f;
    gApp.phi = std::clamp(gApp.phi, eps, 3.1415926535f - eps);
}

// Use up/down arrow keys to zoom in/out by adjusting the camera radius, and escape key to close the window.
void keyCallback(GLFWwindow*, int key, int, int action, int) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

    if (key == GLFW_KEY_UP) {
        gApp.r = std::max(0.3f, gApp.r - 0.2f);
    } else if (key == GLFW_KEY_DOWN) {
        gApp.r += 0.2f;
    } else if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(gApp.window, GLFW_TRUE);
    }
}

// Clean up GPU resources for the mesh and line sets by deleting buffers and vertex arrays, and resetting the structures to default states.
void destroyMesh(GLMesh& mesh) {
    if (mesh.vboNormals) glDeleteBuffers(1, &mesh.vboNormals);
    if (mesh.vboPositions) glDeleteBuffers(1, &mesh.vboPositions);
    if (mesh.vao) glDeleteVertexArrays(1, &mesh.vao);
    mesh = GLMesh{};
}

// Clean up GPU resources for the line set by deleting buffers and vertex arrays, and resetting the structure to default state.
void destroyLineSet(GLLineSet& lineSet) {
    if (lineSet.vbo) glDeleteBuffers(1, &lineSet.vbo);
    if (lineSet.vao) glDeleteVertexArrays(1, &lineSet.vao);
    lineSet = GLLineSet{};
}

// Initialize GLFW, create a window and OpenGL context, set up input callbacks, initialize GLEW, and configure OpenGL state for depth testing and multisampling.
bool initOpenGL() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW.\n";
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    gApp.window = glfwCreateWindow(1200, 900, "CS3388 Assignment 5 - Marching Cubes", nullptr, nullptr);
    if (!gApp.window) {
        std::cerr << "Failed to create GLFW window.\n";
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(gApp.window);
    glfwSetMouseButtonCallback(gApp.window, mouseButtonCallback);
    glfwSetCursorPosCallback(gApp.window, cursorPositionCallback);
    glfwSetKeyCallback(gApp.window, keyCallback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW.\n";
        glfwDestroyWindow(gApp.window);
        glfwTerminate();
        return false;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glClearColor(0.07f, 0.08f, 0.11f, 1.0f);
    return true;
}

// Bonus: main render loop that advances the marching cubees incrementally each frame.
void renderLoop(MarchingRuntime& runtime) {
    bool reportedDone = false;

    while (!glfwWindowShouldClose(gApp.window)) {
        // Advance marching by a fixed budget per frame.
        // Increase this for faster generation; decrease for a slower growth effect.
        processMarchingChunk(runtime, 300);
        if (runtime.gpuDirty) {
            // Only re-upload when new triangles were created.
            updateMeshBuffers(gApp.mesh, runtime.vertices, runtime.normals);
            runtime.gpuDirty = false;
        }
        if (runtime.finished && !reportedDone) {
            std::cout << "Marching complete: " << (runtime.vertices.size() / 9) << " triangles generated.\n";
            reportedDone = true;
        }

        int width = 1;
        int height = 1;
        glfwGetFramebufferSize(gApp.window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Compute camera position from spherical coordinates, and build view and projection matrices for rendering.
        glm::vec3 eye = cameraPositionFromSpherical(gApp.r, gApp.theta, gApp.phi);
        glm::mat4 V = glm::lookAt(eye, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 P = glm::perspective(glm::radians(45.0f),
                                       static_cast<float>(width) / static_cast<float>(height),
                                       0.1f,
                                       100.0f);
        glm::mat4 MVP = P * V;

        // Render the mesh with the mesh shader program, set uniforms for transformatuons and lighting, bind the mesh VAO, and draw the triangles.
        glUseProgram(gApp.meshProgram);
        glUniformMatrix4fv(glGetUniformLocation(gApp.meshProgram, "MVP"), 1, GL_FALSE, glm::value_ptr(MVP));
        glUniformMatrix4fv(glGetUniformLocation(gApp.meshProgram, "V"), 1, GL_FALSE, glm::value_ptr(V));
        glUniform3fv(glGetUniformLocation(gApp.meshProgram, "LightDir"), 1, glm::value_ptr(gApp.lightDir));
        glUniform3fv(glGetUniformLocation(gApp.meshProgram, "modelColor"), 1, glm::value_ptr(gApp.meshColor));

        glBindVertexArray(gApp.mesh.vao);
        glDrawArrays(GL_TRIANGLES, 0, gApp.mesh.vertexCount);
        glBindVertexArray(0);

        glUseProgram(gApp.lineProgram);
        glUniformMatrix4fv(glGetUniformLocation(gApp.lineProgram, "MVP"), 1, GL_FALSE, glm::value_ptr(MVP));

        // Render the bounding box and axes with the line shader program, setting uniform colors for each and drawing the lines.
        glUniform3f(glGetUniformLocation(gApp.lineProgram, "lineColor"), 1.0f, 1.0f, 1.0f);
        glBindVertexArray(gApp.box.vao);
        glDrawArrays(GL_LINES, 0, gApp.box.vertexCount);

        glBindVertexArray(gApp.axes.vao);
        glDepthFunc(GL_LEQUAL);

        // Set line color for each axis and draw them. X=red, Y=green, Z=blue.
        glUniform3f(glGetUniformLocation(gApp.lineProgram, "lineColor"), 1.0f, 0.0f, 0.0f);
        glDrawArrays(GL_LINES, 0, 2);

        glUniform3f(glGetUniformLocation(gApp.lineProgram, "lineColor"), 0.0f, 1.0f, 0.0f);
        glDrawArrays(GL_LINES, 2, 2);

        glUniform3f(glGetUniformLocation(gApp.lineProgram, "lineColor"), 0.0f, 0.0f, 1.0f);
        glDrawArrays(GL_LINES, 4, 2);
        glDepthFunc(GL_LESS);

        glBindVertexArray(0);

        glfwSwapBuffers(gApp.window);
        glfwPollEvents();
    }
}

// Main function: field 1 is default
int main(int argc, char** argv) {
    std::function<float(float, float, float)> field = scalarField1;
    std::string fieldName = "field1";
    gApp.isoValue = 0.0f;
    gApp.minBound = -6.3f;
    gApp.maxBound = 6.3f;
    gApp.stepSize = 0.15f;
    gApp.meshColor = glm::vec3(0.0f, 0.85f, 0.85f);

    // Field 2 function and parameters if "field2" argument is provided, otherwise defaults to field 1.
    if (argc >= 2) {
        std::string arg = argv[1];
        if (arg == "field2") {
            field = scalarField2;
            fieldName = "field2";
            gApp.isoValue = -1.5f;
            gApp.minBound = -3.0f;
            gApp.maxBound = 3.0f;
            gApp.stepSize = 0.10f;
            gApp.meshColor = glm::vec3(0.0f, 0.85f, 0.85f);
        }
    }

    // Camera spherical coords: radius, theta, phi. Start with a reasonable view of the origin where the geometry will be generated.
    glm::vec3 initialEye(5.0f, 5.0f, 5.0f);
    gApp.r = glm::length(initialEye);
    gApp.theta = std::atan2(initialEye.z, initialEye.x);
    gApp.phi = std::acos(initialEye.y / gApp.r);

    // Bonus: initialize incremental marcher state.
    MarchingRuntime runtime = createMarchingRuntime(field,
                                                    gApp.isoValue,
                                                    gApp.minBound,
                                                    gApp.maxBound,
                                                    gApp.stepSize);

    if (!initOpenGL()) {
        return 1;
    }

    // Compile shaders and create shader programs for rendering the mesh and lines. If shader compilation or program linking fails, clean up and exit.
    gApp.meshProgram = createProgram(meshVertexShader, meshFragmentShader);
    gApp.lineProgram = createProgram(lineVertexShader, lineFragmentShader);
    if (!gApp.meshProgram || !gApp.lineProgram) {
        glfwDestroyWindow(gApp.window);
        glfwTerminate();
        return 1;
    }

    // Start with empty mesh buffers; triangles are uploaded progressively.
    gApp.mesh = createMeshBuffers({}, {});
    gApp.box = createLineSet(buildBoundingBoxLines(gApp.minBound, gApp.maxBound));
    gApp.axes = createLineSet(buildAxesLines(gApp.minBound, gApp.maxBound));

    // Print user controls and instructions to the console.
    std::cout << "Controls:\n";
    std::cout << "  Left click + drag: rotate camera around origin\n";
    std::cout << "  Up arrow: move camera closer\n";
    std::cout << "  Down arrow: move camera farther\n";
    std::cout << "  ESC: quit\n";
    std::cout << "  Mesh generation runs live and grows during rendering\n";

    // Rendering now drives marching. The mesh grows while the window is open.
    renderLoop(runtime);

    // Save whatever was generated (normally full mesh once marching is finished).
    std::string plyName = fieldName + ".ply";
    if (!writePLY(runtime.vertices, runtime.normals, plyName)) {
        std::cerr << "Failed to save generated mesh to " << plyName << "\n";
    }

    destroyMesh(gApp.mesh);
    destroyLineSet(gApp.box);
    destroyLineSet(gApp.axes);

    if (gApp.meshProgram) glDeleteProgram(gApp.meshProgram);
    if (gApp.lineProgram) glDeleteProgram(gApp.lineProgram);

    glfwDestroyWindow(gApp.window);
    glfwTerminate();
    return 0;
}