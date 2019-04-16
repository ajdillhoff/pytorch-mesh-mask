#include <cstdio>
#include <cfloat>
#include <cmath>
#include <cassert>
#include <vector>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


typedef long long int64;

__device__ constexpr float kMinimumTriangleArea() { return 1e-13; }
__device__ constexpr float kDegenerateBarycentricCoordinatesCutoff() { return 0.9f; }

#define gpuErrorcheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

__device__ int ClampedIntegerMin(float a, float b, float c, int low, int high) {
    return min(
            max(float2int(floor(min(min(a, b), c))), low),
            high);
}

__device__ int ClampedIntegerMax(float a, float b, float c, int low, int high) {
    return min(
            max(float2int(ceil(max(max(a, b), c))), low),
            high);
}

__device__ int FixedPoint(float f) { return float2int(f * (1 << 16)); }

__device__ bool IsCCW(int v0x, int v0y, int v1x, int v1y, int px, int py) {
    int ex = v1x - v0x;
    int ey = v1y - v0y;
    int x = px - v0x;
    int y = py - v0y;
    int64 ex_y = int64{ex} * int64{y};
    int64 ey_x = int64{ey} * int64{x};
    return ex_y >= ey_x;
}

__device__ bool PixelIsInsideTriangle(int v0x, int v0y, int v1x, int v1y,
        int v2x, int v2y, int px, int py) {
    return (IsCCW(v0x, v0y, v1x, v1y, px, py) &&
            IsCCW(v1x, v1y, v2x, v2y, px, py) &&
            IsCCW(v2x, v2y, v0x, v0y, px, py)) ||
        (IsCCW(v1x, v1y, v0x, v0y, px, py) &&
         IsCCW(v2x, v2y, v1x, v1y, px, py) &&
         IsCCW(v0x, v0y, v2x, v2y, px, py));
}

__global__ void compute_mesh_mask_cuda_forward(
        float* vertices,
        int* triangles,
        float* z_buffer,
        uint8_t* mesh_mask,
        int triangle_count,
        int image_width,
        int image_height,
        int batch_size,
        int num_vertices) {

    const float half_image_width = 0.5 * image_width;
    const float half_image_height = 0.5 * image_height;

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    const float px = x + 0.5;
    const float py = y + 0.5;

    if ((x < 0 || x >= image_width) || (y < 0 || y >= image_height)) { return; }

    for (int batch_id = 0; batch_id < batch_size; batch_id++) {

        for (int triangle_id = 0; triangle_id < triangle_count; triangle_id++) {
            const int v0_x_id = (batch_id * num_vertices * 3) + 3 * triangles[3 * triangle_id];
            const int v1_x_id = (batch_id * num_vertices * 3) + 3 * triangles[3 * triangle_id + 1];
            const int v2_x_id = (batch_id * num_vertices * 3) + 3 * triangles[3 * triangle_id + 2];

            // Convert NDC vertex positions to viewport coordinates.
            const float v0x = (vertices[v0_x_id] + 1.0) * half_image_width;
            const float v0y = (vertices[v0_x_id + 1] + 1.0) * half_image_height;
            const float v1x = (vertices[v1_x_id] + 1.0) * half_image_width;
            const float v1y = (vertices[v1_x_id + 1] + 1.0) * half_image_height;
            const float v2x = (vertices[v2_x_id] + 1.0) * half_image_width;
            const float v2y = (vertices[v2_x_id + 1] + 1.0) * half_image_height;

            // Convert coordinates to fixed-point to make triangle intersection
            // testing consistent and prevent cracks.
            const int fv0x = FixedPoint(v0x);
            const int fv0y = FixedPoint(v0y);
            const int fv1x = FixedPoint(v1x);
            const int fv1y = FixedPoint(v1y);
            const int fv2x = FixedPoint(v2x);
            const int fv2y = FixedPoint(v2y);

            if (!PixelIsInsideTriangle(fv0x, fv0y, fv1x, fv1y, fv2x, fv2y,
                        FixedPoint(px), FixedPoint(py))) {
                continue;
            }

            const int pixel_idx = (batch_id * image_width * image_height) + (image_height - y) * image_height + x;

            // Compute twice the area of two barycentric triangles, as well
            // as the triangle they sit in. The barycentric is the ratio of
            // the triangle areas, so the factor of two does not change the
            // result.
            const float twice_triangle_area =
                (v2x - v0x) * (v1y - v0y) - (v2y - v0y) * (v1x - v0x);
            const float b0 = ((px - v1x) * (v2y - v1y) - (py - v1y) *
                    (v2x - v1x)) / twice_triangle_area;
            const float b1 = ((px - v2x) * (v0y - v2y) - (py - v2y) *
                    (v0x - v2x)) / twice_triangle_area;

            // The three upper triangle partition the lower triangle, so we
            // can compute the third barycentric coordinate using the other
            // two.
            const float b2 = 1.0f - b0 - b1;

            const float v0z = vertices[v0_x_id + 2];
            const float v1z = vertices[v1_x_id + 2];
            const float v2z = vertices[v2_x_id + 2];
            const float z = b0 * v0z + b1 * v1z + b2 * v2z;

            // Skip the pixel if it is farther than the current z-buffer
            // pixel or beyond the near or far clipping plane.
            if (z < -1.0 || z > 1.0 || z > z_buffer[pixel_idx]) {
                mesh_mask[v0_x_id / 3] = 0;
                mesh_mask[v1_x_id / 3] = 0;
                mesh_mask[v2_x_id / 3] = 0;
                continue;
            }

            z_buffer[pixel_idx] = z;
            mesh_mask[v0_x_id / 3] = 1;
            mesh_mask[v1_x_id / 3] = 1;
            mesh_mask[v2_x_id / 3] = 1;
        }
    }
}

std::vector<torch::Tensor> compute_mesh_mask_forward(
        torch::Tensor vertices,
        torch::Tensor triangles,
        int triangle_count,
        int image_width,
        int image_height) {
    const int batch_size = vertices.size(0);
    const int num_vertices = vertices.size(1);
    auto z_buffer = torch::ones({batch_size, image_height * image_width}, vertices.type());
    auto mesh_mask = torch::zeros({batch_size, num_vertices},
            torch::TensorOptions().dtype(torch::kUInt8).device({torch::kCUDA}));

    dim3 block;
    block.x = 32;
    block.y = 32;
    dim3 grid;
    grid.x = (image_width + block.x - 1) / block.x;
    grid.y = (image_height + block.y - 1) / block.y;

    compute_mesh_mask_cuda_forward<<<grid, block>>>(
            vertices.data<float>(),
            triangles.data<int>(),
            z_buffer.data<float>(),
            mesh_mask.data<uint8_t>(),
            triangle_count,
            image_width,
            image_height,
            batch_size,
            vertices.size(1));

    gpuErrorcheck(cudaPeekAtLastError());

    return {mesh_mask};
}
