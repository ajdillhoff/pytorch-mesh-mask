/* Copyright 2017 Google LLC */

/* Licensed under the Apache License, Version 2.0 (the "License"); */
/* you may not use this file except in compliance with the License. */
/* You may obtain a copy of the License at */

/*     http://www.apache.org/licenses/LICENSE-2.0 */

/* Unless required by applicable law or agreed to in writing, software */
/* distributed under the License is distributed on an "AS IS" BASIS, */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. */
/* See the License for the specific language governing permissions and */
/* limitations under the License. */

/* NOTICE: The code in this file is used under the Apache license from
 * www.github.com/google/tf_mesh_renderer/
 *
 * UPDATE: This code has been simplified to demonstrate an issue with
 * accessing tensors returned from the GPU.
 */

#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

#include <torch/extension.h>

typedef long long int64;

// Threshold for a berycentric coordinate triplet's sum, below which the
// coordinates at a pixel are deemed degenerate. Most such degenerate triplets
// of an image will be exactly zero, as this is how pixels outside the mesh are
// rendered.
constexpr float kDegenerateBarycentricCoordinatesCutoff = 0.9f;

// If the area of a triangle is very small in screen space, the corner vertices
// are approaching colinearity, and we should drop the gradient to avoid
// numerical instability (in particular, blowup, as the forward pass computation
// already only has 8 bits of precision).
constexpr float kMinimumTriangleArea = 1e-13;

// TODO: Write function signature.
// Takes the minimum of a, b, and c, rounds down, and converts to an integer
// in the range [low, high].
inline int ClampedIntegerMin(float a, float b, float c, int low, int high) {
  return std::min(
      std::max(static_cast<int>(std::floor(std::min(std::min(a, b), c))), low),
      high);
}

// TODO: Write function signature.
// Takes the maximum of a, b, and c, rounds up, and converts to an integer
// in the range [low, high].
inline int ClampedIntegerMax(float a, float b, float c, int low, int high) {
    return std::min(
            std::max(static_cast<int>(std::ceil(std::max(std::max(a, b), c))), low),
            high);
}

// TODO: Write function signature.
// Converts to fixed point with 16 fractional bits and 16 integer bits.
// Overflows for values outside of (-2^15, 2^15).
inline int FixedPoint(float f) { return static_cast<int>(f * (1 << 16)); }

// TODO: Write function signature.
// Determines whether the point p lies counter-clockwise (CCW) of a directed
// edge between vertices v0 and v1.
bool IsCCW(int v0x, int v0y, int v1x, int v1y, int px, int py) {
    int ex = v1x - v0x;
    int ey = v1y - v0y;
    int x = px - v0x;
    int y = py - v0y;
    // p is CCW of v1 - v0 if det(A) >= 0, where A:
    // | v1x - v0x, px - v0x |
    // | v1y - v0y, py - v0y |
    int64 ex_y = int64{ex} * int64{y};
    int64 ey_x = int64{ey} * int64{x};
    return ex_y >= ey_x;
}

// TODO: Write function signature.
// Determines whether the point p lies inside a triangle.
// Accepts both front-facing and back-facing triangles.
bool PixelIsInsideTriangle(int v0x, int v0y, int v1x, int v1y,
        int v2x, int v2y, int px, int py) {
    // Returns true if the point is counter clockwise to all the edges of either
    // the front facing or back facing triangle.
    return (IsCCW(v0x, v0y, v1x, v1y, px, py) &&
            IsCCW(v1x, v1y, v2x, v2y, px, py) &&
            IsCCW(v2x, v2y, v0x, v0y, px, py)) ||
        (IsCCW(v1x, v1y, v0x, v0y, px, py) &&
         IsCCW(v2x, v2y, v1x, v1y, px, py) &&
         IsCCW(v0x, v0y, v2x, v2y, px, py));
}

// TODO: Write function signature.
std::vector<torch::Tensor> compute_mesh_mask(
            torch::Tensor vertices,
            torch::Tensor triangles,
            int triangle_count,
            int image_width,
            int image_height
        ) {
    auto z_buffer = torch::ones({image_height * image_width}, vertices.type());
    auto mesh_mask = torch::zeros({vertices.size(0)}, torch::TensorOptions().dtype(torch::kUInt8));
    auto mesh_mask_data = mesh_mask.data<uint8_t>();
    auto z_buffer_data = z_buffer.data<float>();

    torch::Tensor triangles_ = triangles.contiguous();
    torch::Tensor vertices_ = vertices.contiguous();
    auto triangles_data = triangles_.data<int>();
    auto vertices_data = vertices_.data<float>();

    const float half_image_width = 0.5 * image_width;
    const float half_image_height = 0.5 * image_height;

    for (int triangle_id = 0; triangle_id < triangle_count; triangle_id++) {
        const int v0_x_id = 3 * triangles_data[3 * triangle_id];
        const int v1_x_id = 3 * triangles_data[3 * triangle_id + 1];
        const int v2_x_id = 3 * triangles_data[3 * triangle_id + 2];

        // Convert NDC vertex positions to viewport coordinates.
        const float v0x = (vertices_data[v0_x_id] + 1.0) * half_image_width;
        const float v0y = (vertices_data[v0_x_id + 1] + 1.0) * half_image_height;
        const float v1x = (vertices_data[v1_x_id] + 1.0) * half_image_width;
        const float v1y = (vertices_data[v1_x_id + 1] + 1.0) * half_image_height;
        const float v2x = (vertices_data[v2_x_id] + 1.0) * half_image_width;
        const float v2y = (vertices_data[v2_x_id + 1] + 1.0) * half_image_height;

        // Find the triangle bounding box enlarged to the nearest integer and
        // clamped to the image boundaries.
        const int left = ClampedIntegerMin(v0x, v1x, v2x, 0, image_width);
		const int right = ClampedIntegerMax(v0x, v1x, v2x, 0, image_width);
		const int bottom = ClampedIntegerMin(v0y, v1y, v2y, 0, image_height);
		const int top = ClampedIntegerMax(v0y, v1y, v2y, 0, image_height);

        // Convert coordinates to fixed-point to make triangle intersection
        // testing consistent and prevent cracks.
        const int fv0x = FixedPoint(v0x);
        const int fv0y = FixedPoint(v0y);
        const int fv1x = FixedPoint(v1x);
        const int fv1y = FixedPoint(v1y);
        const int fv2x = FixedPoint(v2x);
        const int fv2y = FixedPoint(v2y);

		// Iterate over each pixel in the bounding box.
		for (int i = bottom; i < top; i++) {
            for (int j = left; j < right; j++) {
                const float px = j + 0.5;
                const float py = i + 0.5;

                if (!PixelIsInsideTriangle(fv0x, fv0y, fv1x, fv1y, fv2x, fv2y,
                            FixedPoint(px), FixedPoint(py))) {
                    continue;
                }

                const int pixel_idx = i * image_width + j;

                // Compute twice the area of two barycentric triangles, as well
                // as the triangle they sit in; the barycentric is the ratio of
                // the triangle areas, so the factor of two does not change the
                // result.
                const float twice_triangle_area =
                    (v2x - v0x) * (v1y - v0y) - (v2y - v0y) * (v1x - v0x);
                const float b0 = ((px - v1x) * (v2y - v1y) - (py - v1y) *
                        (v2x - v1x)) / twice_triangle_area;
                const float b1 = ((px - v2x) * (v0y - v2y) - (py - v2y) *
                        (v0x - v2x)) / twice_triangle_area;
                // The three upper triangles partition the lower triangle, so we
                // can compute the third barycentric coordinate using the other
                // two.
                const float b2 = 1.0f - b0 - b1;

                const float v0z = vertices_data[v0_x_id + 2];
                const float v1z = vertices_data[v1_x_id + 2];
                const float v2z = vertices_data[v2_x_id + 2];
                const float z = b0 * v0z + b1 * v1z + b2 * v2z;

                // Skip the pixel if it is farther than the current z-buffer
                // pixel or beyond the near or far clipping plane.
                if (z < -1.0 || z > 1.0 || z > z_buffer_data[pixel_idx]) {
                    mesh_mask_data[v0_x_id / 3] = 0;
                    mesh_mask_data[v1_x_id / 3] = 0;
                    mesh_mask_data[v2_x_id / 3] = 0;
                    continue;
                }

                z_buffer[pixel_idx] = z;
                mesh_mask_data[v0_x_id / 3] = 1;
                mesh_mask_data[v1_x_id / 3] = 1;
                mesh_mask_data[v2_x_id / 3] = 1;
            }
        }
    }

    return {mesh_mask};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &compute_mesh_mask, "Compute Mesh Mask Forward");
}
