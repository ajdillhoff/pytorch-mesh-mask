#include <vector>

#include <torch/extension.h>

// CUDA forward declarations
std::vector<torch::Tensor> compute_mesh_mask_forward(
        torch::Tensor vertices,
        torch::Tensor triangles,
        int triangle_count,
        int image_width,
        int image_height);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous.")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> compute_mesh_mask_forward_cuda(
        torch::Tensor vertices,
        torch::Tensor triangles,
        int triangle_count,
        int image_width,
        int image_height) {
    CHECK_INPUT(vertices);
    CHECK_INPUT(triangles);

    return compute_mesh_mask_forward(vertices,
            triangles,
            triangle_count,
            image_width,
            image_height);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &compute_mesh_mask_forward_cuda, "Compute Mesh Mask (Forward, CUDA)");
}
