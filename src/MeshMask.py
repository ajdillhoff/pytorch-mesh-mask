import torch
import compute_mesh_mask, compute_mesh_mask_cuda


class MeshMaskFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vertices, triangles, num_triangles, image_height, image_width):

        if vertices.device == torch.device('cpu'):
            output = compute_mesh_mask.forward(vertices,
                                               triangles,
                                               num_triangles,
                                               image_width,
                                               image_height)
        else:
            output = compute_mesh_mask_cuda.forward(vertices,
                                                    triangles,
                                                    num_triangles,
                                                    image_width,
                                                    image_height)

        return output[0]


class MeshMask(torch.nn.Module):
    """Computes a binary mask indicating which vertices are visible."""

    def __init__(self, image_size, triangles):
        """Initialize weights and other parameters."""
        super(MeshMask, self).__init__()
        self.image_size = image_size
        self.triangles = triangles
        self.num_triangles = triangles.shape[0]

    def forward(self, vertices):
        """Forward pass through the module.

        Args:
            vertices: (N, 3) Tensor of mesh vertices.
            triangles: (N, 3) Tensor of triangle IDs.

        Returns:
            mask: (N, V) Vertex mask of viewable vertices.
        """

        mask = MeshMaskFunction.apply(vertices, self.triangles,
                                      self.num_triangles,
                                      self.image_size[0],
                                      self.image_size[1])

        return mask
