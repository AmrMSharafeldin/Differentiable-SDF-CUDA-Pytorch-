# Differentiable-SDF-CUDA-Pytorch-



# **Architecture**

There are two primary methods for representing 3D objects: explicit and implicit representations. Explicit representation typically involves triangular meshes or point clouds generated by technologies like LIDAR sensors or active stereo imaging. Implicit representation, on the other hand, describes the object's topology using a mathematical function. This function, defined for every point in space, determines whether the point lies on the surface or outside. An example of such a function is the Signed Distance Function. It's worth noting that implicit functions are guaranteed to represent manifolds.
![SDFs](Media/fig2.png)

== SDF Renderer ==
The core of the renderer is based on a signed distance function and ray marching:
* A ray marching layer, followed by surface detection and shading.
* The ray marching layer casts rays and steps through space using the SDF to find surface intersections.
* The SDF returns distances to surfaces, allowing detection of geometry and more complex shapes.
* Normal estimation, shading, and lighting layers are added to:
  * Ensure accurate light interaction.
  * Maintain visual realism.
  * Optimize gradients for geometry learning in differentiable pipelines.

