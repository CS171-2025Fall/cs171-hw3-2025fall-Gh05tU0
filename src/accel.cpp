#include "rdr/accel.h"

#include "rdr/canary.h"
#include "rdr/interaction.h"
#include "rdr/math_aliases.h"
#include "rdr/platform.h"
#include "rdr/shape.h"

RDR_NAMESPACE_BEGIN

/* ===================================================================== *
 *
 * AABB Implementations
 *
 * ===================================================================== */

bool AABB::isOverlap(const AABB &other) const {
  return ((other.low_bnd[0] >= this->low_bnd[0] &&
              other.low_bnd[0] <= this->upper_bnd[0]) ||
             (this->low_bnd[0] >= other.low_bnd[0] &&
                 this->low_bnd[0] <= other.upper_bnd[0])) &&
         ((other.low_bnd[1] >= this->low_bnd[1] &&
              other.low_bnd[1] <= this->upper_bnd[1]) ||
             (this->low_bnd[1] >= other.low_bnd[1] &&
                 this->low_bnd[1] <= other.upper_bnd[1])) &&
         ((other.low_bnd[2] >= this->low_bnd[2] &&
              other.low_bnd[2] <= this->upper_bnd[2]) ||
             (this->low_bnd[2] >= other.low_bnd[2] &&
                 this->low_bnd[2] <= other.upper_bnd[2]));
}

bool AABB::intersect(const Ray &ray, Float *t_in, Float *t_out) const {
  // TODO(HW3): implement ray intersection with AABB.
  // ray distance for two intersection points are returned by pointers.
  //
  // This method should modify t_in and t_out as the "time"
  // when the ray enters and exits the AABB respectively.
  //
  // And return true if there is an intersection, false otherwise.
  //
  // Useful Functions:
  // @see Ray::safe_inverse_direction
  //    for getting the inverse direction of the ray.
  // @see Min/Max/ReduceMin/ReduceMax
  //    for vector min/max operations.
  Vec3f inv_dir = ray.safe_inverse_direction;

  Vec3f t_near_vec = (this->low_bnd - ray.origin) * inv_dir;
  Vec3f t_far_vec  = (this->upper_bnd - ray.origin) * inv_dir;

  Vec3f t1 = Min(t_near_vec, t_far_vec);
  Vec3f t2 = Max(t_near_vec, t_far_vec);

  Float t_enter = ReduceMax(t1);

  Float t_exit = ReduceMin(t2);

  if (t_enter < t_exit + EPS && t_exit > 0) {
    if (t_in) *t_in = t_enter;
    if (t_out) *t_out = t_exit;
    return true;
  }

  return false;
}

/* ===================================================================== *
 *
 * Accelerator Implementations
 *
 * ===================================================================== */

bool TriangleIntersect(Ray &ray, const uint32_t &triangle_index,
    const ref<TriangleMeshResource> &mesh, SurfaceInteraction &interaction) {
  using InternalScalarType = Double;
  using InternalVecType    = Vec<InternalScalarType, 3>;

  AssertAllValid(ray.direction, ray.origin);
  AssertAllNormalized(ray.direction);

  const auto &vertices = mesh->vertices;
  const Vec3u v_idx(&mesh->v_indices[3 * triangle_index]);
  assert(v_idx.x < mesh->vertices.size());
  assert(v_idx.y < mesh->vertices.size());
  assert(v_idx.z < mesh->vertices.size());

  InternalVecType dir = Cast<InternalScalarType>(ray.direction);
  InternalVecType v0  = Cast<InternalScalarType>(vertices[v_idx[0]]);
  InternalVecType v1  = Cast<InternalScalarType>(vertices[v_idx[1]]);
  InternalVecType v2  = Cast<InternalScalarType>(vertices[v_idx[2]]);

  // TODO(HW3): implement ray-triangle intersection test.
  // You should compute the u, v, t as InternalScalarType
  //
  //   InternalScalarType u = ...;
  //   InternalScalarType v = ...;
  //   InternalScalarType t = ...;
  //
  // And exit early with `return false` if there is no intersection.
  //
  // The intersection points is denoted as:
  // (1 - u - v) * v0 + u * v1 + v * v2 == ray.origin + t * ray.direction
  // where the left side is the barycentric interpolation of the triangle
  // vertices, and the right side is the parametric equation of the ray.
  //
  // You should also make sure that:
  // u >= 0, v >= 0, u + v <= 1, and, ray.t_min <= t <= ray.t_max
  //
  // Useful Functions:
  // You can use @see Cross and @see Dot for determinant calculations.

  // Delete the following lines after you implement the function
  InternalVecType origin = Cast<InternalScalarType>(ray.origin);
  InternalVecType E1 = v1 - v0;
  InternalVecType E2 = v2 - v0;

  InternalVecType P = Cross(dir, E2);
  InternalScalarType det = Dot(E1, P);

  constexpr InternalScalarType epsilon = std::numeric_limits<InternalScalarType>::epsilon();
  if (det > -epsilon && det < epsilon) {
      return false;
  }

  InternalScalarType inv_det = InternalScalarType(1.0) / det;
  InternalVecType T = origin - v0;

  InternalScalarType u = Dot(T, P) * inv_det;
  if (u < 0.0 || u > 1.0) {
      return false;
  }

  InternalVecType Q = Cross(T, E1);
  InternalScalarType v = Dot(dir, Q) * inv_det;
  if (v < 0.0 || u + v > 1.0) {
      return false;
  }

  InternalScalarType t = Dot(E2, Q) * inv_det;

  if (ray.withinTimeRange(t)) {
      CalculateTriangleDifferentials(interaction,
          {static_cast<Float>(1 - u - v), static_cast<Float>(u),
              static_cast<Float>(v)},
          mesh, triangle_index);
      AssertNear(interaction.p, ray(t));
      assert(ray.withinTimeRange(t));
      ray.setTimeMax(t);
      return true;
  }

  return false;
}

void Accel::setTriangleMesh(const ref<TriangleMeshResource> &mesh) {
  // Build the bounding box
  AABB bound(Vec3f(Float_INF, Float_INF, Float_INF),
      Vec3f(Float_MINUS_INF, Float_MINUS_INF, Float_MINUS_INF));
  for (auto &vertex : mesh->vertices) {
    bound.low_bnd   = Min(bound.low_bnd, vertex);
    bound.upper_bnd = Max(bound.upper_bnd, vertex);
  }

  this->mesh  = mesh;   // set the pointer
  this->bound = bound;  // set the bounding box
}

void Accel::build() {}

AABB Accel::getBound() const {
  return bound;
}

bool Accel::intersect(Ray &ray, SurfaceInteraction &interaction) const {
  bool success = false;
  for (int i = 0; i < mesh->v_indices.size() / 3; i++)
    success |= TriangleIntersect(ray, i, mesh, interaction);
  return success;
}

RDR_NAMESPACE_END
