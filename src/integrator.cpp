#include "rdr/integrator.h"

#include <omp.h>

#include "rdr/bsdf.h"
#include "rdr/camera.h"
#include "rdr/canary.h"
#include "rdr/film.h"
#include "rdr/halton.h"
#include "rdr/interaction.h"
#include "rdr/light.h"
#include "rdr/math_aliases.h"
#include "rdr/math_utils.h"
#include "rdr/platform.h"
#include "rdr/properties.h"
#include "rdr/ray.h"
#include "rdr/scene.h"
#include "rdr/sdtree.h"

RDR_NAMESPACE_BEGIN

/* ===================================================================== *
 *
 * Intersection Test Integrator's Implementation
 *
 * ===================================================================== */

Vec3f IntersectionTestIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction, Sampler &sampler) const {

    const Vec3f light_origin = {-0.25f, 1.98f, -0.25f};
    const Vec3f light_edge1 = {0.5f, 0.0f, 0.0f};
    const Vec3f light_edge2 = {0.0f, 0.0f, 0.5f};
    const Vec3f light_radiance = {17.0f, 12.0f, 4.0f};
    
    const Vec3f light_normal = Normalize(Cross(light_edge1, light_edge2));
    const Float light_area = Norm(Cross(light_edge1, light_edge2));
    const Float light_pdf = 1.0f / light_area;

    const Vec2f &rand_sample = sampler.get2D();
    const Vec3f point_on_light = light_origin + rand_sample.x * light_edge1 + rand_sample.y * light_edge2;

    const Vec3f to_light_dir = Normalize(point_on_light - interaction.p);
    const Float dist_sq = Dot(point_on_light - interaction.p, point_on_light - interaction.p);
    const Float dist = std::sqrt(dist_sq);

    auto shadow_ray = DifferentialRay(interaction.p, to_light_dir, RAY_DEFAULT_MIN, dist - RAY_DEFAULT_MIN);
    SurfaceInteraction shadow_interaction;
    if (scene->intersect(shadow_ray, shadow_interaction)) {
        return Vec3f(0.0f);
    }

    const Vec3f Le = light_radiance;

    interaction.wi = to_light_dir;
    const Vec3f fr = interaction.bsdf->evaluate(interaction);
    const Float cos_theta_at_surface = std::abs(Dot(interaction.normal, to_light_dir));
    const Float cos_theta_at_light = std::abs(Dot(light_normal, -to_light_dir));
    const Float G = cos_theta_at_surface * cos_theta_at_light / dist_sq;
    
    const Float pdf = light_pdf;

    return Le * fr * G / pdf;
}

void IntersectionTestIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // Statistics
  std::atomic<int> cnt = 0;

  const Vec2i &resolution = camera->getFilm()->getResolution();
#pragma omp parallel for schedule(dynamic)
  for (int dx = 0; dx < resolution.x; dx++) {
    ++cnt;
    if (cnt % (resolution.x / 10) == 0)
      Info_("Rendering: {:.02f}%", cnt * 100.0 / resolution.x);
    Sampler sampler;
    for (int dy = 0; dy < resolution.y; dy++) {
      sampler.setPixelIndex2D(Vec2i(dx, dy));
      for (int sample = 0; sample < spp; sample++) {
        // TODO(HW3): generate #spp rays for each pixel and use Monte Carlo
        // integration to compute radiance.
        //
        // Useful Functions:
        //
        // @see Sampler::getPixelSample for getting the current pixel sample
        // as Vec2f.
        //
        // @see Camera::generateDifferentialRay for generating rays given
        // pixel sample positions as 2 floats.

        // You should assign the following two variables
        // const Vec2f &pixel_sample = ...
        // auto ray = ...

        const Vec2f &pixel_sample = sampler.getPixelSample();
        auto ray = camera->generateDifferentialRay(pixel_sample.x, pixel_sample.y);
        // After you assign pixel_sample and ray, you can uncomment the
        // following lines to accumulate the radiance to the film.
        //
        //
        // Accumulate radiance
        assert(pixel_sample.x >= dx && pixel_sample.x <= dx + 1);
        assert(pixel_sample.y >= dy && pixel_sample.y <= dy + 1);
        const Vec3f &L = Li(scene, ray, sampler);
        camera->getFilm()->commitSample(pixel_sample, L);
      }
    }
  }
}

Vec3f IntersectionTestIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
    Vec3f color(0.0);

    // Cast a ray until we hit a non-specular surface or miss
    // Record whether we have found a diffuse surface
    bool diffuse_found = false;
    SurfaceInteraction interaction;

    for (int i = 0; i < max_depth; ++i) {
        interaction = SurfaceInteraction();
        bool intersected = scene->intersect(ray, interaction);
        // Info_("Iter %d: Intersection found at p = %s", i, interaction.p);
        // if (interaction.bsdf == nullptr) {
        //     Info_("  -> CRITICAL ERROR: BSDF pointer is NULL!");
        // } //else {
        //     Info_("  -> BSDF pointer is valid.");
        //     if (dynamic_cast<const IdealDiffusion *>(interaction.bsdf)) {
        //         Info_("  -> Material is IdealDiffusion.");
        //     } else if (dynamic_cast<const PerfectRefraction
        //     *>(interaction.bsdf)) {
        //         Info_("  -> Material is PerfectRefraction.");
        //     } else {
        //         Info_("  -> Material is of an UNKNOWN type!");
        //     }
        // }
        // Perform RTTI to determine the type of the surface
        bool is_ideal_diffuse =
            dynamic_cast<const IdealDiffusion*>(interaction.bsdf) != nullptr;
        bool is_perfect_refraction =
            dynamic_cast<const PerfectRefraction*>(interaction.bsdf) != nullptr;

        // Set the outgoing direction
        interaction.wo = -ray.direction;

        if (!intersected) {
            break;
        }

        if (is_perfect_refraction) {
            // We should follow the specular direction
            // TODO(HW3): call the interaction.bsdf->sample to get the new
            // direction and update the ray accordingly.
            //
            // Useful Functions:
            // @see BSDF::sample
            // @see SurfaceInteraction::spawnRay
            //
            // You should update ray = ... with the spawned ray
            Float pdf;
            interaction.bsdf->sample(interaction, sampler, &pdf);
            ray = interaction.spawnRay(interaction.wi);
            continue;
        }

        if (is_ideal_diffuse) {
            // We only consider diffuse surfaces for direct lighting
            diffuse_found = true;
            break;
        }

        // We simply omit any other types of surfaces
        break;
    }

  if (!diffuse_found) {
    return color;
  }

  // color = directLighting(scene, interaction);
  
  color = directLighting(scene, interaction, sampler);
  return color;
}

Vec3f IntersectionTestIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction) const {
  Vec3f accumulated_color(0, 0, 0);

  Float dist_to_light = Norm(point_light_position - interaction.p);
  Vec3f light_dir     = Normalize(point_light_position - interaction.p);
  auto test_ray       = DifferentialRay(interaction.p, light_dir, RAY_DEFAULT_MIN, dist_to_light);

  // TODO(HW3): Test for occlusion
  //
  // You should test if there is any intersection between interaction.p and
  // point_light_position using scene->intersect. If so, return an occluded
  // color. (or Vec3f color(0, 0, 0) to be specific)
  //
  // You may find the following variables useful:
  //
  // @see bool Scene::intersect(const Ray &ray, SurfaceInteraction &interaction)
  //    This function tests whether the ray intersects with any geometry in the
  //    scene. And if so, it returns true and fills the interaction with the
  //    intersection information.
  //
  //    You can use iteraction.p to get the intersection position.
  //
  if (scene->isBlocked(test_ray, interaction) && (Norm(interaction.p - test_ray.origin) < dist_to_light)) {
    return Vec3f(0.0f, 0.0f, 0.0f);
  }

  // Not occluded, compute the contribution using perfect diffuse diffuse model
  // Perform a quick and dirty check to determine whether the BSDF is ideal
  // diffuse by RTTI
  const BSDF *bsdf      = interaction.bsdf;
  bool is_ideal_diffuse = dynamic_cast<const IdealDiffusion *>(bsdf) != nullptr;

  if (bsdf != nullptr && is_ideal_diffuse) {
  // if (false) {
    // TODO(HW3): Compute the contribution
    //
    // You can use bsdf->evaluate(interaction) * cos_theta to approximate the
    // albedo. In this homework, we do not need to consider a
    // radiometry-accurate model, so a simple phong-shading-like model is can be
    // used to determine the value of color.

    // The angle between light direction and surface normal
    Float cos_theta =
        std::max(Dot(light_dir, interaction.normal), 0.0f);  // one-sided
    // Float cos_theta = std::abs(Dot(light_dir, interaction.normal));

    // You should assign the value to color
    // color = ...
    // color = bsdf->evaluate(interaction) * cos_theta;
    auto albedo = bsdf->evaluate(interaction) * cos_theta;
    accumulated_color += albedo * point_light_flux  / (4 * PI * dist_to_light * dist_to_light);
  }

  const Vec3f point_light_position_2 = {0.0f, 0.0f, 5.0f};

  const Float dist_to_light_2 = Norm(point_light_position_2 - interaction.p);
  const Vec3f light_dir_2     = Normalize(point_light_position_2 - interaction.p);
  
  auto shadow_ray_2 = DifferentialRay(interaction.p, light_dir_2, RAY_DEFAULT_MIN, dist_to_light_2 - RAY_DEFAULT_MIN);

  SurfaceInteraction shadow_interaction_2;

  if (!scene->intersect(shadow_ray_2, shadow_interaction_2)) {
      const BSDF *bsdf = interaction.bsdf;
      if (dynamic_cast<const IdealDiffusion *>(bsdf) != nullptr) {
          const Float cos_theta_2 = std::max(Dot(light_dir_2, interaction.normal), 0.0f);
          const auto albedo = bsdf->evaluate(interaction) * cos_theta_2;
          accumulated_color += albedo * point_light_flux / (4 * PI * dist_to_light_2 * dist_to_light_2);
      }
  }

  return accumulated_color;
}

/* ===================================================================== *
 *
 * Path Integrator's Implementation
 *
 * ===================================================================== */

void PathIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f PathIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f PathIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

/* ===================================================================== *
 *
 * New Integrator's Implementation
 *
 * ===================================================================== */

// Instantiate template
// clang-format off
template Vec3f
IncrementalPathIntegrator::Li<Path>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
template Vec3f
IncrementalPathIntegrator::Li<PathImmediate>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
// clang-format on

// This is exactly a way to separate dec and def
template <typename PathType>
Vec3f IncrementalPathIntegrator::Li(  // NOLINT
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

RDR_NAMESPACE_END
