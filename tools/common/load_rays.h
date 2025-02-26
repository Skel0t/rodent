#ifndef LOAD_RAYS_H
#define LOAD_RAYS_H

#include <fstream>
#include <anydsl_runtime.hpp>
#include "traversal.h"

template <typename Ray>
struct RayTraits {};

template <>
struct RayTraits<Ray1> {
    enum { RayPerPacket = 1 };
    static void write_ray(const float* org_dir, float tmin, float tmax, int /*j*/, Ray1& ray) {
        ray.org.e[0] = org_dir[0];
        ray.org.e[1] = org_dir[1];
        ray.org.e[2] = org_dir[2];
        ray.dir.e[0] = org_dir[3];
        ray.dir.e[1] = org_dir[4];
        ray.dir.e[2] = org_dir[5];
        ray.tmin = tmin;
        ray.tmax = tmax;
    }
};

template <>
struct RayTraits<Ray4> {
    enum { RayPerPacket = 4 };
    static void write_ray(const float* org_dir, float tmin, float tmax, int j, Ray4& ray) {
        ray.org.e[0].e[j] = org_dir[0];
        ray.org.e[1].e[j] = org_dir[1];
        ray.org.e[2].e[j] = org_dir[2];
        ray.dir.e[0].e[j] = org_dir[3];
        ray.dir.e[1].e[j] = org_dir[4];
        ray.dir.e[2].e[j] = org_dir[5];
        ray.tmin.e[j] = tmin;
        ray.tmax.e[j] = tmax;
    }
};

template <>
struct RayTraits<Ray8> {
    enum { RayPerPacket = 8 };
    static void write_ray(const float* org_dir, float tmin, float tmax, int j, Ray8& ray) {
        ray.org.e[0].e[j] = org_dir[0];
        ray.org.e[1].e[j] = org_dir[1];
        ray.org.e[2].e[j] = org_dir[2];
        ray.dir.e[0].e[j] = org_dir[3];
        ray.dir.e[1].e[j] = org_dir[4];
        ray.dir.e[2].e[j] = org_dir[5];
        ray.tmin.e[j] = tmin;
        ray.tmax.e[j] = tmax;
    }
};

template <typename Ray>
inline bool load_rays(
    const std::string& filename,
    anydsl::Array<Ray>& rays,
    float tmin, float tmax,
    anydsl::Platform platform,
    anydsl::Device device) {
    std::ifstream in(filename, std::ifstream::binary);
    if (!in) return false;

    in.seekg(0, std::ios_base::end);
    auto size = in.tellg();
    in.seekg(0, std::ios_base::beg);

    if (size % (sizeof(float) * 6) != 0) return false;

    auto rays_per_packet = RayTraits<Ray>::RayPerPacket;
    auto ray_count = size / (rays_per_packet * sizeof(float) * 6);
    anydsl::Array<Ray> host_rays(ray_count);

    for (size_t i = 0; i < ray_count; i++) {
        for (int j = 0; j < rays_per_packet; j++) {
            float org_dir[6];
            in.read((char*)org_dir, sizeof(float) * 6);
            RayTraits<Ray>::write_ray(org_dir, tmin, tmax, j, host_rays[i]);
        }
    }

    if (platform != anydsl::Platform::Host) {
        rays = std::move(anydsl::Array<Ray>(platform, device, ray_count));
        anydsl::copy(host_rays, rays);
    } else {
        rays = std::move(host_rays);
    }
    return true;
}

#endif // LOAD_RAYS_H
