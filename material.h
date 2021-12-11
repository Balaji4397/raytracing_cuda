#ifndef MATERIALH
#define MATERIALH

struct hit_record;

#include "ray.h"
#include "hitable.h"


__device__ float schlick(float cos, float reference_index) {
    float ref0 = (1.0f-reference_index) / (1.0f+reference_index);
    ref0 = ref0*ref0;
    return ref0 + (1.0f-ref0)*pow((1.0f - cos),5.0f);
}

__device__ bool refract(const vec3& v, const vec3& n, float nint, vec3& rftd) {
    vec3 unit_vec = unit_vector(v);
    float dt = dot(unit_vec, n);
    float dsc = 1.0f - nint*nint*(1-dt*dt);
    if (dsc < 0) {
        return false;
    }
    else
        rftd = nint*(unit_vec - n*dt) - n*sqrt(dsc);
        return true;
        
}

#define RANDVEC3 vec3(curand_uniform(lrs),curand_uniform(lrs),curand_uniform(lrs))

__device__ vec3 random_in_unit_sphere(curandState *lrs) {
    vec3 p;
    do {
        p = 2.0f*RANDVEC3 - vec3(1,1,1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
     return v - 2.0f*dot(v,n)*n;
}

class material  {
    public:
        __device__ virtual bool scatter(const ray& ray_in, const hit_record& record, vec3& at, ray& sc, curandState *lrs) const = 0;
};

class lambertian : public material {
    public:
        __device__ lambertian(const vec3& a) : alb(a) {}
        __device__ virtual bool scatter(const ray& ray_in, const hit_record& record, vec3& at, ray& sc, curandState *lrs) const  {
             vec3 goal = record.p + record.normal + random_in_unit_sphere(lrs);
             sc = ray(record.p, goal-record.p);
             at = alb;
             return true;
        }

        vec3 alb;
};

class metal : public material {
    public:
        __device__ metal(const vec3& a, float f) : alb(a) { if (f > 1) fu = 1; else fu = f; }
        __device__ virtual bool scatter(const ray& ray_in, const hit_record& record, vec3& at, ray& sc, curandState *lrs) const  {
            vec3 rfl = reflect(unit_vector(ray_in.direction()), record.normal);
            sc = ray(record.p, rfl + fu*random_in_unit_sphere(lrs));
            at = alb;
            return (dot(sc.direction(), record.normal) > 0.0f);
        }
        vec3 alb;
        float fu;
};

class dielectric : public material {
public:
    __device__ dielectric(float ri) : reference_index(ri) {}
    __device__ virtual bool scatter(const ray& ray_in,
                         const hit_record& record,
                         vec3& at,
                         ray& sc,
                         curandState *lrs) const  {
        vec3 out_norm;
        vec3 rfl = reflect(ray_in.direction(), record.normal);
        float nint;
        at = vec3(1.0, 1.0, 1.0);
        vec3 rftd;
        float rfl_pr;
        float cos;
        if (dot(ray_in.direction(), record.normal) < 0.0f) {
            
            out_norm = record.normal;
            nint = 1.0f / reference_index;
            cos = -dot(ray_in.direction(), record.normal) / ray_in.direction().length();
        }
        else {
            out_norm = -record.normal;
            nint = reference_index;
            cos = dot(ray_in.direction(), record.normal) / ray_in.direction().length();
            cos = sqrt(1.0f - reference_index*reference_index*(1-cos*cos));
        }
        if (refract(ray_in.direction(), out_norm, nint, rftd))
            rfl_pr = schlick(cos, reference_index);
        else
            rfl_pr = 1.0f;
        if (curand_uniform(lrs) > rfl_pr)
            sc = ray(record.p, rftd);
        else
            sc = ray(record.p, rfl);
        return true;
    }

    float reference_index;
};
#endif
