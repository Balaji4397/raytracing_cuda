#ifndef CAMERAH
#define CAMERAH

#include <curand_kernel.h>
#include "ray.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ vec3 random_in_unit_disk(curandState *lrs) {
    vec3 p;
    do {
        p = 2.0f*vec3(curand_uniform(lrs),curand_uniform(lrs),0) - vec3(1,1,0);
    } while (dot(p,p) >= 1.0f);
    return p;
}

class camera {
public:
    __device__ camera(vec3 startfrom, vec3 check, vec3 vup, float vector_fov, float as, float ap, float focus_distance) { // vector_fov is top to bottom in degrees
        l_rad = ap / 2.0f;
        float t = vector_fov*((float)M_PI)/180.0f;
        float mid_height = tan(t/2.0f);
        float mid_width = as * mid_height;
        origin = startfrom;
        w = unit_vector(startfrom - check);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        llc = origin  - mid_width*focus_distance*u -mid_height*focus_distance*v - focus_distance*w;
        hor = 2.0f*mid_width*focus_distance*u;
        ver = 2.0f*mid_height*focus_distance*v;
    }
    __device__ ray get_ray(float s, float t, curandState *lrs) {
        vec3 rd = l_rad*random_in_unit_disk(lrs);
        vec3 off = u * rd.s() + v * rd.t();
        return ray(origin + off, llc + s*hor + t*ver - origin - off);
    }

    vec3 origin;
    vec3 llc;
    vec3 hor;
    vec3 ver;
    vec3 u, v, w;
    float l_rad;
};

#endif
