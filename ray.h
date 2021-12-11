#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray
{
    public:
        __device__ ray() {}
        __device__ ray(const vec3& l, const vec3& m) { L = l; M = m; }
        __device__ vec3 origin() const       { return L; }
        __device__ vec3 direction() const    { return M; }
        __device__ vec3 point_at_parameter(float t) const { return L + t*M; }

        vec3 L;
        vec3 M;
};

#endif
