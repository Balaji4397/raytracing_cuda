#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class sphere: public hitable  {
    public:
        __device__ sphere() {}
        __device__ sphere(vec3 center, float rad, material *m_pointer) : center(center), radius(rad), mat_ptr(m_pointer)  {};
        __device__ virtual bool hit(const ray& rad, float tminimum, float tmaximum, hit_record& record) const;
        vec3 center;
        float radius;
        material *mat_ptr;
};

__device__ bool sphere::hit(const ray& rad, float t_minimum, float t_maximum, hit_record& record) const {
    vec3 origin_center = rad.origin() - center;
    float a = dot(rad.direction(), rad.direction());
    float b = dot(origin_center, rad.direction());
    float c = dot(origin_center, origin_center) - radius*radius;
    float dsc = b*b - a*c;
    if (dsc > 0) {
        float tmpr = (-b - sqrt(dsc))/a;
        if (tmpr < t_maximum && tmpr > t_minimum) {
            record.t = tmpr;
            record.p = rad.point_at_parameter(record.t);
            record.normal = (record.p - center) / radius;
            record.mat_ptr = mat_ptr;
            return true;
        }
        tmpr = (-b + sqrt(dsc)) / a;
        if (tmpr < t_maximum && tmpr > t_minimum) {
            record.t = tmpr;
            record.p = rad.point_at_parameter(record.t);
            record.normal = (record.p - center) / radius;
            record.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}


#endif
