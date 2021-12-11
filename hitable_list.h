#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"

class hitable_list: public hitable  {
    public:
        __device__ hitable_list() {}
        __device__ hitable_list(hitable **lst, int lst_length) {list = lst; list_size = lst_length; }
        __device__ virtual bool hit(const ray& r, float tminimum, float tmaximum, hit_record& record) const;
        hitable **list;
        int list_size;
};

__device__ bool hitable_list::hit(const ray& r, float t_minimum, float t_maximum, hit_record& record) const {
        hit_record temp_record;
        bool hitany = false;
        float nearest = t_maximum;
        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit(r, t_minimum, nearest, temp_record)) {
                hitany = true;
                nearest = temp_record.t;
                record = temp_record;
            }
        }
        return hitany;
}

#endif
