#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <stdlib.h>
#include <iostream>

class vec3  {


public:
    __host__ __device__ vec3() {}
    __host__ __device__ vec3(float a, float b, float c) { arr[0] = a; arr[1] = b; arr[2] = c; }
    __host__ __device__ inline float s() const { return arr[0]; }
    __host__ __device__ inline float t() const { return arr[1]; }
    __host__ __device__ inline float r() const { return arr[2]; }
    __host__ __device__ inline float red() const { return arr[0]; }
    __host__ __device__ inline float green() const { return arr[1]; }
    __host__ __device__ inline float blue() const { return arr[2]; }

    __host__ __device__ inline const vec3& operator+() const { return *this; }
    __host__ __device__ inline vec3 operator-() const { return vec3(-arr[0], -arr[1], -arr[2]); }
    __host__ __device__ inline float operator[](int i) const { return arr[i]; }
    __host__ __device__ inline float& operator[](int i) { return arr[i]; };

    __host__ __device__ inline vec3& operator+=(const vec3 &vector2);
    __host__ __device__ inline vec3& operator-=(const vec3 &vector2);
    __host__ __device__ inline vec3& operator*=(const vec3 &vector2);
    __host__ __device__ inline vec3& operator/=(const vec3 &vector2);
    __host__ __device__ inline vec3& operator*=(const float obj1);
    __host__ __device__ inline vec3& operator/=(const float obj1);

    __host__ __device__ inline float length() const { return sqrt(arr[0]*arr[0] + arr[1]*arr[1] + arr[2]*arr[2]); }
    __host__ __device__ inline float squared_length() const { return arr[0]*arr[0] + arr[1]*arr[1] + arr[2]*arr[2]; }
    __host__ __device__ inline void make_unit_vector();


    float arr[3];
};



inline std::istream& operator>>(std::istream &is, vec3 &obj1) {
    is >> obj1.arr[0] >> obj1.arr[1] >> obj1.arr[2];
    return is;
}

inline std::ostream& operator<<(std::ostream &os, const vec3 &obj1) {
    os << obj1.arr[0] << " " << obj1.arr[1] << " " << obj1.arr[2];
    return os;
}

__host__ __device__ inline void vec3::make_unit_vector() {
    float k = 1.0 / sqrt(arr[0]*arr[0] + arr[1]*arr[1] + arr[2]*arr[2]);
    arr[0] *= k; arr[1] *= k; arr[2] *= k;
}

__host__ __device__ inline vec3 operator+(const vec3 &vector1, const vec3 &vector2) {
    return vec3(vector1.arr[0] + vector2.arr[0], vector1.arr[1] + vector2.arr[1], vector1.arr[2] + vector2.arr[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &vector1, const vec3 &vector2) {
    return vec3(vector1.arr[0] - vector2.arr[0], vector1.arr[1] - vector2.arr[1], vector1.arr[2] - vector2.arr[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &vector1, const vec3 &vector2) {
    return vec3(vector1.arr[0] * vector2.arr[0], vector1.arr[1] * vector2.arr[1], vector1.arr[2] * vector2.arr[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &vector1, const vec3 &vector2) {
    return vec3(vector1.arr[0] / vector2.arr[0], vector1.arr[1] / vector2.arr[1], vector1.arr[2] / vector2.arr[2]);
}

__host__ __device__ inline vec3 operator*(float obj1, const vec3 &vector) {
    return vec3(obj1*vector.arr[0], obj1*vector.arr[1], obj1*vector.arr[2]);
}

__host__ __device__ inline vec3 operator/(vec3 vector, float obj1) {
    return vec3(vector.arr[0]/obj1, vector.arr[1]/obj1, vector.arr[2]/obj1);
}

__host__ __device__ inline vec3 operator*(const vec3 &vector, float obj1) {
    return vec3(obj1*vector.arr[0], obj1*vector.arr[1], obj1*vector.arr[2]);
}

__host__ __device__ inline float dot(const vec3 &vector1, const vec3 &vector2) {
    return vector1.arr[0] *vector2.arr[0] + vector1.arr[1] *vector2.arr[1]  + vector1.arr[2] *vector2.arr[2];
}

__host__ __device__ inline vec3 cross(const vec3 &vector1, const vec3 &vector2) {
    return vec3( (vector1.arr[1]*vector2.arr[2] - vector1.arr[2]*vector2.arr[1]),
                (-(vector1.arr[0]*vector2.arr[2] - vector1.arr[2]*vector2.arr[0])),
                (vector1.arr[0]*vector2.arr[1] - vector1.arr[1]*vector2.arr[0]));
}


__host__ __device__ inline vec3& vec3::operator+=(const vec3 &vector){
    arr[0]  += vector.arr[0];
    arr[1]  += vector.arr[1];
    arr[2]  += vector.arr[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3 &vector){
    arr[0]  *= vector.arr[0];
    arr[1]  *= vector.arr[1];
    arr[2]  *= vector.arr[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3 &vector){
    arr[0]  /= vector.arr[0];
    arr[1]  /= vector.arr[1];
    arr[2]  /= vector.arr[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& vector) {
    arr[0]  -= vector.arr[0];
    arr[1]  -= vector.arr[1];
    arr[2]  -= vector.arr[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float obj1) {
    arr[0]  *= obj1;
    arr[1]  *= obj1;
    arr[2]  *= obj1;
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const float obj1) {
    float k = 1.0/obj1;

    arr[0]  *= k;
    arr[1]  *= k;
    arr[2]  *= k;
    return *this;
}

__host__ __device__ inline vec3 unit_vector(vec3 vector) {
    return vector / vector.length();
}

#endif
