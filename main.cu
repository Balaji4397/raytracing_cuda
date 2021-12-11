#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t outcome, char const *const function, const char *const f, int const l) {
    if (outcome) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(outcome) << " at " <<
            f << ":" << l << " '" << function << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}
__device__ vec3 color(const ray& r, hitable **world, curandState *lrs) {
    ray cr = r;
    vec3 ca = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record record;
        if ((*world)->hit(cr, 0.001f, FLT_MAX, record)) {
            ray sc;
            vec3 at;
            if(record.mat_ptr->scatter(cr, record, at, sc, lrs)) {
                ca *= at;
                cr = sc;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 ud = unit_vector(cr.direction());
            float a = 0.5f*(ud.t() + 1.0f);
            vec3 b = (1.0f-a)*vec3(1.0, 1.0, 1.0) + a*vec3(0.5, 0.7, 1.0);
            return ca * b;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rs) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rs);
    }
}

__global__ void render_init(int maximum_x, int maximum_y, curandState *rs) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= maximum_x) || (j >= maximum_y)) return;
    int pi = j*maximum_x + i;
    curand_init(1984+pi, 0, 0, &rs[pi]);
}

__global__ void render(vec3 *pointer_f, int maximum_x, int maximum_y, int ns, camera **cam, hitable **world, curandState *rs) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= maximum_x) || (j >= maximum_y)) return;
    int pi = j*maximum_x + i;
    curandState lrs = rs[pi];
    vec3 clr(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&lrs)) / float(maximum_x);
        float v = float(j + curand_uniform(&lrs)) / float(maximum_y);
        ray r = (*cam)->get_ray(u, v, &lrs);
        clr += color(r, world, &lrs);
    }
    rs[pi] = lrs;
    clr /= float(ns);
    clr[0] = sqrt(clr[0]);
    clr[1] = sqrt(clr[1]);
    clr[2] = sqrt(clr[2]);
    pointer_f[pi] = clr;
}

#define RND (curand_uniform(&lrs))

__global__ void create_world(hitable **dl, hitable **dw, camera **dc, int nx, int ny, curandState *rs) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState lrs = *rs;
        dl[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int x = -11; x < 11; x++) {
            for(int y = -11; y < 11; y++) {
                float cm = RND;
                vec3 center(x+RND,0.2,y+RND);
                if(cm < 0.8f) {
                    dl[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(cm < 0.95f) {
                    dl[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    dl[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        dl[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        dl[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        dl[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rs = lrs;
        *dw  = new hitable_list(dl, 22*22+1+3);

        vec3 startfrom(13,2,3);
        vec3 check(0,0,0);
        float distance_f = 10.0; (startfrom-check).length();
        float ap = 0.1;
        *dc   = new camera(startfrom,
                                 check,
                                 vec3(0,1,0),
                                 30.0,
                                 float(nx)/float(ny),
                                 ap,
                                 distance_f);
    }
}

__global__ void free_world(hitable **dl, hitable **dw, camera **dc) {
    for(int i=0; i < 22*22+1+3; i++) {
        delete ((sphere *)dl[i])->mat_ptr;
        delete dl[i];
    }
    delete *dw;
    delete *dc;
}

int main() {
    int nx = 1200;
    int ny = 800;
    int ns = 10;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " bl.\n";

    int npix = nx*ny;
    size_t pointer_f_size = npix*sizeof(vec3);
    vec3 *pointer_f;
    checkCudaErrors(cudaMallocManaged((void **)&pointer_f, pointer_f_size));
    curandState *drs;
    checkCudaErrors(cudaMalloc((void **)&drs, npix*sizeof(curandState)));
    curandState *drs2;
    checkCudaErrors(cudaMalloc((void **)&drs2, 1*sizeof(curandState)));
    rand_init<<<1,1>>>(drs2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    hitable **dl;
    int nhit = 22*22+1+3;
    checkCudaErrors(cudaMalloc((void **)&dl, nhit*sizeof(hitable *)));
    hitable **dw;
    checkCudaErrors(cudaMalloc((void **)&dw, sizeof(hitable *)));
    camera **dc;
    checkCudaErrors(cudaMalloc((void **)&dc, sizeof(camera *)));
    create_world<<<1,1>>>(dl, dw, dc, nx, ny, drs2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t begin, end;
    begin = clock();
    dim3 bl(nx/tx+1,ny/ty+1);
    dim3 tr(tx,ty);
    render_init<<<bl, tr>>>(nx, ny, drs);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<bl, tr>>>(pointer_f, nx, ny,  ns, dc, dw, drs);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    end = clock();
    double ts = ((double)(end - begin)) / CLOCKS_PER_SEC;
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pi = j*nx + i;
            int i_red = int(255.99*pointer_f[pi].red());
            int i_green = int(255.99*pointer_f[pi].green());
            int i_blue = int(255.99*pointer_f[pi].blue());
            std::cout << i_red << " " << i_green << " " << i_blue << "\n";
        }
    }
    
    std::cerr << "took " << ts << " seconds.\n";
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(dl,dw,dc);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(dc));
    checkCudaErrors(cudaFree(dw));
    checkCudaErrors(cudaFree(dl));
    checkCudaErrors(cudaFree(drs));
    checkCudaErrors(cudaFree(drs2));
    checkCudaErrors(cudaFree(pointer_f));

    cudaDeviceReset();
}
