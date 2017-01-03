__global__ void _add_32_01(int n, float s, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = s+xi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void add_32_01(int n, float s, float *x, float *y) {
    _add_32_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _add_64_01(int n, double s, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = s+xi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void add_64_01(int n, double s, double *x, double *y) {
    _add_64_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _sub_32_01(int n, float s, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = s-xi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sub_32_01(int n, float s, float *x, float *y) {
    _sub_32_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _sub_64_01(int n, double s, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = s-xi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void sub_64_01(int n, double s, double *x, double *y) {
    _sub_64_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _mul_32_01(int n, float s, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = s*xi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void mul_32_01(int n, float s, float *x, float *y) {
    _mul_32_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _mul_64_01(int n, double s, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = s*xi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void mul_64_01(int n, double s, double *x, double *y) {
    _mul_64_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _div_32_01(int n, float s, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = s/xi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void div_32_01(int n, float s, float *x, float *y) {
    _div_32_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _div_64_01(int n, double s, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = s/xi;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void div_64_01(int n, double s, double *x, double *y) {
    _div_64_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _pow_32_01(int n, float s, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = pow(s,xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void pow_32_01(int n, float s, float *x, float *y) {
    _pow_32_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _pow_64_01(int n, double s, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = pow(s,xi);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void pow_64_01(int n, double s, double *x, double *y) {
    _pow_64_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _max_32_01(int n, float s, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = (xi>s?xi:s);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void max_32_01(int n, float s, float *x, float *y) {
    _max_32_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _max_64_01(int n, double s, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = (xi>s?xi:s);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void max_64_01(int n, double s, double *x, double *y) {
    _max_64_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _min_32_01(int n, float s, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = (xi<s?xi:s);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void min_32_01(int n, float s, float *x, float *y) {
    _min_32_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _min_64_01(int n, double s, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = (xi<s?xi:s);
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void min_64_01(int n, double s, double *x, double *y) {
    _min_64_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _eq_32_01(int n, float s, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = xi==s;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void eq_32_01(int n, float s, float *x, float *y) {
    _eq_32_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _eq_64_01(int n, double s, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = xi==s;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void eq_64_01(int n, double s, double *x, double *y) {
    _eq_64_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _gt_32_01(int n, float s, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = xi>s;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void gt_32_01(int n, float s, float *x, float *y) {
    _gt_32_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _gt_64_01(int n, double s, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = xi>s;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void gt_64_01(int n, double s, double *x, double *y) {
    _gt_64_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _ge_32_01(int n, float s, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = xi>=s;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void ge_32_01(int n, float s, float *x, float *y) {
    _ge_32_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _ge_64_01(int n, double s, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = xi>=s;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void ge_64_01(int n, double s, double *x, double *y) {
    _ge_64_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _lt_32_01(int n, float s, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = xi<s;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void lt_32_01(int n, float s, float *x, float *y) {
    _lt_32_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _lt_64_01(int n, double s, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = xi<s;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void lt_64_01(int n, double s, double *x, double *y) {
    _lt_64_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _le_32_01(int n, float s, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = xi<=s;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void le_32_01(int n, float s, float *x, float *y) {
    _le_32_01<<<256,256>>>(n,s,x,y);
  }
}
__global__ void _le_64_01(int n, double s, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = xi<=s;
    i += blockDim.x * gridDim.x;
  }
}
extern "C" {
  void le_64_01(int n, double s, double *x, double *y) {
    _le_64_01<<<256,256>>>(n,s,x,y);
  }
}
