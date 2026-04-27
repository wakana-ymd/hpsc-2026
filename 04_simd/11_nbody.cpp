#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main()
{
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for (int i = 0; i < N; i++)
  {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m512 xvec = _mm512_load_ps(x);
  __m512 yvec = _mm512_load_ps(y);
  __m512 mvec = _mm512_load_ps(m);

  __m512i jvec = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
  
  for (int i = 0; i < N; i++)
  {
    __m512i ivec = _mm512_set1_epi32(i);
    __mmask16 mask = _mm512_cmpneq_epi32_mask(ivec, jvec);

    // calculate rx, ry
    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 yi = _mm512_set1_ps(y[i]);

    __m512 rx = _mm512_sub_ps(xi, xvec);
    __m512 ry = _mm512_sub_ps(yi, yvec);

    // calculate 1/r
    __m512 rx2 = _mm512_mul_ps(rx, rx);
    __m512 ry2 = _mm512_mul_ps(ry, ry);
    __m512 r2  = _mm512_add_ps(rx2, ry2);
    __m512 invr = _mm512_rsqrt14_ps(r2);

    // calculate fx, fy
    __m512 invr2 = _mm512_mul_ps(invr, invr);
    __m512 invr3 = _mm512_mul_ps(invr2, invr);

    __m512 coef = _mm512_mul_ps(mvec, invr3);
    __m512 fxpart = _mm512_mul_ps(coef, rx);
    __m512 fypart = _mm512_mul_ps(coef, ry);

    // mask out if i=j
    __m512 zero = _mm512_setzero_ps();
    fxpart = _mm512_mask_blend_ps(mask, zero, fxpart);
    fypart = _mm512_mask_blend_ps(mask, zero, fypart);

    fx[i] -= _mm512_reduce_add_ps(fxpart);
    fy[i] -= _mm512_reduce_add_ps(fypart);

    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
}
