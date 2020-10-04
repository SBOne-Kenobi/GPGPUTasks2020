#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#ifndef WORK_GROUP_SIZE
#define WORK_GROUP_SIZE 128
#endif

#line 6

__kernel void bad_sum(
        __global const unsigned int *source,
        __global unsigned int *result,
        unsigned int n) {
  const int localId = get_local_id(0);
  const int globalId = get_global_id(0);

  __local unsigned int uploaded[WORK_GROUP_SIZE];
  if (globalId >= n)
    uploaded[localId] = 0;
  else
    uploaded[localId] = source[globalId];

  barrier(CLK_LOCAL_MEM_FENCE);

  if (localId == 0) {
    unsigned int res = 0;
    for (int i = 0; i < WORK_GROUP_SIZE; i++)
      res += uploaded[i];
    atomic_add(result, res);
  }
}

__kernel void opt_sum(
        __global const unsigned int *source,
        __global unsigned int *result,
        unsigned int n) {
  const int localId = get_local_id(0);
  const int globalId = get_global_id(0);

  __local unsigned int uploaded[WORK_GROUP_SIZE];
  if (globalId >= n)
    uploaded[localId] = 0;
  else
    uploaded[localId] = source[globalId];

  barrier(CLK_LOCAL_MEM_FENCE);

  for (unsigned int len = WORK_GROUP_SIZE / 2; len > 0; len /= 2) {
    if (localId < len) {
      unsigned int a = uploaded[localId];
      unsigned int b = uploaded[localId + len];
      uploaded[localId] = a + b;
    }
    if (len > WARP_SIZE)
      barrier(CLK_LOCAL_MEM_FENCE);
    else if (2 * localId >= len)
      return;
  }

  if (localId == 0)
    atomic_add(result, uploaded[0]);
}