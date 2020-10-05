#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#ifndef WORK_GROUP_SIZE
#define WORK_GROUP_SIZE 128
#endif

#line 6

__kernel void init_buffer(
        __global int *buffer,
        __global const int *a,
        int n) {
  const int i = get_global_id(0);
  if (i < n) {
    buffer[3 * i + 0] = a[i];
    if (a[i] > 0) {
      buffer[3 * i + 1] = i + 1;
      buffer[3 * i + 2] = a[i];
    } else {
      buffer[3 * i + 1] = i;
      buffer[3 * i + 2] = 0;
    }
  } else {
    buffer[3 * i + 0] = 0;
    buffer[3 * i + 1] = i;
    buffer[3 * i + 2] = 0;
  }
}


__kernel void reorder_buffer(
        __global int *buffer,
        int len) {
  const int globalId = get_global_id(0);

  if (globalId == 0) {
    int cnt = (len + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
    for (int i = 1; i < cnt; ++i) {
      buffer[3 * i + 0] = buffer[3 * (i * WORK_GROUP_SIZE) + 0];
      buffer[3 * i + 1] = buffer[3 * (i * WORK_GROUP_SIZE) + 1];
      buffer[3 * i + 2] = buffer[3 * (i * WORK_GROUP_SIZE) + 2];
    }
  }
}

__kernel void max_prefix(
        __global int *buffer,
        int len) {
  const int localId = get_local_id(0);
  const int globalId = get_global_id(0);
  const int groupId = get_group_id(0);

  __local int uploaded[3 * WORK_GROUP_SIZE];

  for (int i = 0; i < 3; i++) {
    uploaded[i * WORK_GROUP_SIZE + localId] = buffer[3 * groupId * WORK_GROUP_SIZE + i * WORK_GROUP_SIZE + localId];
  }

  //TODO : optimize
  barrier(CLK_LOCAL_MEM_FENCE);

  if (globalId < len) {
    if (localId == 0) {
      int sum = 0;
      int res = 0;
      int max_sum = 0;
      for (int i = 0; i < min(WORK_GROUP_SIZE, len - globalId); ++i) {
        int part_sum = uploaded[3 * i + 0];
        int part_res = uploaded[3 * i + 1];
        int part_max_sum = uploaded[3 * i + 2] + sum;
        if (max_sum < part_max_sum)
          max_sum = part_max_sum, res = part_res;
        sum += part_sum;
      }
      uploaded[0] = sum;
      uploaded[1] = res;
      uploaded[2] = max_sum;
//      printf("---------LOG : global_id=%d, len=%d, ans=(%d, %d, %d)\n", globalId, len, max_sum, res, sum);
    }
  }

  if (localId == 0 && globalId < len) {
    // replace results back
    buffer[3 * groupId * WORK_GROUP_SIZE + 0] = uploaded[0];
    buffer[3 * groupId * WORK_GROUP_SIZE + 1] = uploaded[1];
    buffer[3 * groupId * WORK_GROUP_SIZE + 2] = uploaded[2];
  }
}