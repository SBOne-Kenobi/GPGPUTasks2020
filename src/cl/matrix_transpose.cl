#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#ifndef WORK_GROUP_SIZE
#define WORK_GROUP_SIZE 128
#endif

#define TILE_SIZE 32

__kernel void matrix_transpose(
        __global const float *a,
        __global float *t_a,
        unsigned int M, unsigned int K
) {
  const unsigned int local_id = get_local_id(0);
  const unsigned int global_id = get_global_id(0);
  const unsigned int group_id = get_group_id(0);
  int block_num_x = (K + TILE_SIZE - 1) / TILE_SIZE;
  int block_x = group_id % block_num_x;
  int block_y = group_id / block_num_x;
  int offset_x = block_x * TILE_SIZE;
  int offset_y = block_y * TILE_SIZE;

  __local float uploaded[TILE_SIZE * (TILE_SIZE + 1)];
  for (int i = local_id; i < TILE_SIZE * TILE_SIZE; i += WORK_GROUP_SIZE) {
    int x = i % TILE_SIZE;
    int y = i / TILE_SIZE;
    int to = x + y * (TILE_SIZE + 1);

    int cur_x = offset_x + x;
    int cur_y = offset_y + y;
    int from = cur_x + cur_y * K;

    if (cur_x < K && cur_y < M) {
      uploaded[to] = a[from];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = local_id; i < TILE_SIZE * TILE_SIZE; i += WORK_GROUP_SIZE) {
    int y = i % TILE_SIZE;
    int x = i / TILE_SIZE;
    int from = x + y * (TILE_SIZE + 1);

    int cur_x = offset_x + x;
    int cur_y = offset_y + y;
    int to = cur_x * M + cur_y;

    if (cur_x < K && cur_y < M) {
      t_a[to] = uploaded[from];
    }
  }

}