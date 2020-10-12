#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#ifndef WORK_GROUP_SIZE
#define WORK_GROUP_SIZE 128
#endif

#define TILE_SIZE 32

__kernel void matrix_multiplication(
        __global const float *a,
        __global const float *b,
        __global float *c,
        unsigned int M, unsigned int K, unsigned int N) {
  const unsigned int local_id = get_local_id(0);
  const unsigned int global_id = get_global_id(0);
  const unsigned int group_id = get_group_id(0);
  int block_num_x = (N + TILE_SIZE - 1) / TILE_SIZE;
  int block_x = group_id % block_num_x;
  int block_y = group_id / block_num_x;
  int offset_x = block_x * TILE_SIZE;
  int offset_y = block_y * TILE_SIZE;

  __local float result[TILE_SIZE * TILE_SIZE];
  for (int i = local_id; i < TILE_SIZE * TILE_SIZE; i += WORK_GROUP_SIZE)
    result[i] = 0;

  barrier(CLK_LOCAL_MEM_FENCE);

  __local float uploaded[TILE_SIZE * (TILE_SIZE + 1)];
  for (int add = 0; add < K; add += TILE_SIZE) {
    for (int i = local_id; i < TILE_SIZE * TILE_SIZE; i += WORK_GROUP_SIZE) {
      int y = i / TILE_SIZE;
      int x = i % TILE_SIZE;
      int to = y + x * (TILE_SIZE + 1);

      int cur_x = offset_x + x;
      int cur_y = add + y;
      int from = cur_x + cur_y * N;

      if (cur_x < N && cur_y < K)
        uploaded[to] = b[from];
      else
        uploaded[to] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = local_id; i < TILE_SIZE * TILE_SIZE; i += WORK_GROUP_SIZE) {
      int x = i % TILE_SIZE;
      int y = i / TILE_SIZE;
      int to = x + y * TILE_SIZE;

      int sum = 0;
      for (int j = 0; j < TILE_SIZE; j++) {
        int le_x = add + j;
        int le_y = offset_y + y;
        int le = le_x + le_y * K;

        int re_x = j;
        int re_y = x;
        int re = re_x + re_y * (TILE_SIZE + 1);

        if (le_x < K && le_y < M) {
          sum += (uploaded[re] * a[le]);
        }
      }
      result[x + y * TILE_SIZE] += sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (int i = local_id; i < TILE_SIZE * TILE_SIZE; i += WORK_GROUP_SIZE) {
    int x = i % TILE_SIZE;
    int y = i / TILE_SIZE;
    int from = x + y * TILE_SIZE;

    int cur_x = offset_x + x;
    int cur_y = offset_y + y;
    int to = cur_x + cur_y * N;

    if (cur_x < N && cur_y < M)
      c[to] = result[from];
  }

}