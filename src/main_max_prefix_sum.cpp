#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/max_prefix_sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
  if (a != b) {
    std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
    throw std::runtime_error(message);
  }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

int main(int argc, char **argv) {
  gpu::Device device = gpu::chooseGPUDevice(argc, argv);
  gpu::Context context;
  context.init(device.device_id_opencl);
  context.activate();

  size_t groupSize = 128;

  std::string defines = " -D WORK_GROUP_SIZE=" + to_string(groupSize);
  ocl::Kernel max_prefix(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix", defines);
  ocl::Kernel init_buffer(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "init_buffer", defines);
  ocl::Kernel reorder_buffer(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "reorder_buffer", defines);
  max_prefix.compile();
  init_buffer.compile();
  reorder_buffer.compile();

  gpu::gpu_mem_32i buffer_gpu, as_gpu;

  int benchmarkingIters = 10;
  int max_n = (1 << 24);

  for (int n = 2; n <= max_n; n *= 2) {
    std::cout << "______________________________________________" << std::endl;
    int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
    std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

    std::vector<int> as(n, 0);
    FastRandom r(n);
    for (int i = 0; i < n; ++i) {
      as[i] = (unsigned int) r.next(-values_range, values_range);
    }

    int reference_max_sum;
    int reference_result;
    {
      int max_sum = 0;
      int sum = 0;
      int result = 0;
      for (int i = 0; i < n; ++i) {
        sum += as[i];
        if (sum > max_sum) {
          max_sum = sum;
          result = i + 1;
        }
      }
      reference_max_sum = max_sum;
      reference_result = result;
    }
    std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")" << std::endl;

    {
      timer t;
      for (int iter = 0; iter < benchmarkingIters; ++iter) {
        int max_sum = 0;
        int sum = 0;
        int result = 0;
        for (int i = 0; i < n; ++i) {
          sum += as[i];
          if (sum > max_sum) {
            max_sum = sum;
            result = i + 1;
          }
        }
        EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
        EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
        t.nextLap();
      }
      std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
      std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
      size_t globalSize = (n + groupSize - 1) / groupSize * groupSize;
      as_gpu.resizeN(n);
      as_gpu.writeN(as.data(), n);
      buffer_gpu.resizeN(3 * globalSize);

      timer t;
      for (int i = 0; i < benchmarkingIters; i++) {
        int max_sum;
        int res;

        gpu::WorkSize workSize(groupSize, globalSize);
        init_buffer.exec(workSize, buffer_gpu, as_gpu, n);

        for (int len = globalSize; len > 1; len /= groupSize) {
          workSize = gpu::WorkSize(groupSize, (len + groupSize - 1) / groupSize * groupSize);

          max_prefix.exec(workSize, buffer_gpu, len);

          //FIXME
          reorder_buffer.exec(workSize, buffer_gpu, len);
        }

        int result_from_buffer[3];
        buffer_gpu.readN(result_from_buffer, 3);

        max_sum = result_from_buffer[2];
        res = result_from_buffer[1];

        EXPECT_THE_SAME(reference_max_sum, max_sum, "GPU result should be consistent!");
        EXPECT_THE_SAME(reference_result, res, "GPU result should be consistent!");
        t.nextLap();
      }
      std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
      std::cout << "GPU: " << (globalSize / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
  }
}
