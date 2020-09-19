#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cassert>

template<typename T>
std::string to_string(T value) {
  std::ostringstream ss;
  ss << value;
  return ss.str();
}

std::string to_string_device_type(cl_device_type type) {
  switch (type) {
    case CL_DEVICE_TYPE_ACCELERATOR:
      return "Accelerator";
    case CL_DEVICE_TYPE_CPU:
      return "CPU";
    case CL_DEVICE_TYPE_GPU:
      return "GPU";
    case CL_DEVICE_TYPE_DEFAULT:
      return "Default";
    default:
      return "Undefined device type";
  }
}

void reportError(cl_int err, const std::string &filename, int line) {
  if (CL_SUCCESS == err)
    return;

  // Таблица с кодами ошибок:
  // libs/clew/CL/cl.h:103
  // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
  std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
  throw std::runtime_error(message);
}

template<typename T, typename F, typename ...Args>
T safeCall(const std::string &filename, int line, F func, Args ...args) {
  cl_int errcode = CL_SUCCESS;
  T result = func(args..., &errcode);
  reportError(errcode, filename, line);
  return result;
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)
#define RET_SAFE_CALL(type, func, ...) safeCall<type>(__FILE__, __LINE__, func, __VA_ARGS__)

void printDeviceInfo(cl_device_id device) {
  std::cout << "Chosen device description:" << std::endl;
  size_t deviceNameSize = 0;
  OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
  std::vector<unsigned char> deviceName(deviceNameSize, 0);
  OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
  std::cout << "\tDevice name: " << deviceName.data() << std::endl;

  size_t vendorNameSize = 0;
  OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VENDOR, 0, nullptr, &vendorNameSize));
  std::vector<unsigned char> vendorName(vendorNameSize, 0);
  OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VENDOR, vendorNameSize, vendorName.data(), nullptr));
  cl_uint vendorId;
  OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof vendorId, &vendorId, nullptr));
  std::cout << "\tDevice vendor: " << vendorName.data() << " (id: " << vendorId << ")" << std::endl;

  cl_device_type deviceType;
  OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));
  std::cout << "\tDevice type: " << to_string_device_type(deviceType) << std::endl;

  cl_ulong deviceMemSize;
  OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
                                sizeof(deviceMemSize), &deviceMemSize, nullptr));
  deviceMemSize >>= 20;
  std::cout << "\tDevice memory size: " << deviceMemSize << " mb" << std::endl;

  cl_ulong deviceCacheSize;
  OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                                sizeof(deviceCacheSize), &deviceCacheSize, nullptr));

  deviceCacheSize >>= 10;
  std::cout << "\tDevice cache size: " << deviceCacheSize << " gb" << std::endl;

  cl_uint deviceCacheLineSize;
  OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
                                sizeof(deviceCacheLineSize), &deviceCacheLineSize, nullptr));

  std::cout << "\tDevice chacheline size: " << deviceCacheLineSize << " b" << std::endl;

  cl_uint deviceWorkItemDim;
  OCL_SAFE_CALL(
          clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &deviceWorkItemDim, nullptr));
  std::cout << "\tDevice work item dims: " << deviceWorkItemDim << std::endl;

  std::cout << "-------------------------" << std::endl;
}

cl_device_id getDevice() {
  const cl_uint NVIDIA_ID = 4318;

  cl_device_id chosenDevice = nullptr;

  cl_uint num_platforms;
  OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &num_platforms));
  std::vector<cl_platform_id> platforms(num_platforms);
  OCL_SAFE_CALL(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));
  for (auto platform: platforms) {
    cl_uint num_devices;
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices));
    std::vector<cl_device_id> devices(num_devices);
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr));
    for (auto device: devices) {
      cl_device_type type;
      OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof type, &type, nullptr));
      if (type == CL_DEVICE_TYPE_GPU || !chosenDevice)
        chosenDevice = device;
      cl_uint id;
      OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof id, &id, nullptr));
      if (id == NVIDIA_ID)
        return device;
    }
  }

  return chosenDevice;
}

cl_context getContext(cl_device_id device) {
  return RET_SAFE_CALL(cl_context, clCreateContext, nullptr, 1, &device, nullptr, nullptr);
}

cl_command_queue getCmdQueue(cl_context context, cl_device_id device,
                             cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE) {
  return RET_SAFE_CALL(cl_command_queue, clCreateCommandQueue, context, device, properties);
}

template<typename T>
class DataManage {
  using F = cl_int(*)(T data);
private:
  F func;
public:
  T data;

  DataManage(T data, F func) : func(func), data(data) {}

  ~DataManage() {
    OCL_SAFE_CALL(func(data));
  }
};

int main() {
  // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
  if (!ocl_init())
    throw std::runtime_error("Can't init OpenCL driver!");

  cl_device_id device = getDevice();
  printDeviceInfo(device);

  DataManage<cl_context> context(getContext(device), clReleaseContext);
//  cl_context context = getContext(device);

  DataManage<cl_command_queue> queue(getCmdQueue(context.data, device), clReleaseCommandQueue);
//  cl_command_queue queue = getCmdQueue(context, device);

  unsigned int n = 100 * 1000 * 1000;
  // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
  std::vector<float> as(n, 0);
  std::vector<float> bs(n, 0);
  std::vector<float> cs(n, 0);
  FastRandom r(n);
  for (unsigned int i = 0; i < n; ++i) {
    as[i] = r.nextf();
    bs[i] = r.nextf();
  }
  std::cout << "Data generated for n=" << n << "!" << std::endl;

  DataManage<cl_mem> as_buf(
          RET_SAFE_CALL(cl_mem, clCreateBuffer, context.data, CL_MEM_READ_ONLY, sizeof(float) * n, nullptr),
          clReleaseMemObject);

  DataManage<cl_mem> bs_buf(
          RET_SAFE_CALL(cl_mem, clCreateBuffer, context.data, CL_MEM_READ_ONLY, sizeof(float) * n, nullptr),
          clReleaseMemObject);

  DataManage<cl_mem> cs_buf(
          RET_SAFE_CALL(cl_mem, clCreateBuffer, context.data, CL_MEM_WRITE_ONLY, sizeof(float) * n, nullptr),
          clReleaseMemObject);

  OCL_SAFE_CALL(
          clEnqueueWriteBuffer(queue.data, as_buf.data, CL_TRUE, 0, sizeof(float) * n, as.data(), 0, nullptr, nullptr));

  OCL_SAFE_CALL(
          clEnqueueWriteBuffer(queue.data, bs_buf.data, CL_TRUE, 0, sizeof(float) * n, bs.data(), 0, nullptr, nullptr));

  std::string kernel_file = "src/cl/aplusb.cl";
  std::string kernel_sources;
  std::ifstream file(kernel_file);
  kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
  if (kernel_sources.empty()) {
    throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
  }
//  std::cout << kernel_sources << std::endl;
  const char *kernel_sources_c_str = kernel_sources.c_str();
  size_t sources_size = kernel_sources.size();
  DataManage<cl_program> program_aplusb(
          RET_SAFE_CALL(cl_program, clCreateProgramWithSource, context.data,
                        1, &kernel_sources_c_str, &sources_size),
          clReleaseProgram);

  cl_int build_err = clBuildProgram(program_aplusb.data, 1, &device, nullptr, nullptr, nullptr);

  size_t log_size = 0;
  OCL_SAFE_CALL(
          clGetProgramBuildInfo(program_aplusb.data, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
  std::vector<char> log(log_size, 0);
  OCL_SAFE_CALL(
          clGetProgramBuildInfo(program_aplusb.data, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));
  if (log_size > 1) {
    std::cout << "-------------------------" << std::endl;
    std::cout << "Log:" << std::endl;
    std::cout << log.data() << std::endl;
    std::cout << "-------------------------" << std::endl;
  }

  OCL_SAFE_CALL(build_err);

  DataManage<cl_kernel> kernel(
          RET_SAFE_CALL(cl_kernel, clCreateKernel, program_aplusb.data, "aplusb"),
          clReleaseKernel);

  // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь что тип количества элементов такой же в кернеле)
  {
    unsigned int i = 0;
    OCL_SAFE_CALL(clSetKernelArg(kernel.data, i++, sizeof(as_buf.data), &as_buf.data));
    OCL_SAFE_CALL(clSetKernelArg(kernel.data, i++, sizeof(bs_buf.data), &bs_buf.data));
    OCL_SAFE_CALL(clSetKernelArg(kernel.data, i++, sizeof(cs_buf.data), &cs_buf.data));
    OCL_SAFE_CALL(clSetKernelArg(kernel.data, i++, sizeof(unsigned int), &n));
  }

  {
    size_t workGroupSize = 128;
    size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
    timer t; // Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
    for (unsigned int i = 0; i < 20; ++i) {
      cl_event event;
      OCL_SAFE_CALL(clEnqueueNDRangeKernel(queue.data, kernel.data, 1, nullptr,
                                           &global_work_size, &workGroupSize, 0, nullptr, &event));
      OCL_SAFE_CALL(clWaitForEvents(1, &event));
      t.nextLap(); // При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
    }
    // Среднее время круга (вычисления кернела) на самом деле считаются не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклониение)
    // подробнее об этом - см. timer.lapsFiltered
    // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще) достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
    std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

    double GFlops = (n / t.lapAvg()) / 1e9;
    std::cout << "GFlops: " << GFlops << std::endl;

    double bandwidth = ((3. * n * sizeof(float)) / t.lapAvg()) / (1024 * 1024 * 1024);
    std::cout << "VRAM bandwidth: " << bandwidth << " GB/s" << std::endl;
  }

  // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
  {
    timer t;
    for (unsigned int i = 0; i < 20; ++i) {
      OCL_SAFE_CALL(clEnqueueReadBuffer(queue.data, cs_buf.data, CL_TRUE,
                                        0, n * sizeof(float), cs.data(), 0, nullptr, nullptr));
      t.nextLap();
    }
    std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    double bandwidth = ((1. * n * sizeof(float)) / t.lapAvg()) / (1024 * 1024 * 1024);
    std::cout << "VRAM -> RAM bandwidth: " << bandwidth << " GB/s" << std::endl;
  }

  // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
  for (unsigned int i = 0; i < n; ++i) {
    if (cs[i] != as[i] + bs[i]) {
      throw std::runtime_error("CPU and GPU results differ!");
    }
  }
  return 0;
}
