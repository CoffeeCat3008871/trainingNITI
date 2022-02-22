/*
* Copyright 2021 Xilinx, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <chrono>

#include "xcl2.hpp" 
#include "cmdlineparser.h" 

#include "coefficients.h"
#include "common.h" 

using namespace sda;
using namespace sda::utils;

#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"

static uint64_t get_duration_ns(const std::vector<cl::Event>& events) {
    uint64_t duration = 0;
    for (size_t i=0; i<events.size(); i++) {
        uint64_t start, end;
        events[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &start);
        events[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &end);
        duration += end - start;
    }
    return duration;
}

// -------------------------------------------------------------------------------------------
// This version of the host application uses random input images instead of actual images
// loaded through OpenCV APIs. This program can be used as an alternative if OpenCV is not
// installed on the host machine.
// -------------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------------
// Class used to manage requests to the Filter2D kernel
// -------------------------------------------------------------------------------------------

class Filter2DRequest 
{
    cl::Kernel             kernel;
    cl::CommandQueue       q;
    cl::Buffer             coef_buffer;
    cl::Buffer             src_buffer;
    cl::Buffer             dst_buffer;
    std::vector<cl::Event> events;

  public:
    Filter2DRequest(cl::Context &context, cl::Program &program, cl::CommandQueue &queue) 
    {
      cl_int err;

      q = queue;

      OCL_CHECK(err, kernel = cl::Kernel(program,"Filter2DKernel", &err));

      // Allocate input and output buffers
      OCL_CHECK(err, coef_buffer = cl::Buffer(context, CL_MEM_READ_ONLY,  (FILTER_V_SIZE*FILTER_V_SIZE)*sizeof(char), nullptr, &err));
      OCL_CHECK(err, src_buffer  = cl::Buffer(context, CL_MEM_READ_ONLY,  (32*32)*sizeof(char), nullptr, &err));
      OCL_CHECK(err, dst_buffer  = cl::Buffer(context, CL_MEM_WRITE_ONLY, (16*16)*sizeof(char), nullptr, &err));

      // Set kernel arguments - this pins the buffers to specific global memory banks
      OCL_CHECK(err, err = kernel.setArg(0, coef_buffer));
      OCL_CHECK(err, err = kernel.setArg(6, src_buffer));
      OCL_CHECK(err, err = kernel.setArg(7, dst_buffer));

      // Make buffers resident in the device
      // If done after setArg, then buffers are pinned and runtime knows in which bank they should be made resident,
      // removing the need for using the vendor extensions to explicitely map to DDR.
      OCL_CHECK(err, err = q.enqueueMigrateMemObjects({coef_buffer, src_buffer, dst_buffer}, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED));

      // Make sure buffers are migrated before continuing
      q.finish();
    };
 
    void Filter2D(
      const char        coeffs[FILTER_V_SIZE][FILTER_H_SIZE],
      float             factor,
      short             bias,
      unsigned short    width,
      unsigned short    height,
      unsigned short    stride,
      unsigned char    *src,
      unsigned char    *dst ) 
    {

      assert(width  <= 32);
      assert(height <= 32);
      assert(stride%32 == 0);

      cl_int    err;
      cl::Event in1_event;
      cl::Event in2_event;
      cl::Event run_event;
      cl::Event out_event;

      size_t offset = 0;
      int nbytes = stride*height*sizeof(char);

      // If a previous transaction is pending, wait until it completes
      uint64_t dur = 0;
      dur = finish();
      printf("Return enqueueTask duration: %lu\n", dur);

          double kernel_time_in_sec = 0;
    std::chrono::duration<double> kernel_time(0);

      // Set kernel arguments - since buffers are reused, no need to set these args each time...
      // OCL_CHECK(err, err = kernel.setArg(0, coef_buffer));
      OCL_CHECK(err, err = kernel.setArg(1, factor));
      OCL_CHECK(err, err = kernel.setArg(2, bias));
      OCL_CHECK(err, err = kernel.setArg(3, width));
      OCL_CHECK(err, err = kernel.setArg(4, height));
      OCL_CHECK(err, err = kernel.setArg(5, stride));
      // OCL_CHECK(err, err = kernel.setArg(6, src_buffer));
      // OCL_CHECK(err, err = kernel.setArg(7, dst_buffer));

      // Schedule the writing of the inputs from host to device
      OCL_CHECK(err, err = q.enqueueWriteBuffer(coef_buffer, CL_FALSE, offset, (FILTER_V_SIZE*FILTER_V_SIZE)*sizeof(char), &coeffs[0][0], nullptr, &in1_event) );
      OCL_CHECK(err, err = q.enqueueWriteBuffer(src_buffer,  CL_FALSE, offset, nbytes, src, nullptr, &in2_event) );
      // events.push_back(in1_event);
      // events.push_back(in2_event);

        auto kernel_start = std::chrono::high_resolution_clock::now();
      // Schedule the execution of the kernel
      OCL_CHECK(err, err = q.enqueueTask(kernel, &events, &run_event));
        auto kernel_end = std::chrono::high_resolution_clock::now();
        kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
        kernel_time_in_sec = kernel_time.count();
        std::cout << "THROkernel_time_in_sec = " << kernel_time_in_sec << " s" << std::endl;


      events.push_back(run_event);

      // Schedule the reading of the outputs from device back to host
      OCL_CHECK(err, err = q.enqueueReadBuffer(dst_buffer, CL_FALSE, offset, nbytes/4, dst, &events, &out_event) );
      // events.push_back(out_event);       
    }

    uint64_t finish()
    {
      uint64_t duration_ns = 0;
      if (events.size()>0) {
        duration_ns = get_duration_ns(events);
        events.back().wait();
        events.clear();
        if (getenv("XCL_EMULATION_MODE") != NULL) {
          printf("  finished Filter2DRequest\n");
        }  
      }
      return duration_ns;   
    }
};

// -------------------------------------------------------------------------------------------
// Class used to manage multiple simultaneous request to the Filter2D kernel
// Implements SW pipelining and scales with multiple CUs
// -------------------------------------------------------------------------------------------

class Filter2DDispatcher
{
  std::vector<Filter2DRequest> req;
  int max;
  int cnt;

  public:
    Filter2DDispatcher(cl::Context &context, cl::Program &program, cl::CommandQueue &queue, int nreqs) 
    {
      cnt = 0;
      max = nreqs; 
      for(int i=0; i<max; i++) {
        req.push_back( Filter2DRequest(context, program, queue) );
      }     
    };
 
    int operator () (
      const char        coeffs[FILTER_V_SIZE][FILTER_H_SIZE],
      float             factor,
      short             bias,
      unsigned short    width,
      unsigned short    height,
      unsigned short    stride,
      unsigned char    *src,
      unsigned char    *dst ) 
    {
      cnt++;
      req[cnt%max].Filter2D(coeffs, factor, bias, width, height, stride, src, dst);
      return (cnt%max);
    }

    void finish(int id) {
      if (id<max) {
        req[id].finish();
      }
    }

    auto finish() {
      auto duration_ns = 0;
      for(int i=0; i<max; i++) {
        duration_ns += req[i].finish();
      }
      return duration_ns;
    }
};


int main(int argc, char** argv)
{
  printf("----------------------------------------------------------------------------\n");
  printf("\n"); 
  printf("Xilinx 2D Filter Example Application (Randomized Input Version)\n"); 
  printf("\n"); 
         
  // ---------------------------------------------------------------------------------
  // Parse command line
  // ---------------------------------------------------------------------------------

  CmdLineParser parser;
  parser.addSwitch("--nruns",   "-n", "Number of times to image is processed", "1");
  parser.addSwitch("--fpga",    "-x", "FPGA binary (xclbin) file to use");
  parser.addSwitch("--width",   "-w", "Image width",  "32");
  parser.addSwitch("--height",  "-h", "Image height", "32");
  parser.addSwitch("--filter",  "-f", "Filter type (0-6)", "0");
  parser.addSwitch("--maxreqs", "-r", "Maximum number of outstanding requests", "3");
  parser.addSwitch("--compare", "-c", "Compare FPGA and SW performance", "1", true);

  //parse all command line options
  parser.parse(argc, argv);
  string   fpgaBinary      = parser.value("fpga");
  unsigned int width       = parser.value_to_int("width");
  unsigned int height      = parser.value_to_int("height");
  unsigned int numRuns     = parser.value_to_int("nruns");
  unsigned int filterType  = parser.value_to_int("filter");
  unsigned int maxReqs     = parser.value_to_int("maxreqs");
  bool     comparePerf     = parser.value_to_bool("compare");

  if (fpgaBinary.size() == 0) {
    printf("ERROR: FPGA binary file (.xclbin) must be specified with the -x command line switch\n");
    return -1;
  }
  if ((width>MAX_IMAGE_WIDTH) || (height>MAX_IMAGE_HEIGHT)) {
    printf("ERROR: Maximum image size is %dx%d\n", MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT);
    return -1;
  }
  if (filterType>6) {
    printf("ERROR: Supported filter type values are [0:6]\n");
    return -1;
  }

  printf("FPGA binary       : %s\n", fpgaBinary.c_str());
  printf("Number of runs    : %d\n", numRuns);
  printf("Image width       : %d\n", width);
  printf("Image height      : %d\n", height);
  printf("Filter type       : %d\n", filterType);
  printf("Max requests      : %d\n", maxReqs);
  printf("Compare perf.     : %d\n", comparePerf);
  printf("\n"); 
    
  // ---------------------------------------------------------------------------------
  // Load XCLBIN file, create OpenCL context, device and program
  // ---------------------------------------------------------------------------------

  printf("Programming FPGA device\n");    
  cl_int err;
  std::vector<cl::Device> devices = xcl::get_xil_devices();
  devices.resize(1); // (arbitrarily) use the first Xilinx device that is found
  OCL_CHECK(err, cl::Context context(devices[0], NULL, NULL, NULL, &err));
  unsigned fileBufSize;
  char* fileBuf = xcl::read_binary_file(fpgaBinary.c_str(), fileBufSize);
  cl::Program::Binaries bins{{fileBuf, fileBufSize}};
  OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
  OCL_CHECK(err, cl::CommandQueue queue(context, devices[0], cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder, &err));

  // ---------------------------------------------------------------------------------
  // Read input image and format inputs
  // ---------------------------------------------------------------------------------
  
  // Read Input image
  unsigned int stride    = ceil(width/32.0)*32;
  unsigned int nbytes    = (stride*height);

  // Input and output buffers (Y,U, V)
  unsigned char *y_src = (unsigned char *)malloc(nbytes);
  unsigned char *u_src = (unsigned char *)malloc(nbytes);
  unsigned char *v_src = (unsigned char *)malloc(nbytes);
  unsigned char *y_dst = (unsigned char *)malloc(nbytes/4);
  unsigned char *u_dst = (unsigned char *)malloc(nbytes/4);
  unsigned char *v_dst = (unsigned char *)malloc(nbytes/4);

  // Generate random image data
  printf("Generating a random %dx%d input image\n", width, height);
  for (unsigned int i=0; i<nbytes; i++) {
  	// y_src[i] = rand();
  	// u_src[i] = rand();
  	// v_src[i] = rand();
    y_src[i] = 1;
  	u_src[i] = 1;
  	v_src[i] = 1;
  }
    for (unsigned int i=0; i<nbytes/4; i++) {
  	y_dst[i] = 0;
  	u_dst[i] = 0;
  	v_dst[i] = 0;
  }

  // Retrieve filter factor and bias
  float factor = filterFactors[filterType];
  short bias   = filterBiases[filterType];  

  // ---------------------------------------------------------------------------------
  // Make requests to kernel(s) 
  // ---------------------------------------------------------------------------------

  printf("Running FPGA accelerator on %d images\n", numRuns);    

  // Dispatcher of requests to the kernel
  // 'maxReqs' controls the maximum number of outstanding requests to the kernel 
  // and equates to the depth of SW pipelining.  
  Filter2DDispatcher Filter2DKernel(context, program, queue, maxReqs);

auto fpga_begin = std::chrono::high_resolution_clock::now();

  for(unsigned int n=0; n<numRuns; n++)
  {
    // Enqueue independent requests to Blur Y, U and V planes
    // Requests will run sequentially if there is a single kernel
    // Requests will run in parallel is there are two or more kernels
    Filter2DKernel(filterCoeffs[filterType], factor, bias, width, height, stride, y_src, y_dst);
    Filter2DKernel(filterCoeffs[filterType], factor, bias, width, height, stride, u_src, u_dst);
    Filter2DKernel(filterCoeffs[filterType], factor, bias, width, height, stride, v_src, v_dst);
  }
  auto duration_ns = Filter2DKernel.finish();

auto fpga_end = std::chrono::high_resolution_clock::now();
std::cout << "Duration using events profiling: " << duration_ns << " ns" << std::endl;

  // ---------------------------------------------------------------------------------
  // Compute reference results and compare
  // ---------------------------------------------------------------------------------

  if (comparePerf) {
  	printf("Running Software version\n");    
  }

  // Create output buffers for reference results
  unsigned char *y_ref = (unsigned char *)malloc(nbytes/4);
  unsigned char *u_ref = (unsigned char *)malloc(nbytes/4);
  unsigned char *v_ref = (unsigned char *)malloc(nbytes/4);

  unsigned numRunsSW = comparePerf?numRuns:1;

auto cpu_begin = std::chrono::high_resolution_clock::now();

  #pragma omp parallel for num_threads(3)
  for(unsigned int n=0; n<numRunsSW; n++)
  {
    // Compute reference results
    Filter2D(filterCoeffs[filterType], factor, bias, width, height, stride, y_src, y_ref);
    Filter2D(filterCoeffs[filterType], factor, bias, width, height, stride, u_src, u_ref);
    Filter2D(filterCoeffs[filterType], factor, bias, width, height, stride, v_src, v_ref);
  }

auto cpu_end = std::chrono::high_resolution_clock::now();

  printf("Comparing results\n");    

  // Compare results
  bool diff = false;
  for (unsigned int y = 0; y < height/2; y++) {
      for (unsigned int x = 0; x < width/2; x++) {
          if ( y_dst[y*stride/2+x] != y_ref[y*stride/2+x] ) diff = true;
          printf("%d: FPGA(%d) SW(%d)\n", y*stride/2+x, y_dst[y*stride/2+x], y_ref[y*stride/2+x]);
          if ( u_dst[y*stride/2+x] != u_ref[y*stride/2+x] ) diff = true;
          if ( v_dst[y*stride/2+x] != v_ref[y*stride/2+x] ) diff = true;
      }
  }

  if(diff) {
      printf("\n%sTest FAILED: Output has mismatches with reference%s\n", RED, RESET);    
  } else {
      printf("\n%sTest PASSED: Output matches reference%s\n", GREEN, RESET);    
  }

  // Report performance (if not running in emulation mode)
  if (getenv("XCL_EMULATION_MODE") == NULL) {
      std::chrono::duration<double> fpga_duration = fpga_end - fpga_begin;
      std::chrono::duration<double> cpu_duration  = cpu_end  - cpu_begin;
      float fpga_throughput = (double) numRuns*3*nbytes / fpga_duration.count() / (1024.0*1024.0);
      float cpu_throughput  = (double) numRuns*3*nbytes / cpu_duration.count() / (1024.0*1024.0);
      float fpga_events_throughput = (double) numRuns*3*nbytes / duration_ns; // Bytes/ns => GB/s
      printf("\n");
      printf("FPGA Time         : %10.4f s\n",    fpga_duration.count());
      printf("FPGA Throughput   : %10.4f MB/s\n", fpga_throughput);
      printf("FPGA Events Throughput   : %10.4f GB/s\n", fpga_events_throughput);
      printf("Total enqueueTask events duration: %d\n", duration_ns);
      std::cout << "Data Transfer Size: " << (double) numRuns*3*nbytes << " Bytes." << std::endl;
      // if (comparePerf) {
      printf("CPU  Time         : %10.4f s\n",    cpu_duration.count());
      printf("CPU  Throughput   : %10.4f MB/s\n", cpu_throughput);      
      printf("FPGA Speedup      : %10.4f x\n",    cpu_duration.count() / fpga_duration.count());  
      // }    
  }

  printf("----------------------------------------------------------------------------\n"); 

  return (diff?1:0);
}
