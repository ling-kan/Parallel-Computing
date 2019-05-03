// --------- Sources -----//
// https://stackoverflow.com/questions/53849/how-do-i-tokenize-a-string-in-c%EF%BC%89
// http://www.bealto.com/gpu-sorting_parallel-bitonic-local.html
// https://github.com/kevin-albert/cuda-mergesort/blob/master/mergesort.cu
// https://onezork.wordpress.com/2014/08/29/gpu-mergesort/

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
//#define CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE //returnes information about kernel object specfic to the device 

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <iterator>
#include <iomanip> // for std::setprecision - Manages the precision (i.e. how many digits are generated) of floating point output 
#include <CL/cl.hpp>
#include <malloc.h>
#include <stdlib.h>
#include "Utils.h"
#include <algorithm>
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

using namespace std;
void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

vector<string> split(const char *str, char space = ' ')
{
	// Go through each line and store it as a string
	vector<string> result;
	do
	{
		// Pointer for the start of the string 
		const char *start = str; 
		// If there is a space then add to a new string
		while (*str != space && *str)
			str++;
		result.push_back(string(start, str)); 
		// Appends character to the end of the string
		// increasing its value by one
	} while (0 != *str++);
	return result;
}

int main(int argc, char **argv) {
	// ---------- 1. handle command line options such as device selection, verbosity, etc. ----------//
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	// Read file, load data in host memory
	string line;
	string filename;
	filename = "temp_lincolnshire_short.txt";
	ifstream textFile(filename);
	//Initating vectors
	vector<string> places;
	vector<string> dateTimes;
	vector<float> temp;
	
	// Open the file 
	if (textFile.is_open())
	{
		//Get line in file
		while (getline(textFile, line))
		{
			vector<std::string> x = split(line.c_str(), ' '); //Split vector
			// x.at(i) - return the letter at position i in the string/vector.
			places.push_back(x.at(0)); //First set of data (vector) is places
			dateTimes.push_back(x.at(1) + " " + x.at(2) + " " + x.at(3) + " " + x.at(4)); //Date and time 
			temp.push_back(stof(x.at(5))); //Temperature // stof- convert string to float
		}
		// Close the file once compelete
		textFile.close();
		cout << filename << " File Uploaded!" << endl;
	}
	// If error with file show the following message
	else cout << "Unable to open file\n" << endl;

	// Detect any potential exceptions
	try {
		// ----------- 2.  Host operations -----------//
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		// Display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		// Create a queue to which we will push commands for the device
		// profiling needs to be enable when the queue is created: 
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;
		AddSources(sources, "device.cl");
		cl::Program program(context, sources);

		// Build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//assign alternative names to existing datatypes
		typedef int mytype;

		// ------------ 4. Memory allocation --------- //
		//host - input
		//allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results!
	   //std::vector<mytype> A(10, 1);

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = 1024; //1024; //work group size - higher work group size can reduce 
		size_t padding_size = temp.size() % local_size;
		size_t localpad_size = local_size - padding_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			vector <int> temp_ext(localpad_size, 0);
			//append that extra vector to our input
			temp.insert(temp.end(), temp_ext.begin(), temp_ext.end());
		}

		size_t input_elements = temp.size(); //number of input elements
		size_t input_size = input_elements * sizeof(mytype); //size in bytes
		size_t nr_groups = input_elements / local_size; //partial sum needed for reduction - this is removed later 

		//host - output
		//size_t output_size = input_elements * sizeof(mytype);//size in bytes
		size_t output_size = input_size;//size in bytes


		std::vector<mytype> RMin(output_size);
		std::vector<mytype> RMax(output_size);
		std::vector<mytype> RSum(output_size);
		std::vector<mytype> RVar(output_size);
		std::vector<mytype> RSort(output_size);

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_RMin(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_RMax(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_RSum(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_RVar(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_RSort(context, CL_MEM_READ_WRITE, output_size);

		// ---------- 5. Device operations ----------- //
		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &temp[0]);
		//zero buffer on device memory for the output
		queue.enqueueFillBuffer(buffer_RMin, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_RMax, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_RSum, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_RVar, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_RSort, 0, 0, output_size);

		//5.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_1 = cl::Kernel(program, "min_reduce");
		kernel_1.setArg(0, buffer_A); //Input
		kernel_1.setArg(1, buffer_RMin); //Output
		kernel_1.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

		cl::Kernel kernel_2 = cl::Kernel(program, "max_reduce");
		kernel_2.setArg(0, buffer_A); //Input
		kernel_2.setArg(1, buffer_RMax); //Output
		kernel_2.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

		cl::Kernel kernel_3 = cl::Kernel(program, "sum_reduce");
		kernel_3.setArg(0, buffer_A); //Input
		kernel_3.setArg(1, buffer_RSum); //Output
		kernel_3.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

	/*	int global_threads = (temp.size() / 256) * 256;
		global_threads += 256;
		std::cout << "threads: " << global_threads << std::endl;*/

		// ------------- Profiling Event ----------- //
		// Kernel - An unsigned 64-bit integer (cl_ulong)
		cl_ulong start_kernel, end_kernel;

		// Need to get profiling events to measure how long it takes for a kernel to run 
		// Link  event when launching a kernel
		cl::Event prof_event1;
		cl::Event prof_event2;
		cl::Event prof_event3;

		// call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event1);
		queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event2);
		queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event3);
		
		//allocating memory on the host (C compatible with openCL as it uses C pointers) 
		//Pointer Sum/min/max would like (temp.size() * sizeof(float))amount of bytes of (malloc) memory allocated 
		float* Min = (float*)malloc(temp.size() * sizeof(float));
		float* Max = (float*)malloc(temp.size() * sizeof(float));
		float* Sum = (float*)malloc(temp.size() * sizeof(float));

		//5.3 Copy the result from device to host;
		queue.enqueueReadBuffer(buffer_RMin, CL_TRUE, 0, output_size, &Min[0]);
		queue.enqueueReadBuffer(buffer_RMax, CL_TRUE, 0, output_size, &Max[0]);
		queue.enqueueReadBuffer(buffer_RSum, CL_TRUE, 0, output_size, &Sum[0]);

		// Wait for all the enqueue tasks to finish
		prof_event1.wait();
		prof_event2.wait();
		prof_event3.wait();

		prof_event1.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_kernel);
		prof_event1.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_kernel);
		cl_ulong prof_time1 = (end_kernel - start_kernel);

		prof_event2.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_kernel);
		prof_event2.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_kernel);
		cl_ulong prof_time2 = (end_kernel - start_kernel);

		prof_event3.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_kernel);
		prof_event3.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_kernel);
		cl_ulong prof_time3 = (end_kernel - start_kernel);

		//5.3 Copy the result from device to host;
		//queue.enqueueReadBuffer(buffer_RSum, CL_TRUE, 0, output_size, &result[0]);
		queue.enqueueReadBuffer(buffer_RSum, CL_TRUE, 0, output_size, &Sum[0]);
		queue.enqueueReadBuffer(buffer_RMin, CL_TRUE, 0, output_size, &Min[0]);
		queue.enqueueReadBuffer(buffer_RMax, CL_TRUE, 0, output_size, &Max[0]);
	//	queue.enqueueReadBuffer(buffer_RSum2, CL_TRUE, 0, output_size, &RSum2[0]);
		//queue.enqueueReadBuffer(buffer_RMin2, CL_TRUE, 0, output_size, &RMin2[0]);
		//queue.enqueueReadBuffer(buffer_RMax2, CL_TRUE, 0, output_size, &RMax2[0]);*/

		queue.finish();

		// Amount of elements - removing padding 
		auto num_elements = (input_elements - localpad_size);
		auto min_temp = 0.0f;
		auto max_temp = 0.0f;
		auto sum_temp = 0.0f;

		for (size_t i = 0; i < num_elements; i += local_size)
		{
			// MIN - Find the minimum value in the workgroup to get the final value
			if (Min[i] < min_temp) {
				min_temp = Min[i];
			}
			// MAX - Find the maximum value in the workgroup to get the final value
			if (Max[i] > max_temp) {
				max_temp = Max[i];
			}
			//  SUM - Add all values to get the sum  
			sum_temp += Sum[i];
		}

		float mean = sum_temp / num_elements;
		// ---------------- Print Values ----------------//
		cout << "\n\t------------- Reduction (Float) ------------" << endl;
		cout << "Min (Float):\t\t" << min_temp << "\t\tProfiling Info (nSec): " << prof_time1 << endl;
		cout << "Max (Float):\t\t" << max_temp << "\t\tProfiling Info (nSec): " << prof_time2 << endl;
		cout << "Sum (Float):\t\t" << sum_temp << "\t\tProfiling Info (nSec): " << prof_time3 << endl;
		cout << "Num of Elements:\t" << num_elements << endl;
		cout << "Average:\t\t" << mean << endl;

		//----------  To calculate variance and standard dev ----------------------//
		//5.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_4 = cl::Kernel(program, "variance");
		kernel_4.setArg(0, buffer_A); //Input
		kernel_4.setArg(1, buffer_RVar); //Output
		kernel_4.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size
		//kernel_4.setArg(2, input_elements);//local memory size
		kernel_4.setArg(3, mean);

		cl::Kernel kernel_5 = cl::Kernel(program, "sorting2");
		kernel_5.setArg(0, buffer_A); //Input
		kernel_5.setArg(1, buffer_RSum); //Output
		kernel_5.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

		// Need to get profiling events to measure how long it takes for a kernel to run 
		cl::Event prof_event4; 
		cl::Event prof_event5;

		// call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_4, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event4);
		queue.enqueueNDRangeKernel(kernel_5, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event5);

		// Allocating memory on the host
		float* Var = (float*)malloc(temp.size() * sizeof(float));
		float* Srt = (float*)malloc(temp.size() * sizeof(float));
	
		//5.3 Copy the result from device to host;
		// Put the result back into the host (output)
		queue.enqueueReadBuffer(buffer_RVar, CL_TRUE, 0, output_size, &Var[0]);
		queue.enqueueReadBuffer(buffer_RVar, CL_TRUE, 0, output_size, &Srt[0]);

		// Wait for all the enqueue tasks to finish
		prof_event4.wait();
		prof_event5.wait();

		// Get Profiling information
		prof_event4.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_kernel);
		prof_event4.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_kernel);
		cl_ulong prof_time4 = (end_kernel - start_kernel);

		prof_event5.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_kernel);
		prof_event5.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_kernel);
		cl_ulong prof_time5 = (end_kernel - start_kernel);

		auto variance = 0.0f; // Create a variable to store the variance
		auto sorting = 0.0f; // Create a variable to store the variance
		// Loop each work group to get the final value 
		for (size_t i = 0; i < num_elements; i += local_size)
		{
			variance += Var[i];
			sorting = Srt[i];
//			cout << sorting << endl;
		}

		float stdev = sqrt(variance /num_elements); // square root variance to find standard dev
		cout << "\nVariance:\t\t" << variance << "\t\tProfiling Info (nSec): " << prof_time4 << endl;
		cout << "Standard Dev:\t\t" << stdev << endl;
		cout << "\n\t------------- Sorting (Bitonic Sort) ------------" << endl;
		cout << "1st Quart:\t\t" << sorting << "\t\tProfiling Info (nSec): " << prof_time2 << endl;
		cout << "Median:\t\t\t" << "0" << "\t\tProfiling Info (nSec): " << prof_time2 << endl;
		cout << "3rd Quart:\t\t" << "0" << "\t\tProfiling Info (nSec): " << prof_time2 << endl;

		cout << "\nExpected Values for  temp_lincolnshire_short.txt\nTotal of 18732 temperatures processed" << endl;
		cout << "AVG = 9.73 \tMIN = -25.00 \tMAX = 31.50 \nSTD = 5.91 \t1QT = 5.10 \t3QT = 14.0  \tMED = 9.80" << endl;
		cout << "\nExpected Values for  temp_lincolnshire.txt\nTotal of 18732 temperatures processed" << endl;
		cout << "AVG = 9.77 \tMIN = -25.00 \tMAX = 45.50 \nSTD = 5.92 \t1QT = 5.30 \t3QT = 14.0  \tMED = 9.80" << endl;

	
		// Press any key to continue 
		system("pause");
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
 }
