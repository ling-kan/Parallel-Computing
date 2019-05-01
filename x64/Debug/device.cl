#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
// ----------------------- REDUCTION ---------------------------------//
// reduce using local memory (so called privatisation)
// ----------- Reduction Sum Using Local Memory (Privatisation) Float ----------//
__kernel void sum_reduce(__global const float *input, __global float *output, __local float *l_mem){ 
	int g_id = get_global_id(0);
	int l_id = get_local_id(0);
	int l_size = get_local_size(0);

	l_mem[l_id] = input[g_id];		// Copy all loval size values from the global memory to local memory
	barrier(CLK_LOCAL_MEM_FENCE);   // Wait for all local threads to finish copying from global to local memory
	/*
		for (int i = 1; i < l_size; i *= 2) {
			if (!(l_id % (i * 2)) && ((l_id + i) < l_size)) 
			l_mem[lid] += l_mem[lid + i];
	*/
	for (int i = l_size/2; i > 0; i /= 2) { // Loop then divide into 2 work groups 
			if(l_id < i) {					// Add elements from stride to local id
				l_mem[l_id] += l_mem[l_id + i];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
	}  
	 if (l_id == 0){ 
		    output[g_id] = l_mem[0];		  // Write result into output array 
		}
}

// -------------- Reduction Min Float ------------//
__kernel void min_reduce(__global float *input, __global float *output, __local float *l_mem){ 
	int g_id = get_global_id(0);
	int l_id = get_local_id(0);
	int l_size = get_local_size(0); 

	l_mem[l_id] = input[g_id];		// Copy all loval size values from the global memory to local memory
	barrier(CLK_LOCAL_MEM_FENCE);   // Wait for all local threads to finish copying from global to local memory		
	
	//Loop local memory then divide into 2 work groups, Iterate through and find the minimum value in the list
	for (int i = l_size/2; i > 0; i /= 2) { // Loop then divide into 2 work groups 
			if(l_id < i) {	
			if (l_mem [l_id + i] < l_mem [l_id])	// Iterate through workgroup; If the local value is lower, the value is saved 
				l_mem [l_id] = l_mem [l_id + i];	// Add elements from local memory to local id	
		}
		barrier(CLK_LOCAL_MEM_FENCE); // Wait for all the iteration to complete
	}   
	 if (l_id == 0){ 
			output[g_id] = l_mem[0];  // Compute min and store minimum value in output
	}
}

//-------------- Reduction Max Float ------------//
__kernel void max_reduce(__global const float *input, __global float *output, __local float *l_mem){ 
	int g_id = get_global_id(0);
	int l_id = get_local_id(0);
	int l_size = get_local_size(0); 

	l_mem[l_id] = input[g_id];		// Copy all loval size values from the global memory to local memory
	barrier(CLK_LOCAL_MEM_FENCE);   // Wait for all local threads to finish copying from global to local memory
		
	//Loop local memory then divide into 2 work groups, Iterate through and find the maximum value in the list
	for (int i = l_size/2; i > 0; i /= 2) { // Loop then divide into 2 work groups 
			if(l_id < i) {	
			if (l_mem [l_id + i] > l_mem [l_id])	// Iterate through workgroup; If the local value is lower, the value is saved 
				l_mem [l_id] = l_mem [l_id + i];	// Add elements from local memory to local id	
		}
		barrier(CLK_LOCAL_MEM_FENCE);  // Wait for all the iteration to complete
	}   
	 if (l_id == 0){ 
			output[g_id] = l_mem[0];   //Compute max and store maximum value in output
	}
}
//-------------- Variance Float for Standard Deviation ------------//
__kernel void variance (__global const float* input, __global float* output, __local float* l_mem, float mean){ 
		int g_id = get_global_id(0);
		int l_id = get_local_id(0);
		int l_size = get_local_size(0);

	// Variance Formula = (nth Value - mean)2 / number of elements 
	float val = input[g_id] - mean;      // number of elemenets - mean
       	barrier(CLK_LOCAL_MEM_FENCE);	
		l_mem[l_id] = (val * val) ;		 // Square the value while copying over 
       	barrier(CLK_LOCAL_MEM_FENCE);     // Wait for all local threads to finish copying from global to local memory

	//Loop then divide into 2 work groups 
	for (int i = l_size/2; i > 0; i /= 2) {
			if(l_id < i) { 
				l_mem[l_id] += l_mem[l_id + i];  // Add elements from stride to local id
			}
			barrier(CLK_LOCAL_MEM_FENCE);
	}  
	 if (l_id == 0){ 
		    output[g_id] = l_mem[0];   // Write result into output array 
		}
}

//--------------  Sort the file to find (Bitonic Sort - Using Scan) ; Lower,Upper Quartile and Median ------------//
__kernel void sorting(__global const float* input, __global float* output, __local float* l_mem) {
	int g_id = get_global_id(0);
	int l_id = get_local_id(0);
	int l_size = get_local_size(0); // Work Group
	__local int *swap; //used for buffer swap

	l_mem[l_id] = input[g_id];	  // Cache all N values from global memory to local memory
	barrier(CLK_LOCAL_MEM_FENCE); // Wait for all local threads to finish copying from global to local memory
	
	// Load values 
	for (int i = l_size/2; i > 0; i /= 2) { // Loop and go through each element in the work group(l_size)
		/*	if (l_id >= i)
				l_mem[l_id] = l_mem[l_id] + l_mem[l_id - i];
			else
				l_mem[l_id] = l_mem[l_id];
		*/
		int l_pos = l_id ^ i; // local position for the value o compare 
		
		// If the value is smaller swap the value 
        bool smaller = (l_mem[l_pos] < l_mem[l_id]) || ( l_mem[l_pos] == l_mem[l_id] && l_pos < l_id );
		//directon = (( g_id  & l_id) != 0); // sort by ascending 
		bool swap = smaller ^ (l_pos < g_id) ^ (( g_id  & l_id) != 0);	
		barrier(CLK_LOCAL_MEM_FENCE);

		// Swapping the data based what value is smaller 
		l_mem[l_id] = (swap)?l_mem[l_pos]: l_mem[l_id];	 
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	output[g_id] = l_mem[l_id];	 //copy the cache to output array
}



__kernel void sorting2(__global const float* input, __global float* output, __local float* l_mem) {
	int g_id = get_global_id(0);
	int l_id = get_local_id(0);
	int l_size = get_local_size(0); // Work Group
	__local int *swap; //used for buffer swap


	l_mem[l_id] = input[g_id];	  // Cache all N values from global memory to local memory
	barrier(CLK_LOCAL_MEM_FENCE); // Wait for all local threads to finish copying from global to local memory
	
	// Load values 
    for (int i = 1; i < l_size; i <<= 1)	// Loop on sorted sequence length
    {
		for (int j = i; j > 0; j >>= 1)	// Loop on comparison distance (between keys)
		{ // Loop and go through each element in the work group(l_size)
			/*	if (l_id >= i)
					l_mem[l_id] = l_mem[l_id] + l_mem[l_id - i];
				else
					l_mem[l_id] = l_mem[l_id];
			*/
			int l_pos = l_id ^ i; // local position for the value o compare 
		    float a = l_mem[l_id];
			float b = l_mem[l_pos];

			// If the value is smaller swap the value 
			bool smaller = (b < a) || ( b == a && l_pos < l_id );
			//directon = (( g_id  & l_id) != 0); // sort by ascending 
			bool swap = smaller ^ (l_pos < l_id) ^ (( l_id  & i << 1) != 0);
			
			barrier(CLK_LOCAL_MEM_FENCE);

			// Swapping the data based what value is smaller 
			l_mem[l_id] = (swap)?l_mem[l_pos]: l_mem[l_id];	 
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
	output[g_id] = l_mem[l_id];	 //copy the cache to output array
}
