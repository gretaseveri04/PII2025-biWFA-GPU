CUDA_COMPILER = nvcc
CUDA_FLAGS = -use_fast_math -O3 -std=c++11
CUDA_DEBUG_FLAGS = -G -O2 -Xcompiler -rdynamic -lineinfo -std=c++11 -rdc=true -Xptxas=-v -use_fast_math
CUDA_ARCH_FLAG = -arch=sm_86

WFA_LIB = -IWFA2-lib -LWFA2-lib/lib -lwfa

CXX = g++

all: cuda cpp

cpp: 
	$(CXX) src/biWFA.cpp -o biwfa_cpp

cuda: 
	$(CUDA_COMPILER) $(CUDA_FLAGS) src/biWFA.cu -o biwfa_cuda $(WFA_LIB)

debug: 
	$(CUDA_COMPILER) $(CUDA_DEBUG_FLAGS) src/*.cu -o biwfa_debug 

	
.PHONY: clean

clean:
	rm -f biwfa_*
	