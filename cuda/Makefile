# Requires: nvcc (CUDA toolkit)
NVCC = nvcc
NVCCFLAGS = -arch=sm_87 -O2

flux_cuda: flux_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o flux_cuda flux_cuda.cu

test: flux_cuda
	./flux_cuda

clean:
	rm -f flux_cuda
