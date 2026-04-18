# FLUX CUDA Build
# Requires: nvcc (CUDA toolkit)
#
# Targets:
#   make              - Build both CUDA kernels
#   make batch        - Build the batch kernel (batch_kernel.cu)
#   make test         - Build and run native CUDA tests
#   make gotest       - Run Go tests (CPU fallback, no CUDA needed)
#   make gotest-gpu   - Run Go tests with GPU (requires CUDA)
#   make clean        - Remove build artifacts

NVCC ?= nvcc
NVCCFLAGS ?= -arch=sm_87 -O2

.PHONY: all batch test gotest gotest-gpu clean

all: flux_cuda batch_kernel

# Legacy single-program kernel
flux_cuda: flux_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o flux_cuda flux_cuda.cu

# Batch execution kernel (Phase 1)
batch_kernel: batch_kernel.cu batch_executor.cuh
	$(NVCC) $(NVCCFLAGS) -DFLUX_CUDA_AVAILABLE -c batch_kernel.cu -o batch_kernel.o

# Run native CUDA test
test: flux_cuda
	./flux_cuda

# Go tests (CPU fallback, no CUDA needed)
gotest:
	cd .. && go test -v ./cuda/

# Go tests with CUDA support
gotest-gpu:
	cd .. && go test -v -tags cuda ./cuda/

clean:
	rm -f flux_cuda flux_cuda_kernel batch_kernel.o
