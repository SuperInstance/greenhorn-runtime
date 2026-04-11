CXX=g++
CXXFLAGS=-std=c++17 -Wall -O2

test: flux_vm
	timeout 5 ./flux_vm

flux_vm: test_flux_vm.cpp flux_vm.cpp flux_vm.hpp
	$(CXX) $(CXXFLAGS) -o flux_vm test_flux_vm.cpp flux_vm.cpp

clean:
	rm -f flux_vm
