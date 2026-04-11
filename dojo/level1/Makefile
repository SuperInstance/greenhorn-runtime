CXX=g++
CXXFLAGS=-std=c++17 -Wall -O2
FLUX_SRC=../../cpp

validate: validate.cpp
	$(CXX) $(CXXFLAGS) -I$(FLUX_SRC) -o validate validate.cpp $(FLUX_SRC)/flux_vm.cpp

test: validate
	./validate

clean:
	rm -f validate
