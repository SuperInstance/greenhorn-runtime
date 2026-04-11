.PHONY: build test clean

build:
	go build -o greenhorn ./cmd/greenhorn

test:
	go test -v ./pkg/flux/
	go test -v ./pkg/profiler/

clean:
	rm -f greenhorn
