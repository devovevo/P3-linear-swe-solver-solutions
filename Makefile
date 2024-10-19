CPP=g++
OPTFLAGS=-O3 -ffast-math
CFLAGS=-lm
DEBUGFLAGS=-g -pg
PYTHON=python3

serial/serial: common/main.cpp common/scenarios.cpp serial/basic_serial.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

.PHONY: clean
clean:
	rm -f serial/serial
	rm -f gpu/gpu
	rm -f mpi/mpi