CC=gcc
OPTFLAGS=-O3 -ffast-math
CFLAGS=-lm
DEBUGFLAGS=-g -pg
PYTHON=python3

serial: main.c basic_serial.c scenarios.c
	$(CC) $^ -o $@ $(CFLAGS) $(OPTFLAGS)

debug: main.c basic_serial.c scenarios.c
	$(CC) $^ -o $@ $(CFLAGS) $(DEBUGFLAGS)

.PHONY: clean
clean:
	rm -f main
	rm -f *.o