CFLAGS=-Wall -O3 -march=native -g -fopenmp

OBJ=$(patsubst %.c,%.o,$(wildcard *.c))

all: main
main: $(OBJ)

clean:
	rm $(OBJ) main

test: main
	./main input/mini.mnt

# si un .h ou le Makefile change tout recompiler :
$(OBJ): $(wildcard *.h) Makefile
