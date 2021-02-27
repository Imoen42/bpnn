all: train test

CFLAGS = -std=c99 -Wall -Wextra -pedantic -Ofast -flto -march=native

LDFLAGS = -lm

CC = gcc

train: train.o bpnn.o

test: test.o bpnn.o

clean:
	rm -f *.o
