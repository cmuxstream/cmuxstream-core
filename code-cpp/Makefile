CC=g++
CFLAGS=-Wall -g --std=c++11
SOURCES := $(wildcard *.cpp)
OBJS := $(SOURCES:.cpp=.o)

.PHONY: clean

optimized: CFLAGS += -Ofast -march=native -mtune=native
optimized: xstream 

debug: CFLAGS += -DDEBUG -DVERBOSE -g
debug: xstream

xstream: $(OBJS)
		$(CC) $(CFLAGS) $^ -o $@

.cpp.o:
		$(CC) $(CFLAGS) -c $<

clean:
		rm -f xstream $(OBJS)
