CC=g++
CFLAGS=-Wall --std=c++11
SOURCES := $(wildcard *.cpp)
OBJS := $(SOURCES:.cpp=.o)

.PHONY: clean

optimized: CFLAGS += -Ofast -march=native -mtune=native
optimized: xstream 

debug: CFLAGS += -DDEBUG -fsanitize=address -fsanitize=undefined -g
debug: xstream

profile: CFLAGS += -L/home/localdirs/stufs1/emanzoor/.local/lib -Wl,--no-as-needed,-lprofiler,--as-needed -g
profile: xstream

xstream: $(OBJS)
		$(CC) $(CFLAGS) $^ -o $@

.cpp.o:
		$(CC) $(CFLAGS) -c $<

clean:
		rm -f xstream $(OBJS)
