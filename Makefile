OBJDIR=objs
SRCDIR=src

.PHONY: all clean

CXX=g++

# this flag only works for linux machine, not Max OS
CXXFLAGS+=-g -fopenmp -Wall -Wno-unused-parameter -O2 -std=c++11

INCLUDES = -I./include \
		   -I./proto \
		   -I/home/15-418/Halide/include \
		   -I/home/15-418/Halide/tools \
		   -I/home/15-418/protobuf-2.6.1/include  \
		   -I/home/yangwu/git/H-Net/proto \
		   -I/home/15-418/gflags-2.1.2/include \
		   -I/home/15-418/caffe2/caffe2/core \
		   `pkg-config --cflags-only-I protobuf`

# define library paths in addition to /usr/lib
LDFLAGS = -L/home/15-418/Halide/bin -lHalide -ldl\
		  -L/home/15-418/protobuf-2.6.1/lib -lprotobuf \
		  -L/afs/cs/academic/class/15418-s13/public/lib -lglog \
		  #-L/home/15-418/gflags-2.1.2/lib -lgflags

# define any libraries to link into executable:
LIBS = -ldl

EXTRA_SCRIPTS = `pkg-config --libs protobuf libpng`

CCFILES = $(wildcard ./src/*.cpp)

OBJS=$(OBJDIR)/main

all: dirs $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INC) $(CCFILES) -o $(OBJS)  $(LDFLAGS) $(INCLUDES) $(EXTRA_SCRIPTS)

dirs:
	mkdir -p $(OBJDIR)

io: test/test_io.cpp
	$(CXX) $(CXXFLAGS) $(INC) $(CCFILES) test/test_io.cpp -o test_io.out $(LDFLAGS) $(INCLUDES) $(EXTRA_SCRIPTS)

test: dirs
	$(CXX) $(CXXFLAGS) $(INC) $(CCFILES) test/test_layers.cpp -o $(OBJDIR)/test_layer.out $(LDFLAGS) $(INCLUDES) $(EXTRA_SCRIPTS)

clean:
	rm -rf $(OBJDIR) *.out

