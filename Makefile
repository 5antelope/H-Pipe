OBJDIR=objs
SRCDIR=src

.PHONY: all clean

CXX=g++


# this flag only works for linux machine, not Max OS
CXXFLAGS+=-g -fopenmp -Wall -Wextra -O2 -std=c++11

INCLUDES = -I/home/15-418/Halide/include \
   -I/home/15-418/protobuf-2.6.1/include

# define library paths in addition to /usr/lib
LDFLAGS = -L/home/15-418/Halide/bin -lHalide \
		  -L/home/15-418/protobuf-2.6.1/lib -lprotobuf \
		  -L/afs/cs/academic/class/15418-s13/public/lib -lglog

# define any libraries to link into executable:
LIBS = -ldl

EXTRA_SCRIPTS = `pkg-config --libs protobuf libpng`

OBJS=$(OBJDIR)/main
CCFILES = $(SRCDIR)/main.cpp

all: dirs $(OBJDIR)
			$(CXX) $(CXXFLAGS) -o $(OBJS) $(CCFILES)

dirs:
	mkdir -p $(OBJDIR)

test: layer_test.cpp layers.h
		$(CXX) $(CXXFLAGS) layer_test.cpp data/data.pb.cc \
			           -o layer_test.out $(LDFLAGS) $(INCLUDES)

clean:
	rm -rf $(OBJDIR) *.out

