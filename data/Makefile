CXX ?= g++
GXX ?= g++

CXXFLAGS += -std=c++11 -g

INC += -I/home/15-418/protobuf-2.6.1/include

# LDFLAGS += -L/home/15-418/protobuf-3.0.0-beta-2/lib -lprotobuf
LDFLAGS += -L/home/15-418/protobuf-2.6.1/lib -lprotobuf
LDFLAGS += -L/home/15-418/opencv/build/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
LDFLAGS += -L/afs/cs/academic/class/15418-s13/public/lib -lglog -lleveldb
LDFLAGS += -L/usr/local/cuda/lib64/ -lcudart
LDFLAGS += -L/home/yangwu/lib -lsnappy
LDFLAGS += -L/home/yangwu/lmdb/libraries/liblmdb -llmdb
LDFLAGS += -L/afs/cs/academic/class/15418-s13/public/lib -lboost_system -lm
LDFLAGS += -L/afs/cs/academic/class/15418-s13/public/boost_1_53_0/bin.v2/libs/thread/build/gcc-4.8.0/release/link-static/threading-multi -lboost_thread

all: test_io

data.pb.h data.pb.cc: data.proto
	protoc data.proto --cpp_out=./

io.o: io.cpp io.hpp data.pb.h
	$(CXX) $(CXXFLAGS) $(INC) -c -Wall io.cpp

# test_io: io.o test_io.cpp data.pb.h data.pb.cc
# 	$(CXX) $(CXXFLAGS) data.pb.cc test_io.cpp io.o -o test_io.out $(LDFLAGS)

test_io: test_io.cpp data.pb.h data.pb.cc
	$(CXX) $(CXXFLAGS) $(INC) data.pb.cc test_io.cpp -o test_io.out $(LDFLAGS)

test_protoc: test_protoc.cpp data.pb.h data.pb.cc
	$(CXX) $(CXXFLAGS) $(INC) data.pb.cc test_protoc.cpp -o test_protoc.out $(LDFLAGS)

clean:
	rm -f io.o test_io.out data.pb.h data.pb.cc
