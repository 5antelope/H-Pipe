CXX ?= g++
GXX ?= g++

CXXFLAGS += -std=c++11 -g

LDFLAGS += -L/home/15-418/opencv/build/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
LDFLAGS += -L/afs/cs/academic/class/15418-s13/public/lib -lglog -lleveldb -lprotobuf
# LDFLAGS += -L/afs/cs/academic/class/15418-s13/public/lib -lglog -lgflags -lprotobuf -lleveldb
LDFLAGS += -L/usr/local/cuda/lib64/ -lcudart
LDFLAGS += -L/home/yangwu/lib -lsnappy
LDFLAGS += -L/home/yangwu/lmdb/libraries/liblmdb -llmdb
LDFLAGS += -L/afs/cs/academic/class/15418-s13/public/lib -lboost_system -lm
LDFLAGS += -L/afs/cs/academic/class/15418-s13/public/boost_1_53_0/bin.v2/libs/thread/build/gcc-4.8.0/release/link-static/threading-multi -lboost_thread

all: test_io

data.pb.h data.pb.cc: data.proto
	protoc data.proto --cpp_out=./

io.o: io.cpp io.hpp data.pb.h
	$(CXX) $(CXXFLAGS) -c -Wall io.cpp

db.o: db.cpp db.hpp data.pb.h
	$(CXX) $(CXXFLAGS) -c -Wall db.cpp

test_io: io.o db.o test_io.cpp data.pb.h data.pb.cc
	$(CXX) $(CXXFLAGS) data.pb.cc test_io.cpp io.o db.o -o test_io.out $(LDFLAGS)

clean:
	rm -f io.o db.o test_io.out data.pb.h data.pb.cc *.jpg
