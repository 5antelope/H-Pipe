OBJDIR=objs
SRCDIR=src
INCDIR=include
LIBDIR=lib

.PHONY: all clean

CXX=g++

# this flag only works for linux machine, not Max OS
CXXFLAGS+=-Wall -Wextra -O2 -I$(INCDIR) -L$(LIBDIR) -lHalide -lpthread -ldl -std=c++11

OBJS=$(OBJDIR)/main
CCFILES = $(SRCDIR)/main.cpp

all: dirs $(OBJDIR)
			$(CXX) $(CXXFLAGS) -o $(OBJS) $(CCFILES)

dirs:
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR)

