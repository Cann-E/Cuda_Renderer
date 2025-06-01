
EXECUTABLE := render

CU_FILES   := cudaRenderer.cu

CU_DEPS    :=

CC_FILES   := main.cpp display.cpp benchmark.cpp refRenderer.cpp \
              noise.cpp ppm.cpp sceneLoader.cpp

LOGS	   := logs

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -g -I ../freeglut-3.2.0/include
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS :=

NVCCFLAGS=-g -lineinfo -O3 -m64 --gpu-architecture compute_86 -maxrregcount=128
LIBS += GL glut cudart

LDFLAGS=-L/usr/local/cuda-12.6/lib64 -lcudart

LDFLAGS+=-L../freeglut-3.2.0/build/lib -lGL -lGLU -lglut 

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

OBJS=$(OBJDIR)/main.o $(OBJDIR)/display.o $(OBJDIR)/benchmark.o $(OBJDIR)/refRenderer.o \
     $(OBJDIR)/cudaRenderer.o $(OBJDIR)/noise.o $(OBJDIR)/ppm.o $(OBJDIR)/sceneLoader.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS)

check:	default
		./checker.pl

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@

submit:
	zip $(USER)_assignment3.zip Makefile cudaRenderer.cu
	cp $(USER)_assignment3.zip /mnt/data1/submissions/assignment3/
