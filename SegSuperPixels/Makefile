INCDIR = -I. -I ../vlfeat-0.9.20/
DBG    = -g
OPT    = -O0
CPP    = g++
CFLAGS = $(DBG)  $(INCDIR)
LINK   =  -lm -ltiff  -lopencv_highgui -lopencv_core -lvl

.cpp.o:
	$(CPP) $(CFLAGS) -c $< -o $@

all: segment

segment: segment.cpp segment-image.h segment-graph.h disjoint-set.h
	$(CPP) $(CFLAGS) -o segment segment.cpp $(LINK)

clean:
	/bin/rm -f segment *.o

clean-all: clean
	/bin/rm -f *~ 



