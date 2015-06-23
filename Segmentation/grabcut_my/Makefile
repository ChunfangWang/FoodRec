INCDIR = -I. -I /Users/Chunfang/Documents/RecogProj/vlfeat-0.9.20
DBG    = -g -Weverything -std=c++98
OPT    = -O0
CPP    = g++
CFLAGS = $(DBG) $(INCDIR) 
LINK   =  -L /Users/Chunfang/Documents/RecogProj/vlfeat-0.9.20/bin/maci64 -lm -ltiff -lopencv_highgui -lopencv_core \
	-lopencv_imgproc -lopencv_calib3d -lopencv_features2d -lopencv_nonfree -lopencv_photo -lopencv_objdetect \
	-lvl

.cpp.o:
	$(CPP) $(CFLAGS) -c $< -o $@

all: grabcut

#segment: segment.cpp $(CPP) $(CFLAGS) -o segment segment.cpp $(LINK)
grabcut: grabcut.cpp 
	$(CPP) $(CFLAGS) -o grabcut grabcut.cpp $(LINK) 

clean:
	/bin/rm -f grabcut *.o

clean-all: clean
	/bin/rm -f *~ 



