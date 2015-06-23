INCDIR = -I. -I ../vlfeat-0.9.20/
DBG    = -g -Weverything -std=c++11
OPT    = -O0
CPP    = g++
CFLAGS = $(DBG)  $(INCDIR) 
LINK   =  -L ../vlfeat-0.9.20/bin/maci64 -lm -ltiff -lopencv_highgui -lopencv_core \
	-lopencv_imgproc -lopencv_calib3d -lopencv_features2d -lopencv_nonfree -lopencv_photo -lopencv_objdetect \
	-lvl

.cpp.o:
	$(CPP) $(CFLAGS) -c $< -o $@

all: pca_opencv

#segment: segment.cpp $(CPP) $(CFLAGS) -o segment segment.cpp $(LINK)
pca_opencv: pca_opencv.cpp 
	$(CPP) $(CFLAGS) -o pca_opencv pca_opencv.cpp $(LINK) 

clean:
	/bin/rm -f pca_opencv *.o

clean-all: clean
	/bin/rm -f *~ 



