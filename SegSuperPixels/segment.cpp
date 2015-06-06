/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#include <cstdio>
#include <cstdlib>
#include "image.h"
#include "misc.h"
#include "pnmfile.h"
#include "segment-image.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <cstring>
#include <vl/fisher.h>
#include <vl/gmm.h>
#include <vl/slic.h>

using namespace std;
using namespace cv;


int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: %s sigma k min input(ppm) output(ppm)\n", argv[0]);
    return 1;
  }
  
  float sigma = atof(argv[1]);
  float k = atof(argv[2]);
  int min_size = atoi(argv[3]);
	
  printf("loading input image.\n");
  
  //code chunfang wang added
  Mat image1;
  image1 = imread(argv[4], CV_LOAD_IMAGE_COLOR);
    if(! image1.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }
  //since images in OpenCV are always stored in the format of BGR, 
  int width = image1.cols;
  int height = image1.rows;
  image<rgb> *im = new image<rgb>(width, height);
  for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
        imRef(im, x, y).b = image1.at<Vec3b>(y,x)[0];
        imRef(im, x, y).g = image1.at<Vec3b>(y,x)[1];
        imRef(im, x, y).r = image1.at<Vec3b>(y,x)[2];
      }  
    }

    // namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    // imshow( "Display window", image1 );                   // Show our image inside it.
    // waitKey(0);                                          // Wait for a keystroke in the window

//  image<rgb> *input = loadPPM(argv[4]);
 printf("processing\n");
 int num_ccs; 
 image<rgb> *seg = segment_image(im, sigma, k, min_size, &num_ccs); 
 
 //save *seg into a jpeg image
 for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
        image1.at<Vec3b>(y,x)[0] = imRef(seg, x, y).b;
        image1.at<Vec3b>(y,x)[1] = imRef(seg, x, y).g;
        image1.at<Vec3b>(y,x)[2] = imRef(seg, x, y).r;
      }  
    }

//save image into OpenCV
 imwrite(argv[5], image1 );
 printf("got %d components\n", num_ccs);
 printf("done! uff...thats hard work.\n");

//try the vl_slic(im, regionSize, regularizer), where im is a single array
regionSize = 10;
regularizer = 10;
segments = vl_slic(im->data, regionSize, reglarizer);

  return 0;
}
