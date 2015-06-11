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
#include <ctime>
// #include "image.h"
// #include "misc.h"
// #include "pnmfile.h"
// #include "segment-image.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <iterator>

extern "C" {
  #include "vl/generic.h"
  #include "vl/slic.h"
  #include "vl/fisher.h"
  #include "vl/gmm.h"
}

using namespace std;
using namespace cv;


void printSegmentation(vl_uint32* &segmentation, Mat &mat, vl_size height, vl_size width
  , vl_size channels)
{
  // Convert segmentation, and also store the superpixel regions
    int** labels = new int*[mat.rows];
    int num_superpixels = 0;
    for (int i = 0; i < mat.rows; ++i) {
        labels[i] = new int[mat.cols];

        for (int j = 0; j < mat.cols; ++j) {
            labels[i][j] = (int) segmentation[j + mat.cols*i];
            cout<<labels[i][j]<<" ";
        }
        cout<<endl;
    }

    // Compute a contour image: this actually colors every border pixel
    // red such that we get relatively thick contours.
    int label = 0;
    int labelTop = -1;
    int labelBottom = -1;
    int labelLeft = -1;
    int labelRight = -1;

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {

            label = labels[i][j];

            labelTop = label;
            if (i > 0) {
                labelTop = labels[i - 1][j];
            }

            labelBottom = label;
            if (i < mat.rows - 1) {
                labelBottom = labels[i + 1][j];
            }

            labelLeft = label;
            if (j > 0) {
                labelLeft = labels[i][j - 1];
            }

            labelRight = label;
            if (j < mat.cols - 1) {
                labelRight = labels[i][j + 1];
            }

            if (label != labelTop || label != labelBottom || label!= labelLeft || label != labelRight) {
                mat.at<cv::Vec3b>(i, j)[0] = 0;
                mat.at<cv::Vec3b>(i, j)[1] = 0;
                mat.at<cv::Vec3b>(i, j)[2] = 255;
            }
        }
    }

    // Save the contour image.
    cv::imwrite("SuperPixel_contour.jpg", mat);
}

float* IFVEncode(float* data, int dimensions, int numData, int numClusters)
{
    VlGMM* gmm = vl_gmm_new(VL_TYPE_FLOAT, dimensions, numClusters);
    vl_gmm_cluster(gmm, data, numData);
    float* enc = (float*)vl_malloc(sizeof(float) * 2 * dimensions * numClusters);
    vl_fisher_encode(enc, VL_TYPE_FLOAT,  vl_gmm_get_means(gmm), dimensions, numClusters,
    vl_gmm_get_covariances(gmm), vl_gmm_get_priors(gmm), data, numData, VL_FISHER_FLAG_IMPROVED);

    //PCA_Whitening of the IFV feature vector
    return enc;
}

//return a feature vector of size 8,576 values
vector<float> featureVofSuperpixel(int x_min, int x_max, int y_min, int y_max, 
    vector<KeyPoint> &keypoints, Mat &descriptors, Mat &mat_Lab)
{
    //find the index of keypoints which fall in this superpixel region
    vector<KeyPoint>::iterator it;
    int sp_n = 0; //the index id
    vector<float> sp_descriptors;
    for (it = keypoints.begin(); it < keypoints.end(); it++, sp_n++)
    {
        if ((*it).pt.x <= x_max && (*it).pt.y <= y_max && 
            (*it).pt.x >= x_min && (*it).pt.y >= y_min)
        {
            descriptors.row(i).copyTo(sp_descriptors);
        }      
    }

    cout<<"Number of keypoints in this superpixel region: "<<sp_n<<endl;
    //Use Improved Fisher Vector to encode the two feature channel.
    //Feature Channel 1: SURF@64
    float* enc_1 = IFVEncode((float*)sp_descriptors, descriptors.cols, sp_n, 64);

    //Feature Channel 2: Lab@64
    vector<float> sp_Lab;
    int count = 0;
    for(int x = x_min; x <= x_max; x++)
        for(int y = y_min; y <= y_max; y++, count++)
        {
            (float)mat_Lab.at<cv::Vec3b>(x, y).copyTo
        }

    cout<<"Number of points in this superpixel region: "<<i<<endl;
    float* enc_2 = IFVEncode(sp_Lab.t(), 64);

    return NULL;
}

int main(int argc, char **argv) {

//try the vl_slic(im, regionSize, regularizer), where im is a single array
clock_t t1,t2;
t1=clock();
cv::Mat mat = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
Mat mat_Lab = cvCreateMat(mat.rows, mat.cols, CV_8UC3);
cvtColor(mat, mat_Lab, CV_BGR2Lab);
imwrite("LabMat.jpg", mat_Lab);

    // Convert image to one-dimensional array.
    float* image = new float[mat.rows*mat.cols*mat.channels()];
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            // Assuming three channels ...
            image[j + mat.cols*i + mat.cols*mat.rows*0] = mat.at<cv::Vec3b>(i, j)[0];
            image[j + mat.cols*i + mat.cols*mat.rows*1] = mat.at<cv::Vec3b>(i, j)[1];
            image[j + mat.cols*i + mat.cols*mat.rows*2] = mat.at<cv::Vec3b>(i, j)[2];
        }
    }

// The algorithm will store the final segmentation in a one-dimensional array.
    vl_uint32* segmentation = new vl_uint32[mat.rows*mat.cols];
    vl_size height = mat.rows;
    vl_size width = mat.cols;
    vl_size channels = mat.channels();

    // The region size defines the number of superpixels obtained.
    // Regularization describes a trade-off between the color term and the
    // spatial term.
    vl_size region = 80;        
    float regularization = 0.1;
    vl_size minRegion = 200;

    vl_slic_segment(segmentation, image, width, height, channels, region, regularization, minRegion);
    printSegmentation(segmentation, mat, height, width, channels);

//extract features both the SURF@64+COLOR@64 encoded using Improved Fisher Vectors and a GMM with 64 modes
    //Detect the keypoints using SURF Detector
    //first convert the RGB image into Grayscale
    Mat grayMat = cvCreateMat(mat.rows, mat.cols, CV_8UC1);
    cvtColor(mat, grayMat, CV_BGR2GRAY);
    int minHessian = 400;
    //surf(hessionThreshold, nOctaves, nOctaveLayers, extended(0:64, 1:128), upright)
    SURF surf(minHessian, 4, 2, 0, 0);
    Mat descriptors;
    SurfFeatureDetector SurfDetector(minHessian);
    vector<KeyPoint> keypoints;
    surf(grayMat, cv::Mat(), keypoints, descriptors, false);
    cout<<"Size of keypoints "<<keypoints.size()<<endl; 
    cout<<"Descriptors size "<<descriptors.rows<<" "<<descriptors.cols<<endl;

    //for each superpixels, compute two feature channels.
//Step1: find the bounding box for each superpixel region based on segmentation
    //map<label, <x_min, x_max, y_max, y_max>>
    map<int, vector<int>> label_box;
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            int type = (int) segmentation[j + mat.cols*i];
            if(!label_box[type].size()) {
                label_box[type] = vector<int>(4);
                label_box[type][0] = i;
                label_box[type][1] = i;
                label_box[type][2] = j;
                label_box[type][3] = j;
            } else {
                label_box[type][0] = label_box[type][0] > i ? i : label_box[type][0]; 
                label_box[type][1] = label_box[type][1] < i ? i : label_box[type][1]; 
                label_box[type][2] = label_box[type][2] > j ? j : label_box[type][2]; 
                label_box[type][3] = label_box[type][3] < j ? j : label_box[type][3]; 
            }
        }
    }

vector<float*> FV = featureVofSuperpixel(label_box[0][0], label_box[0][0], label_box[0][0], 
label_box[0][0], keypoints, descriptors, mat_Lab);

//encoded the two feature channels using the IFV+GMM with 64 modes
//numClusters = 64, dimensions = 64, so the descriptor need to be transposed
    Mat img_keypoints;
    drawKeypoints( grayMat, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    imwrite("SURF_keypoints.jpg", img_keypoints);


    
    t2=clock();
    cout<<(((float)t2-(float)t1)/CLOCKS_PER_SEC)<<endl;
  return 0;
}
