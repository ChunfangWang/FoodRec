#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <iterator>
#include <vector>

extern "C" {
  #include "vl/generic.h"
  #include "vl/slic.h"
  #include "vl/fisher.h"
  #include "vl/gmm.h"
}

using namespace cv;
using namespace std;

// Reads the images and labels from a given CSV file, a valid file would
// look like this:
//
//      /path/to/person0/image0.jpg;0
//      /path/to/person0/image1.jpg;0
//      /path/to/person1/image0.jpg;1
//      /path/to/person1/image1.jpg;1
//      ...
//
void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels) {
    std::ifstream file(filename.c_str(), ifstream::in);
    if(!file)
        throw std::exception();
    std::string line, path, classlabel;
    // For each line in the given file:
    while (std::getline(file, line)) {
        // Get the current line:
        std::stringstream liness(line);
        // Split it at the semicolon:
        std::getline(liness, path, ';');
        std::getline(liness, classlabel);
        // And push back the data into the result vectors:
        images.push_back(imread(path, IMREAD_GRAYSCALE));
        labels.push_back(atoi(classlabel.c_str()));
    }
}

void read_txt(const string& filename, const string& prefix, map<string, vector<string> >& dict)
{
    std::ifstream file(filename.c_str(), ifstream::in);
    if(!file)
        throw std::exception();
    std::string line, path, classlabel;
    // For each line in the given file:
    while (std::getline(file, line)) {
        // Get the current line:
        std::stringstream liness(line);
        // Split it at the semicolon:
        std::getline(liness, classlabel, '/');
        path =  prefix+line+".jpg";
        dict[classlabel].push_back(path);
    }
}

// Normalizes a given image into a value range between 0 and 255.
Mat norm_0_255(const Mat& src) {
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

// Converts the images given in src into a row matrix.
Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha = 1, double beta = 0) {
    // Number of samples:
    size_t n = src.size();
    // Return empty matrix if no matrices given:
    if(n == 0)
        return Mat();
    // dimensionality of (reshaped) samples
    size_t d = src[0].total();
    // Create resulting data matrix:
    Mat data(n, d, rtype);
    // Now copy data:
    for(int i = 0; i < n; i++) {
        //
        if(src[i].empty()) {
            string error_message = format("Image number %d was empty, please check your input data.", i);
            CV_Error(CV_StsBadArg, error_message);
        }
        // Make sure data can be reshaped, throw a meaningful exception if not!
        if(src[i].total() != d) {
            string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
            CV_Error(CV_StsBadArg, error_message);
        }
        // Get a hold of the current row:
        Mat xi = data.row(i);
        // Make reshape happy by cloning for non-continuous matrices:
        if(src[i].isContinuous()) {
            src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

int main(int argc, const char *argv[]) {

    // Holds some images:
    Mat SURF_images;
    Mat Lab_images;

    string prefix = "/Users/Chunfang/Documents/food-101/images/";

    map<string, vector<string> > dict;
    read_txt("train30.txt", prefix, dict);
    long num_images= 0;
    for(map<string, vector<string> >::iterator it = dict.begin(); it!= dict.end(); ++it)
    {
        for(int i = 0; i < 1; ++i)
        {
            cout<<it->second[i]<<endl;
            Mat im_rgb = imread(it->second[i], CV_LOAD_IMAGE_COLOR);
        /*************************************
            SURF
            */
        Mat im = cvCreateMat(im_rgb.rows, im_rgb.cols, CV_8UC1);
        cvtColor(im_rgb, im, CV_BGR2GRAY);
        int minHessian = 400; float epsilon = 0.1;
        
        //surf(hessionThreshold, nOctaves, nOctaveLayers, extended(0:64, 1:128), upright)
        SURF surf(minHessian, 4, 2, 0, 0);
        Mat descriptors;
        SurfFeatureDetector SurfDetector(minHessian);
        vector<KeyPoint> keypoints;
        surf(im, cv::Mat(), keypoints, descriptors, false);

        //perform PCA on the descriptors extracted on one image
        PCA pca(descriptors, Mat(), CV_PCA_DATA_AS_ROW); //calculate the mean of data
        // And copy the PCA results:
        Mat mean = pca.mean.clone();
        Mat eigenvalues = pca.eigenvalues.clone();
        Mat eigenvectors = pca.eigenvectors.clone();

        Mat descriptors_rot = pca.project( (descriptors - repeat(mean, descriptors.rows, 1)) );
        Mat sqrofeigenvalues;
        // Mat temp2 = eigenvalues.reshape(0,1)+epsilon;
        // std::cout<<eigenvalues.at<float>(5)<<" "<<temp2.at<float>(5)<<std::endl;
        sqrt(eigenvalues.reshape(0,1)+epsilon, sqrofeigenvalues);

        Mat descriptors_rotwhiten;
        divide(descriptors_rot, repeat( sqrofeigenvalues, descriptors_rot.rows, 1), descriptors_rotwhiten);
        SURF_images.push_back(descriptors_rotwhiten);
        // cout<<"SURF_images type "<<SURF_images.type()<<" "SURF_images.data().type()<<endl;
        // SURF_image_kp.push_back(keypoints);

        /***************************************
            Lab color
            */
        Mat im_Lab = cvCreateMat(im_rgb.rows, im_rgb.cols, CV_8UC3);
        Mat Lab_vec = cvCreateMat(im_rgb.rows * im_rgb.cols, 3, CV_32FC1);
        cvtColor(im_rgb, im_Lab, CV_BGR2Lab);
        //reshape the LAB values into a matrix with the rows are [l, a, b] values for each pixel
        im_Lab.reshape(1, im_rgb.rows * im_rgb.cols).convertTo(Lab_vec, CV_32FC1);
        // std::cout<<Lab_vec.at<float>(15, 2)<<" "<<(int)im_Lab.at<Vec3b>(0, 15)[2]<<std::endl;
        PCA pca_lab(Lab_vec, Mat(), CV_PCA_DATA_AS_ROW); 
        Mat mean_lab = pca_lab.mean.clone();
        Mat eigenvalues_lab = pca_lab.eigenvalues.clone();
        Mat eigenvectors_lab = pca_lab.eigenvectors.clone();
        Mat Lab_rot = pca_lab.project( (Lab_vec - repeat(mean_lab, Lab_vec.rows, 1)) );
        Mat sqrofeigenvalues_lab;
        sqrt(eigenvalues_lab.reshape(0,1)+epsilon, sqrofeigenvalues_lab);

        // cout<<"Lab eigenvalues..."<<endl;
        // for(int i = 0; i < sqrofeigenvalues_lab.cols; i++)
        //     std::cout<<sqrofeigenvalues_lab.at<float>(i)<<std::endl;

        Mat Lab_rotwhiten;
        divide(Lab_rot, repeat( sqrofeigenvalues_lab, Lab_rot.rows, 1), Lab_rotwhiten);
        Lab_images.push_back(Lab_rotwhiten);

        num_images++;
        }
    }

    //estimate the ram need to store the SURF and Lab samples (around 20G)
    int numClusters = 64; int dimensions = SURF_images.cols;
    int max_descriptors_per_img = 2048;
    long descriptors_length = SURF_images.cols;
    double num_feats = (double)num_images * max_descriptors_per_img;
    double feature_size_gb = num_feats * descriptors_length * sizeof(float) / 1024. / 1024. / 1024.;
    double posteriory_size_gb = num_feats * numClusters * sizeof(float) / 1024. / 1024. / 1024.;
    double total_gb = feature_size_gb +  posteriory_size_gb;
    std::cout << "will presumably use a bit more than " 
    << total_gb << " GB of memory. (feat="<< feature_size_gb << "gb, posteriors="
    << posteriory_size_gb << "gb)";

    //surf-gmm
    VlGMM* gmm_surf = vl_gmm_new(VL_TYPE_FLOAT, dimensions, numClusters);
    vector<float> SURF_data;
    SURF_images.reshape(0, 1).copyTo(SURF_data);
    vl_gmm_cluster(gmm_surf, (void*)SURF_data.data(), SURF_images.rows);
    float* gmm_surf_mean = (float*)vl_gmm_get_means(gmm_surf);
    float* surf_covariances =  (float*)vl_gmm_get_covariances(gmm_surf);
    float* surf_priors = (float*)vl_gmm_get_priors(gmm_surf);

    //lab-gmm
    VlGMM* gmm_lab = vl_gmm_new(VL_TYPE_FLOAT, dimensions, numClusters);
    vector<float> Lab_data;
    Lab_images.reshape(0, 1).copyTo(Lab_data);
    vl_gmm_cluster(gmm_surf, (void*)Lab_data.data(), Lab_images.rows);
    float* gmm_lab_mean = (float*)vl_gmm_get_means(gmm_surf);
    float* lab_covariances =  (float*)vl_gmm_get_covariances(gmm_surf);
    float* lab_priors = (float*)vl_gmm_get_priors(gmm_surf);

    // Success!
    return 0;
}
