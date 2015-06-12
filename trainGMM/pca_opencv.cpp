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
    vector<Mat> SURF_images;
    vector<Mat> Lab_images;

    string prefix = "/Users/Chunfang/Documents/food-101/images/";

    map<string, vector<string> > dict;
    read_txt("train30.txt", prefix, dict);
    for(map<string, vector<string> >::iterator it = dict.begin(); it!= dict.end(); ++it)
    {
        for(int i = 0; i < 1; ++i)
        {
            Mat im_rgb = imread(it->second[i], CV_LOAD_IMAGE_COLOR);
        /*************************************
            SURF
            */
        // Mat im = imread(it->second[i], IMREAD_GRAYSCALE);
        Mat im = cvCreateMat(im_rgb.rows, im_rgb.cols, CV_8UC1);
        cvtColor(im_rgb, im, CV_BGR2GRAY);
        int minHessian = 400;
        
        //surf(hessionThreshold, nOctaves, nOctaveLayers, extended(0:64, 1:128), upright)
        SURF surf(minHessian, 4, 2, 0, 0);
        Mat descriptors;
        SurfFeatureDetector SurfDetector(minHessian);
        vector<KeyPoint> keypoints;
        surf(im, cv::Mat(), keypoints, descriptors, false);

        //perform PCA on the descriptors extracted on one image
        int num_components = 32;
        PCA pca(descriptors, Mat(), CV_PCA_DATA_AS_ROW, num_components); //calculate the mean of data
        // And copy the PCA results:
        Mat mean = pca.mean.clone();
        Mat eigenvalues = pca.eigenvalues.clone();
        Mat eigenvectors = pca.eigenvectors.clone();

        Mat descriptors_rot = pca.project( (descriptors - repeat(mean, descriptors.rows, 1)) );
        Mat sqrofeigenvalues;
        sqrt(eigenvalues.reshape(0,1), sqrofeigenvalues);
        Mat descriptors_rotwhiten;
        divide(descriptors_rot, repeat( sqrofeigenvalues, descriptors_rot.rows, 1), descriptors_rotwhiten);
        SURF_images.push_back(descriptors_rotwhiten);
        // SURF_image_kp.push_back(keypoints);

        /***************************************
            Lab color
            */
        Mat im_Lab = cvCreateMat(im_rgb.rows, im_rgb.cols, CV_8UC1);
        Mat Lab_vec = cvCreateMat(im_rgb.rows * im_rgb.cols, 3, CV_8UC1);
        cvtColor(im_rgb, im_Lab, CV_BGR2Lab);
        //reshape the LAB values into a matrix with the rows are [l, a, b] values for each pixel
        Lab_vec = im_Lab.clone().reshape(1, im_rgb.rows * im_rgb.cols);
        std::cout<<Lab_vec.at<int>(15, 2)<<" "<<im_Lab.at<Vec3b>(0, 15)[2]<<std::endl;
        PCA pca_lab(Lab_vec, Mat(), CV_PCA_DATA_AS_ROW, 2); 
        Mat mean_lab = pca_lab.mean.clone();
        Mat eigenvalues_lab = pca_lab.eigenvalues.clone();
        Mat eigenvectors_lab = pca_lab.eigenvectors.clone();

        Mat temp = repeat(mean_lab, Lab_vec.rows, 1);
        Mat temp1 =  Lab_vec - temp;
        Mat Lab_rot = pca_lab.project( (Lab_vec - repeat(mean_lab, Lab_vec.rows, 1)) );
        Mat sqrofeigenvalues_lab;
        sqrt(eigenvalues_lab.reshape(0,1), sqrofeigenvalues_lab);
        Mat Lab_rotwhiten;
        divide(Lab_rot, repeat( sqrofeigenvalues_lab, Lab_rot.rows, 1), Lab_rotwhiten);
        Lab_images.push_back(Lab_rotwhiten);
        }

    }
    // Build a matrix with all the observations in row:
    Mat SURF_data = asRowMatrix(SURF_images, CV_32FC1);
    Mat LAB_data = asRowMatrix(Lab_images, CV_32FC1);

    // Number of components to keep for the PCA:
    // int num_components = 100;

    // Perform a PCA:
    // PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, 0.99);

    // // And copy the PCA results:
    // Mat mean = pca.mean.clone();
    // Mat eigenvalues = pca.eigenvalues.clone();
    // Mat eigenvectors = pca.eigenvectors.clone();
    

    // db.push_back(eigenvectors); 
    // The following would read the images from a given CSV file
    // instead, which would look like:
    //
    //      /path/to/person0/image0.jpg;0
    //      /path/to/person0/image1.jpg;0
    //      /path/to/person1/image0.jpg;1
    //      /path/to/person1/image1.jpg;1
    //      ...
    //
    // Uncomment this to load from a CSV file:
    //

    /*
    vector<int> labels;
    read_csv("/home/philipp/facerec/data/at.txt", db, labels);
    */


    // Success!
    return 0;
}
