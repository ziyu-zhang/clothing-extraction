//
//  main.cpp
//  Markable
//
//  Created by Ziyu Zhang on 7/18/16.
//  Copyright (c) 2016 Ziyu Zhang. All rights reserved.
//

#include <iostream>
#include <string>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/ml/ml.hpp>
#include <opencv2/opencv.hpp>

// for parallelization
#ifdef WITH_MPI
#include "mpi.h"
#endif

#ifdef WITH_OMP
#include <omp.h>
#endif


using namespace cv;
using namespace std;
using namespace cv::ml;

String face_cascade_filename = "/Users/zzhang/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade(face_cascade_filename);

int DetectFaces(Mat& image, vector<Rect>& faces, bool vis=false) {
    Mat gray_image;
    cvtColor(image, gray_image, CV_BGR2GRAY);
    
    face_cascade.detectMultiScale(gray_image, faces, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));
    
    if (vis) {
        for (size_t i = 0; i < faces.size(); ++i) {
            rectangle(gray_image, faces[i], Scalar(255, 0, 255), 4, 8, 0);
        }
        imshow("Detected Faces", gray_image);
        waitKey(0);
    }
    
    return 0;
}

int GetSkinLikelihood(Mat& image, vector<Rect>& faces, Mat& skin, bool vis=false) {
    Mat hsv, hist;
    float hue_range[] = {0, 179};
    float sat_range[] = {0, 255};
    const float* ranges[] = {hue_range, sat_range};
    int channels[] = {0, 1};
    
    if (!faces.empty()) {
        Rect face = faces[0];
        Mat face_image = image(face);
        
        cvtColor(face_image, hsv, CV_BGR2HSV);
        
        int histSize[] = {15, 16};
        
        calcHist(&hsv, 1, channels, Mat(), hist, 2, histSize, ranges);
        
        normalize(hist, hist, 0, 255, NORM_MINMAX);
        imwrite("/Users/zzhang/clothing-extraction/hist.jpg", hist);
    } else {
        hist = imread("/Users/zzhang/clothing-extraction/hist.jpg");
    }
    
    cvtColor(image, hsv, CV_BGR2HSV);
    
    skin.convertTo(skin, hist.depth());
    calcBackProject(&hsv, 1, channels, hist, skin, ranges);
    
    if (vis) {
        imshow("Skin Likelihood", skin);
        waitKey(0);
    }
    
    return 0;
}


struct Options {
    string image_list;
    string output_folder;
};


int ParseInput(int argc, char** argv, struct Options& options) {
    if (argc < 5) {
        std::cout << "Not enough inputs." << std::endl;
        exit(-1);
    }
    
    for(int k = 1; k < argc; ++k) {
        if(strcmp(argv[k], "--image_list") == 0 && k+1 != argc) {
            options.image_list = argv[++k];
        } else if (strcmp(argv[k], "--output_folder") == 0 && k+1 != argc) {
            options.output_folder = argv[++k];
        }
    }
    
    return 0;
}

int main(int argc, char** argv) {
    char hostname[256];
    hostname[0] = '\0';
#ifdef WITH_MPI
    gethostname(hostname, 255);
    std::cout << hostname << std::endl;
#endif
    
    int ClusterSize = 1;
    int ClusterID = 0;
#ifdef WITH_MPI
    if (!MPI::Is_initialized()) {
        MPI::Init();
    }
    ClusterSize = MPI::COMM_WORLD.Get_size();
    ClusterID = MPI::COMM_WORLD.Get_rank();
#endif
    
    Options options;
    ParseInput(argc, argv, options);
    
    vector<string> image_filenames;
    cout << "Searching for files... ";
    string image_filename;
    ifstream ifs(options.image_list.c_str(), std::ios_base::binary | std::ios_base::in);
    ifs >> image_filename;
    while (image_filename.size() > 0) {
        image_filenames.push_back(image_filename);
        image_filename.clear();
        ifs >> image_filename;
    }
    ifs.close();
    std::cout << image_filenames.size() << " files found." << std::endl;
    
#ifdef WITH_OMP
    omp_set_num_threads(2);
#endif
    
    for (int x = ClusterID; x < int(image_filenames.size()); x+=ClusterSize) {
        vector<string>::const_iterator iter = image_filenames.begin();
        for (int i = 0; i < x; ++i) {
            ++iter;
        }
        
#ifdef WITH_MPI
        std::cout << "Processing " << *iter << " by host: " << hostname << std::endl;
#else
        std::cout << "Processing " << *iter << std::endl;
#endif
        
        // read in an image
        image_filename = *iter;
        Mat image = imread(image_filename, CV_LOAD_IMAGE_COLOR);
        
        // train a mixture of gaussians model on the background pixels
        Mat float_image;
        image.convertTo(float_image, CV_32F);
        
        vector<Rect> faces;
        DetectFaces(image, faces);
        
        // create the mask for GrabCut, default all pixels to be hard background
        Mat mask(image.rows, image.cols, CV_8UC1);
        mask.setTo(GC_BGD);
        
        Mat definite_background = mask(Range(0, image.rows), Range(image.cols/6, image.cols - image.cols/6));
        definite_background.setTo(GC_PR_FGD);
        
        // assign the center part of the image to be soft foreground
        //Mat possible_foreground = mask(Range(0, image.rows), Range(image.cols/3, image.cols - image.cols/3));
        //possible_foreground.setTo(GC_PR_FGD);
        
        
        // assign face to be hard background
        if (!faces.empty()) {
            Mat face = mask(faces[0]);
            face.setTo(GC_BGD);
        }
        
        // obtain skin likelihood given face detection
        Mat skin;
        GetSkinLikelihood(image, faces, skin);
        skin = (skin == 255);
        
        // assign skin to be soft foreground
        for(int i=0; i < mask.rows; i++) {
            for(int j=0; j < mask.cols; j++) {
                if (skin.at<uchar>(i, j) > 0) {
                    mask.at<uchar>(i,j) = GC_BGD;
                }
            }
        }
        
        // run GrabCut
        Mat bgdModel,fgdModel;
        grabCut(image, mask, cv::Rect(), bgdModel, fgdModel, 10, GC_INIT_WITH_MASK);
        
        // get foreground pixels
        Mat1b mask_fgd = (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);
        Mat3b out = Mat3b::zeros(image.rows, image.cols);
        image.copyTo(out, mask_fgd);
        
        String image_name = image_filename.substr(image_filename.find_last_of("/")+1);
        String out_filename = options.output_folder + "/" + image_name;
        imwrite(out_filename, out);
        
    }
    
#ifdef WITH_MPI
    std::cout << "Finalizing MPI...";
    if (!MPI::Is_finalized()) {
        MPI::Finalize();
    }
    std::cout << "done" << std::endl;
#endif
    
    return 0;
}
