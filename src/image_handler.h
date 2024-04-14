#include <opencv2/opencv.hpp>
#include <iostream>
#include <eigen3/Eigen/Core>

// transposes, then flips an image
void preprocess_image(cv::Mat &b);

// Converts a cv matrix to an eigen matrix
Eigen::MatrixXd mat2eig(const cv::Mat &mat);

// converts an eigen matrix to a cv matrix
cv::Mat eig2mat(const Eigen::MatrixXd &eig);

// displays some image stats
void stat_image(cv::Mat &image);

// loads an image from file
cv::Mat load_image(char* filename);