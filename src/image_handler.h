#include <opencv2/opencv.hpp>
#include <iostream>
#include <eigen3/Eigen/Core>

void preprocess_image(cv::Mat &b);
Eigen::MatrixXd mat2eig(const cv::Mat &mat);
cv::Mat eig2mat(const Eigen::MatrixXd &eig);
void stat_image(cv::Mat &image);
cv::Mat load_image(char* filename);