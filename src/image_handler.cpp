#include "image_handler.h"

void preprocess_image(cv::Mat &b) {
  cv::cvtColor(b.t(), b, cv::COLOR_BGR2GRAY);
  cv::flip(b, b, 1);
}

Eigen::MatrixXd mat2eig(const cv::Mat &mat) {
  if (mat.empty()){
    std::cerr << "input matrix is empty" << std::endl;
    return Eigen::MatrixXd();
  }
  Eigen::MatrixXd eigenMat(mat.rows, mat.cols);
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      eigenMat(i,j) = static_cast<double>(mat.at<uint8_t>(i,j));
    }
  }
  return eigenMat;
}

cv::Mat eig2mat(const Eigen::MatrixXd &eig) {
  cv::Mat mat(eig.rows(), eig.cols(), CV_8UC1);
  for (int i = 0; i < eig.rows(); ++i) {
    for (int j = 0; j < eig.cols(); ++j) {
      mat.at<uint8_t>(i,j) = static_cast<uint8_t>(eig(i,j));
    }
  }
  return mat;
}


void stat_image(cv::Mat &image) {
  cv::flip(image, image, 1);
  image = image.t();
  int rows = image.rows;
  int cols = image.cols;
  int channels = image.channels();
  std::cout << "rows:" << rows << "\ncols:" << cols <<"\nchannels:" << channels << std::endl;
  cv::imshow("Loaded image",image);
  cv::waitKey(0);
}

cv::Mat load_image(char* filename) {
  cv::Mat image = cv::imread(filename);

  if (image.empty()) {
    std::cout << "file not found" << std:: endl;
  }
  return image;
}
