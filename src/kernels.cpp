#include "kernels.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <thread>


std::vector<double> generateGaussianKernel(int n, double sigma) {
  std::vector<double> kernel(n);
  double sum = 0.0;

  for (int i = 0; i < n; ++i) {
    double x = i - (n - 1) / 2;
    kernel[i] = exp(-(x * x) / (2 * sigma * sigma)) / (sqrt(2 * M_PI) * sigma);
    sum += kernel[i];
  }
  
  for (int i = 0; i < n; ++i) {
    kernel[i] /= sum;
  }

  return kernel;
}


std::vector<double> generateMeanKernel(double n) {
  std::vector<double> kernel(n,1.0/n);
  return kernel;
}

std::vector<double> generateDerivativeKernel(int n) {
  std::vector<double> derivative = {-1,1};
  return derivative;
}

Eigen::MatrixXd conv(Eigen::MatrixXd &matrix, std::vector<double> &h) { 
  Eigen::MatrixXd y(matrix.rows(),matrix.cols());
  y.setZero();
  for (int r = 0; r < matrix.rows(); ++r) {
    for (int i = 0; i < matrix.cols(); ++i) {
      for (int j = 0; j < h.size(); ++j) {
        if (i - j < 0)
          continue;
        y(r,i) = y(r,i) + h[j] * matrix(r, i-j);
      }
    }
  }
  return y;
}

/**
 * @nj was here
 * Multithreaded version of `conv` above. 
 * Spawns 1 thread per row simultaniously. 
 * Debatable if the threaded function is even slow enough to warrant its own thread. 
 * @Todo: profile code, thread with batch might be better.
 * @param matrix Eigen::MatrixXd& matrix to use
 * @oaram h std::vector<double>& whatever the fuck h is
 * @return Eigen::MatrixXd the altered (in-place) matrix
*/
Eigen::MatrixXd conv_optimized(Eigen::MatrixXd &matrix, std::vector<double> &h) { 
  Eigen::MatrixXd y(matrix.rows(),matrix.cols());
  y.setZero();
  // lambda function for calculating the conv sum on a unique row.
  auto convSumRow = [&](int r) {
    for (int i = 0; i < matrix.cols(); ++i) {
      for (int j = 0; j < h.size(); ++j) {
        if (i - j < 0)
          continue;
        y(r,i) = y(r,i) + h[j] * matrix(r, i-j);
      }
    }
  };
  
  // Create threads to edit each row
  std::vector<std::thread> threads;
  for (int r = 0; r < matrix.rows(); ++r) {    
    threads.push_back(std::thread(convSumRow, r));
  }

  // Join all threads
  for (auto& thread : threads) {
      thread.join();
  }
  return y;
}
