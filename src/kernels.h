#include <vector>
#include <eigen3/Eigen/Core>

// generates a gaussian kernel for blurring
std::vector<double> generateGaussianKernel(int n, double sigma);

// generates a mean kernel
std::vector<double> generateMeanKernel(double n);

// generates a first-difference kernel
std::vector<double> generateDerivativeKernel(int n);

// convolution
Eigen::MatrixXd conv(Eigen::MatrixXd &x, std::vector<double> &h);

// multithreaded convolution
Eigen::MatrixXd conv_optimized(Eigen::MatrixXd &matrix, std::vector<double> &h);