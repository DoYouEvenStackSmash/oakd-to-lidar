#include <vector>
#include <eigen3/Eigen/Core>

std::vector<double> generateGaussianKernel(int n, double sigma);

std::vector<double> generateMeanKernel(double n);

std::vector<double> generateDerivativeKernel(int n);

Eigen::MatrixXd conv(Eigen::MatrixXd &x, std::vector<double> &h);
Eigen::MatrixXd conv_optimized(Eigen::MatrixXd &matrix, std::vector<double> &h);