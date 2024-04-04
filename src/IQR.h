#include <iostream>
#include <eigen3/Eigen/Core>
#include <algorithm>
#include <vector>

std::vector<double> calculateQ1Q3(Eigen::VectorXd row);

Eigen::MatrixXd calculateIQR(const Eigen::MatrixXd& matrix);

void outlier_rejection(Eigen::MatrixXd& matrix, double threshold=1.5);
