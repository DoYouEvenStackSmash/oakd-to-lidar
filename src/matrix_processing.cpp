#include "kernels.h"
#include "image_handler.h"
#include "IQR.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py=pybind11;
void normalize(Eigen::MatrixXd &matrix) {
  //matrix = (882.5*7.5)/(matrix.array()+1);
  double minval = matrix.minCoeff();
  double maxval = matrix.maxCoeff();
  // std::cout << minval << ',' << maxval << std::endl;
  matrix = (matrix.array() - minval) / (maxval - minval) * -1 + 1;
  matrix =(matrix.array()) * (maxval - minval);

}

int sign(double &a) {
  if (a==0)
    return 0;
  return a < 0 ? -1 : 1;
}

Eigen::MatrixXd numpyToEigen(py::array_t<double, py::array::c_style | py::array::forcecast> array) {
    // Request buffer info
    py::buffer_info buf = array.request();
    double *ptr = (double *) buf.ptr;

    // Create Eigen MatrixXd
    Eigen::MatrixXd matrix(400,640);

    // Copy data from numpy array to Eigen MatrixXd
    for (ssize_t i = 0; i < 400; i++) {
        for (ssize_t j = 0; j < 640; j++) {
            matrix(i, j) = ptr[i * 640 + j];
        }
    }

    return matrix;
}

py::array process(py::array input) {
    // setting up the pipeline
    int n = 5;
    int mean_delay = 9;
    int gauss_delay = 8;
    double sigma = 1;
    int lb = 50;
    int ub = 350;
    int window = 300;
    int cutoff = 120;
    int max_val = 300;
    double SLOPE_THRESHOLD = 0.1;
    int ROLLBACK = 5;
    std::vector<double> gaussianKernel = generateGaussianKernel(gauss_delay, sigma);
    std::vector<double> meanKernel = generateMeanKernel(mean_delay);
    std::vector<double> derivativeKernel = generateDerivativeKernel(n);
    std::vector<std::vector<double>> pipeline = {gaussianKernel, meanKernel, derivativeKernel};
    std::vector<std::string> pipeline_names = {"gaussian","mean","derivative"};

    // conversion and preprocessing
    Eigen::MatrixXd imgMatrix = numpyToEigen(input).transpose().rowwise().reverse();
    Eigen::MatrixXd val = imgMatrix;
    
    // normalization
    normalize(val);
    
    // bluring
    val = conv_optimized(val, pipeline[0]);
    
    // windowing
    Eigen::MatrixXd win_val = val.block(0,lb,val.rows(),window);
    
    // mean filter
    Eigen::MatrixXd nwin_val = win_val.rowwise().reverse();
    nwin_val = conv_optimized(nwin_val, pipeline[1]);
    win_val = nwin_val.rowwise().reverse();
    
    // first derivative
    win_val = conv_optimized(win_val, pipeline[2]);

    // outlier rejection
    outlier_rejection(win_val);

    // Eigen::VectorXd row_means(win_val.rows(),1);
    // row_means = win_val.rowwise().mean();
    
    // obstacle mask
    Eigen::MatrixXd obstacles(val.rows(),window);
    obstacles.setZero();
    obstacles = obstacles.array() + 1;
    int filter_padding = mean_delay + gauss_delay;

    // calculate obstacles using the sign of the first derivative
    for (int i = 0; i < win_val.rows(); ++i) {
      for (int j = filter_padding; j < window; ++j) {
        if ((sign(win_val(i,j)) != sign(win_val(i,j-1)))  || win_val(i,j) < SLOPE_THRESHOLD){
          for (int k = j-ROLLBACK; k < obstacles.cols(); ++k) {
            obstacles(i,k) = 0;
          }
          break;
        }
      }
    }
    
    // propagate the index of the obstacle (necessary?)
    std::vector<int> posns(obstacles.rows(),window);
    Eigen::MatrixXd endval(val.rows(), val.cols());
    endval.setZero();
    for (int i = 0; i < obstacles.rows(); ++i) {
      for (int j =0; j < obstacles.cols(); ++j) {
        if (obstacles(i,j) == 0) {
          posns[i] = j;
          break;
        }
      }
    }

    // create a new numpy array to return result
    py::array_t<double> result = py::array_t<double>({posns.size()});
    auto result_buf = result.request();
    double *result_ptr = (double *) result_buf.ptr;
    
    for (size_t i = 0; i < posns.size(); i++) {
        if (posns[i] < cutoff) 
          result_ptr[i] = posns[i]; // Copy input array to result array
        else
          result_ptr[i] = max_val;
    }
    return result;
}

py::array_t<double> arr_conv(py::array_t<double> array) {
    py::buffer_info buf = array.request();
    double *ptr = (double *) buf.ptr;
    std::vector<double> h = {0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0};
    std::vector<double> x(buf.size,0);
    for (int i = 0; i<x.size(); i++) {
      x[i] = ptr[i];
    }

    std::vector<double> yv(x.size()+h.size() - 1);
    for (int i = 0; i < yv.size(); i++) {
      for (int j = 0; j < h.size(); j++) {
        if (i - j < 0 || i-j >= x.size())
          continue;
        yv[i] = yv[i] + h[j] * x[i-j];
      }
    }
    py::array_t<double> y = py::array_t<double>({x.size()});
    auto y_buf = y.request();
    double *y_ptr = (double *) y_buf.ptr;
    for (int i = h.size(); i < h.size() + x.size()-1; i++)
      y_ptr[i-h.size()] = yv[i];
    return y;
}
// // Binding code
PYBIND11_MODULE(matrix_processing, m) {
    m.def("process", &process, "Process numpy array");
    m.def("arr_conv",&arr_conv,"convolve numpy array with low pass filter");
}