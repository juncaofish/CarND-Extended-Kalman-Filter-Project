#include "kalman_filter.h"

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;
// Please note that the Eigen library does not initialize
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
}

void KalmanFilter::Predict() {
    /**
     *
      * predict the state
    */
    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::UpdateCommon(const VectorXd &y) {
    /**
      * update the state by using Kalman Filter equations
    */
    MatrixXd Ht = H_.transpose();  // laser: 4x2  radar: 4x3
    MatrixXd S = H_ * P_ * Ht + R_; // laser: 2x2 radar: 3x3
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si; // laser: 4x2 radar: 4x3

    //new estimate
    x_ = x_ + (K * y);
    size_t x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}

void KalmanFilter::Update(const VectorXd &z) {
    /**
      * update the state by using Kalman Filter equations
    */
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    UpdateCommon(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    /**
      * update the state by using Extended Kalman Filter equations
    */
    float px = x_(0);
    float py = x_(1);
    float vx = x_(2);
    float vy = x_(3);

    double rho = sqrt(px * px + py * py);
    double phi = atan2(py, px);
    double rho_dot = (px * vx + py * vy) / max(rho, 0.0001);
    VectorXd z_pred(3);
    z_pred << rho, phi, rho_dot;
    VectorXd y = z - z_pred;
    while (y(1) > M_PI) y(1) -= 2 * M_PI;
    while (y(1) < -M_PI) y(1) += 2 * M_PI;

    UpdateCommon(y);

}
