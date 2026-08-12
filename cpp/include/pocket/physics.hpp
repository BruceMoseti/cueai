// C++ physics core — cloth sliding / rolling step (NumPy-parity helpers)
#pragma once
#include <cmath>
#include <array>

namespace pocket {

constexpr double G = 9.81;

struct BallState {
    double x{0}, y{0};
    double vx{0}, vy{0};
    double wx{0}, wy{0}, wz{0};
    double R{0.028575};
    double m{0.170};
    bool pocketed{false};

    double I() const { return 0.4 * m * R * R; }

    std::array<double, 2> slip() const {
        return {vx + wy * R, vy - wx * R};
    }

    double speed() const { return std::hypot(vx, vy); }
};

inline void integrate_sliding(BallState& b, double mu_s, double mu_sp, double dt) {
    auto u = b.slip();
    double um = std::hypot(u[0], u[1]);
    if (um < 1e-9) return;
    double ux = u[0] / um, uy = u[1] / um;
    double ax = -mu_s * G * ux;
    double ay = -mu_s * G * uy;
    double I = b.I();
    double awx = -b.R * b.m * ay / I;
    double awy =  b.R * b.m * ax / I;
    double awz = 0.0;
    if (std::fabs(b.wz) > 1e-6)
        awz = -((b.wz > 0) ? 1.0 : -1.0) * 2.5 * mu_sp * G / b.R;
    b.vx += ax * dt; b.vy += ay * dt;
    b.wx += awx * dt; b.wy += awy * dt; b.wz += awz * dt;
    b.x += b.vx * dt; b.y += b.vy * dt;
}

inline void integrate_rolling(BallState& b, double mu_r, double mu_sp, double dt) {
    double v = b.speed();
    if (v < 1e-9) { b.vx = b.vy = 0; return; }
    double ax = -mu_r * G * (b.vx / v);
    double ay = -mu_r * G * (b.vy / v);
    b.vx += ax * dt; b.vy += ay * dt;
    b.wx = -b.vy / b.R;
    b.wy =  b.vx / b.R;
    if (std::fabs(b.wz) > 1e-6)
        b.wz -= ((b.wz > 0) ? 1.0 : -1.0) * 2.5 * mu_sp * G / b.R * dt;
    b.x += b.vx * dt; b.y += b.vy * dt;
}

}  // namespace pocket
