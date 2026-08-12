#include "pocket/physics.hpp"
#include <iostream>
#include <iomanip>

int main() {
    pocket::BallState b;
    b.x = 0.5; b.y = 0.5;
    b.vx = 2.0; b.vy = 0.5;
    b.wy = -2.0 / b.R;  // backspin-ish slip

    const double dt = 0.001;
    for (int i = 0; i < 2000; ++i) {
        auto u = b.slip();
        double um = std::hypot(u[0], u[1]);
        if (um > 1e-3)
            pocket::integrate_sliding(b, 0.2, 0.044, dt);
        else
            pocket::integrate_rolling(b, 0.01, 0.044, dt);
        if (b.speed() < 1e-4) break;
    }
    std::cout << std::fixed << std::setprecision(4)
              << "end_pos=(" << b.x << "," << b.y << ") "
              << "speed=" << b.speed() << "\n";
    return 0;
}
