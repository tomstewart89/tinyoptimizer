#include "tinyoptimizer.h"

using namespace tinyoptimizer;

struct QuadraticProgram : public Problem<2, 0, 4>
{
    Matrix<1> objective(const Matrix<2>& x) const override
    {
        return {(x(0) - 2) * (x(0) - 2) + (x(1) - 2.0) * (x(1) - 2.0)};
    }

    Matrix<4> inequality_constraints(const Matrix<2>& x) const override
    {
        Matrix<4> residuals;

        for (int i = 0; i < 4; ++i)
        {
            residuals(i) = x(0) * cos(2 * M_PI * i / 4 / 2.0) + x(1) * sin(2 * M_PI * i / 4 / 2.0) - 1.0;
        }
        return residuals;
    }
};

void setup()
{
    Matrix<2> initial_guess = {0.0, 0.0};

    auto x_star = solve(QuadraticProgram(), initial_guess);

    Serial << x_star;  // should be : 0.618789, 0.849413
}

void loop() {}