#pragma once

#include <functional>

#include "matrix.h"

namespace tiny_sqp_solver
{
template <int Inputs, int Outputs>
Matrix<Inputs, Outputs> differentiate(const std::function<Matrix<Outputs>(const Matrix<Inputs> &)> f,
                                      const Matrix<Inputs> &x, double epsilon = 1e-5)
{
    Matrix<Inputs, Outputs> D;
    Matrix<Inputs> perturbation = Zeros<Inputs>();

    for (int i = 0; i < Outputs; ++i)
    {
        for (int j = 0; j < Inputs; ++j)
        {
            perturbation(j) = epsilon / 2.0;

            double high = f(x + perturbation)(i);
            double low = f(x - perturbation)(i);

            D(j, i) = (high - low) / epsilon;

            perturbation(j) = 0;
        }
    }

    return D;
}

template <int Inputs>
Matrix<Inputs, Inputs> twice_differentiate(const std::function<Matrix<1>(const Matrix<Inputs> &)> f,
                                           const Matrix<Inputs> &x, double epsilon = 1e-5)
{
    return differentiate<Inputs, Inputs>([&f](const Matrix<Inputs> &x) { return differentiate<Inputs, 1>(f, x); }, x);
}

};  // namespace tiny_sqp_solver