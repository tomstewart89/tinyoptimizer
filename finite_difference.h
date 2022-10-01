#pragma once

#include "functional.h"
#include "matrix.h"

namespace tinyoptimizer
{

template <int Inputs, int Outputs>
Matrix<Inputs, Outputs> differentiate(const Function<Inputs, Outputs> &f, const Matrix<Inputs> &x,
                                      double epsilon = 1e-5)
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
struct DifferentiatedFunction : public Function<Inputs, Inputs>
{
    const Function<Inputs, 1> &f_;

    DifferentiatedFunction(const Function<Inputs, 1> &f) : f_(f) {}

    Matrix<Inputs> operator()(const Matrix<Inputs> &input) const override { return differentiate(f_, input); }
};

template <int Inputs>
Matrix<Inputs, Inputs> twice_differentiate(const Function<Inputs, 1> &f, const Matrix<Inputs> &x, double epsilon = 1e-5)
{
    return differentiate<Inputs, Inputs>(DifferentiatedFunction<Inputs>(f), x);
}

};  // namespace tinyoptimizer
