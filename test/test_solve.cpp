#include <gtest/gtest.h>

#include "solve.h"

using namespace tiny_sqp_solver;

TEST(Solve, LinearQuadraticApproximation)
{
    LinearQuadraticApproximation<3, 1> approximation;

    Matrix<3> initial_guess = {1, 2, 3};

    approximation.Q = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    approximation.c = {1, 2, 3};
    approximation.A = {2, 0, 0};
    approximation.Ax_b = {1};

    auto x_star = initial_guess + solve(approximation);

    EXPECT_NEAR(x_star(0), 0.5, 1e-3);
    EXPECT_NEAR(x_star(1), 0.0, 1e-3);
    EXPECT_NEAR(x_star(2), 0.0, 1e-3);
}

class EqualityConstrainedProblem : public Problem<3, 0, 0>
{
    Matrix<1> objective(const Matrix<3>& x) const override { return x.transpose() * x; }
};

TEST(Solve, Unconstrained)
{
    Matrix<3> initial_guess = {50.0, -12.5, 0.5};

    auto x_star = solve(EqualityConstrainedProblem(), initial_guess);

    EXPECT_NEAR(x_star(0), 0.0, 1e-3);
    EXPECT_NEAR(x_star(1), 0.0, 1e-3);
    EXPECT_NEAR(x_star(2), 0.0, 1e-3);
}

class KhanAcademyLagrangeMultiplierProblem : public Problem<2, 1, 0>
{
    Matrix<1> objective(const Matrix<2>& x) const override
    {
        Matrix<1> cost;
        cost(0) = x(0) * x(0) * x(1);
        return cost;
    }

    Matrix<1> equality_constraints(const Matrix<2>& x) const override
    {
        Matrix<1> residuals;
        residuals(0) = x(0) * x(0) + x(1) * x(1) - 1;
        return residuals;
    }
};

TEST(Solve, KhanAcademyLagrangeMultiplierExample)
{
    Matrix<2> initial_guess = {1.0, 1.0};

    auto x_star = solve(KhanAcademyLagrangeMultiplierProblem(), initial_guess);

    EXPECT_NEAR(x_star(0), std::sqrt(2.0 / 3.0), 1e-3);
    EXPECT_NEAR(x_star(1), std::sqrt(1.0 / 3.0), 1e-3);
}

template <int NumConstraints>
struct QuadraticProgram : public Problem<2, 0, NumConstraints>
{
    Matrix<1> objective(const Matrix<2>& x) const override
    {
        return {(x(0) - 2) * (x(0) - 2) + (x(1) - 2.0) * (x(1) - 2.0)};
    }

    Matrix<NumConstraints> inequality_constraints(const Matrix<2>& x) const override
    {
        Matrix<NumConstraints> residuals;

        for (int i = 0; i < NumConstraints; ++i)
        {
            residuals(i) = x(0) * std::cos(2 * M_PI * i / NumConstraints / 2.0) +
                           x(1) * std::sin(2 * M_PI * i / NumConstraints / 2.0) - 1.0;
        }
        return residuals;
    }
};
TEST(Solve, QP)
{
    Matrix<2> initial_guess = {0.0, 0.0};

    auto x_star = solve(QuadraticProgram<5>(), initial_guess);

    EXPECT_NEAR(x_star(0), 0.618789, 1e-3);
    EXPECT_NEAR(x_star(1), 0.849413, 1e-3);
}

template <int N>
struct LuksanVlcek1Problem : public Problem<N, N - 2, N * 2>
{
    Matrix<1> objective(const Matrix<N>& x) const override
    {
        Matrix<1> cost = Zeros<1>();

        for (int i = 0; i < N - 1; ++i)
        {
            cost(0) += 100 * std::pow(std::pow(x(i), 2) - x(i + 1), 2) + std::pow(x(i) - 1, 2);
        }

        cost(0) = x(0) * x(3) * (x(0) + x(1) + x(2)) + x(2);
        return cost;
    }

    Matrix<N - 2> equality_constraints(const Matrix<N>& x) const override
    {
        Matrix<N - 2> residuals;

        for (int i = 0; i < N - 2; ++i)
        {
            residuals(i) = 3 * std::pow(x(i + 1), 3) + 2 * x(i + 2) - 5 +
                           std::sin(x(i + 1) - x(i + 2)) * std::sin(x(i + 1) + x(i + 2)) + 4 * x(i + 1) -
                           x(i) * std::exp(x(i) - x(i + 1)) - 3;
        }

        return residuals;
    }

    Matrix<N * 2> inequality_constraints(const Matrix<N>& x) const override
    {
        Matrix<N * 2> residuals;

        for (int i = 0; i < N; ++i)
        {
            residuals(i * 2) = x(i) - 5;
            residuals(i * 2 + 1) = -x(i) - 5;
        }

        return residuals;
    }
};

TEST(Solve, LuksanVlcek1)
{
    Matrix<10> initial_guess = Ones<10>();

    LuksanVlcek1Problem<10> problem;

    auto x_star = solve(problem, initial_guess);

    auto equality_residuals = problem.equality_constraints(x_star);
    auto inequality_residuals = problem.inequality_constraints(x_star);

    EXPECT_LT(problem.objective(x_star)(0), problem.objective(initial_guess)(0));
    EXPECT_LT((equality_residuals.transpose() * equality_residuals)(0), 1e-6);

    for (int i = 0; i < 10 * 2; ++i)
    {
        EXPECT_LT(inequality_residuals(i), 0.0);
    }
}

struct HockSchittkowsky71Problem : public Problem<4, 1, 9>
{
    Matrix<1> objective(const Matrix<4>& x) const override
    {
        Matrix<1> cost;
        cost(0) = x(0) * x(3) * (x(0) + x(1) + x(2)) + x(2);
        return cost;
    }

    Matrix<1> equality_constraints(const Matrix<4>& x) const override
    {
        Matrix<1> residuals;
        residuals(0) = x(0) * x(0) + x(1) * x(1) + x(2) * x(2) + x(3) * x(3) - 40;

        return residuals;
    }

    Matrix<9> inequality_constraints(const Matrix<4>& x) const override
    {
        Matrix<9> residuals;

        for (int i = 0; i < 4; ++i)
        {
            residuals(i * 2) = x(i) - 5;
            residuals(i * 2 + 1) = -x(i) + 1;
        }

        residuals(8) = -x(0) * x(1) * x(2) * x(3) + 25;

        return residuals;
    }
};

TEST(Solve, HockSchittkowsky71)
{
    HockSchittkowsky71Problem problem;

    Matrix<4> initial_guess = solve_strictly_feasible(problem, Matrix<4>{1, 5, 5, 1});

    auto x_star = solve(problem, initial_guess);

    EXPECT_NEAR(x_star(0), 1.0, 1e-2);
    EXPECT_NEAR(x_star(1), 4.742, 1e-2);
    EXPECT_NEAR(x_star(2), 3.821, 1e-2);
    EXPECT_NEAR(x_star(3), 1.379, 1e-2);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
