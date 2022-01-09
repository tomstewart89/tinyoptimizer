#include <gtest/gtest.h>

#include "solve.h"

using namespace tiny_sqp_solver;

TEST(Solve, EqualityConstrainedQuadraticProblem)
{
    EqualityConstrainedQuadraticProblem<3, 1> problem;

    Matrix<3> initial_guess = {1, 2, 3};

    problem.Q = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    problem.c = {1, 2, 3};
    problem.A = {2, 0, 0};
    problem.Ax_b = {1};

    auto x_star = initial_guess + solve(problem);

    EXPECT_NEAR(x_star(0), 0.5, 1e-3);
    EXPECT_NEAR(x_star(1), 0.0, 1e-3);
    EXPECT_NEAR(x_star(2), 0.0, 1e-3);
}

TEST(Solve, Unconstrained)
{
    EqualityConstrainedProblem<3, 0> problem;

    problem.objective = [](const Matrix<3>& x) { return x.transpose() * x; };
    problem.constraints = [](const Matrix<3>& x) { return Matrix<0>(); };

    Matrix<3> initial_guess = {50.0, -12.5, 0.5};

    auto x_star = solve(problem, initial_guess);

    EXPECT_NEAR(x_star(0), 0.0, 1e-3);
    EXPECT_NEAR(x_star(1), 0.0, 1e-3);
    EXPECT_NEAR(x_star(2), 0.0, 1e-3);
}

TEST(Solve, KhanAcademyLagrangeMultiplierExample)
{
    EqualityConstrainedProblem<2, 1> problem;

    problem.objective = [](const Matrix<2>& x)
    {
        Matrix<1> cost;
        cost(0) = x(0) * x(0) * x(1);
        return cost;
    };

    problem.constraints = [](const Matrix<2>& x)
    {
        Matrix<1> residuals;
        residuals(0) = x(0) * x(0) + x(1) * x(1) - 1;
        return residuals;
    };

    Matrix<2> initial_guess = {1.0, 1.0};

    auto x_star = solve(problem, initial_guess);

    EXPECT_NEAR(x_star(0), std::sqrt(2.0 / 3.0), 1e-3);
    EXPECT_NEAR(x_star(1), std::sqrt(1.0 / 3.0), 1e-3);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
