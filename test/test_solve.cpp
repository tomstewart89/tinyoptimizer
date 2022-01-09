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

TEST(Solve, QuadraticProgram)
{
    constexpr int n_constraints = 5;

    InequalityConstrainedProblem<2, 0, n_constraints> problem;

    problem.objective = [](const Matrix<2>& x) -> Matrix<1> {
        return {std::pow(x(0) - 2, 2) + std::pow(x(1) - 2.0, 2)};
    };

    problem.equality_constraints = [](const Matrix<2>& x) { return Matrix<0>(); };

    problem.inequality_constraints = [](const Matrix<2>& x)
    {
        Matrix<n_constraints> residuals;

        for (int i = 0; i < n_constraints; ++i)
        {
            residuals(i) = x(0) * std::cos(2 * M_PI * i / n_constraints / 2.0) +
                           x(1) * std::sin(2 * M_PI * i / n_constraints / 2.0) - 1.0;
        }
        return residuals;
    };

    Matrix<2> initial_guess = {0.0, 0.0};

    auto x_star = solve(problem, initial_guess);

    EXPECT_NEAR(x_star(0), 0.618789, 1e-3);
    EXPECT_NEAR(x_star(1), 0.849413, 1e-3);
}

TEST(Solve, LuksanVlcek1)
{
    constexpr int N = 10;

    InequalityConstrainedProblem<N, N - 2, N * 2> problem;

    problem.objective = [](const Matrix<N>& x)
    {
        Matrix<1> cost = Zeros<1>();

        for (int i = 0; i < N - 1; ++i)
        {
            cost(0) += 100 * std::pow(std::pow(x(i), 2) - x(i + 1), 2) + std::pow(x(i) - 1, 2);
        }

        cost(0) = x(0) * x(3) * (x(0) + x(1) + x(2)) + x(2);
        return cost;
    };

    problem.equality_constraints = [](const Matrix<N>& x)
    {
        Matrix<N - 2> residuals;

        for (int i = 0; i < N - 2; ++i)
        {
            residuals(i) = 3 * std::pow(x(i + 1), 3) + 2 * x(i + 2) - 5 +
                           std::sin(x(i + 1) - x(i + 2)) * std::sin(x(i + 1) + x(i + 2)) + 4 * x(i + 1) -
                           x(i) * std::exp(x(i) - x(i + 1)) - 3;
        }

        return residuals;
    };

    problem.inequality_constraints = [](const Matrix<N>& x)
    {
        Matrix<N * 2> residuals;

        for (int i = 0; i < N; ++i)
        {
            residuals(i * 2) = x(i) - 5;
            residuals(i * 2 + 1) = -x(i) - 5;
        }

        return residuals;
    };

    Matrix<N> initial_guess = Ones<N>();

    auto x_star = solve(problem, initial_guess);

    auto equality_residuals = problem.equality_constraints(x_star);
    auto inequality_residuals = problem.inequality_constraints(x_star);

    EXPECT_LT(problem.objective(x_star)(0), problem.objective(initial_guess)(0));
    EXPECT_LT((equality_residuals.transpose() * equality_residuals)(0), 1e-6);

    for (int i = 0; i < N * 2; ++i)
    {
        EXPECT_LT(inequality_residuals(i), 0.0);
    }
}

TEST(Solve, HockSchittkowsky71)
{
    InequalityConstrainedProblem<4, 1, 9> problem;

    problem.objective = [](const Matrix<4>& x)
    {
        Matrix<1> cost;
        cost(0) = x(0) * x(3) * (x(0) + x(1) + x(2)) + x(2);
        return cost;
    };

    problem.equality_constraints = [](const Matrix<4>& x)
    {
        Matrix<1> residuals;
        residuals(0) = x(0) * x(0) + x(1) * x(1) + x(2) * x(2) + x(3) * x(3) - 40;

        return residuals;
    };

    problem.inequality_constraints = [](const Matrix<4>& x)
    {
        Matrix<9> residuals;

        for (int i = 0; i < 4; ++i)
        {
            residuals(i * 2) = x(i) - 5;
            residuals(i * 2 + 1) = -x(i) + 1;
        }

        residuals(8) = -x(0) * x(1) * x(2) * x(3) + 25;

        return residuals;
    };

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
