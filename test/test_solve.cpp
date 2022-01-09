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

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
