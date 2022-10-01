#include <gtest/gtest.h>

#include "finite_difference.h"

using namespace tinyoptimizer;

TEST(FiniteDifference, Univariate)
{
    Matrix<1> p = {1.0};

    NonMemberFunction<1, 1> f([](const Matrix<1>& x) { return Matrix<1>{std::exp(x(0))}; });

    auto f_dot = differentiate<1, 1>(f, p);
    auto f_ddot = twice_differentiate<1>(f, p);

    EXPECT_NEAR(f_dot(0), f(p)(0), 1e-3);
    EXPECT_NEAR(f_ddot(0), f(p)(0), 1e-3);

    NonMemberFunction<1, 1> g([](const Matrix<1>& x) { return Matrix<1>{std::sin(x(0))}; });
    auto g_dot = differentiate<1, 1>(g, p);
    auto g_ddot = twice_differentiate<1>(g, p);

    EXPECT_NEAR(g_dot(0), std::cos(p(0)), 1e-3);
    EXPECT_NEAR(g_ddot(0), -std::sin(p(0)), 1e-3);
}

TEST(FiniteDifference, Multivariate)
{
    Matrix<3> p = {1.0, 2.0, 3.0};

    NonMemberFunction<3, 1> f([](const Matrix<3>& x)
                              { return Matrix<1>{5.0 * x(0) * x(0) + 3.5 * x(1) * x(2) + 2.0 * x(0) - x(2)}; });

    auto f_dot = differentiate<3, 1>(f, p);
    auto f_ddot = twice_differentiate<3>(f, p);

    EXPECT_NEAR(f_dot(0), 12.0, 1e-3);
    EXPECT_NEAR(f_dot(1), 10.5, 1e-3);
    EXPECT_NEAR(f_dot(2), 6.0, 1e-3);

    EXPECT_NEAR(f_ddot(0, 0), 10.0, 1e-3);
    EXPECT_NEAR(f_ddot(1, 2), 3.5, 1e-3);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
