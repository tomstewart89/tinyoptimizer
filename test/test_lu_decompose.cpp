#include <gtest/gtest.h>

#include "lu_decompose.h"

using namespace tiny_sqp_solver;

TEST(LUDecompose, Decompose)
{
    Matrix<7, 7> A = {16, 78, 50, 84, 70, 63, 2, 32, 33, 61, 40, 17, 96, 98, 50, 80, 78, 27, 86, 49, 57, 10, 42, 96, 44,
                      87, 60, 67, 16, 59, 53, 8, 64, 97, 41, 90, 56, 22, 48, 32, 12, 4,  45, 78, 43, 11, 7,  8,  12};

    auto A_orig = A;

    auto decomp = lu_decompose(A);

    EXPECT_FALSE(decomp.singular);

    Matrix<7, 7> P, L, U;

    for (int i = 0; i < 7; ++i)
    {
        for (int j = 0; j < 7; ++j)
        {
            P(i, j) = L(i, j) = U(i, j) = 0.0;
        }
    }

    for (int i = 0; i < 7; ++i)
    {
        P(decomp.permutation[i], i) = 1.0;
        L(i, i) = 1.0;

        for (int j = 0; j < 7; ++j)
        {
            if (i <= j)
            {
                U(i, j) = decomp.LU(i, j);
            }

            if (i > j)
            {
                L(i, j) = decomp.LU(i, j);
            }
        }
    }

    auto A_reconstructed = P * L * U;

    for (int i = 0; i < 7; ++i)
    {
        for (int j = 0; j < 7; ++j)
        {
            EXPECT_DOUBLE_EQ(A_reconstructed(i, j), A_orig(i, j));
        }
    }
}

TEST(LUDecompose, Solve)
{
    Matrix<3, 3> A = {2, 5, 8, 0, 8, 6, 6, 7, 5};
    Matrix<3, 1> b = {10, 11, 12};
    Matrix<3, 1> x_expected = {0.41826923, 0.97115385, 0.53846154};

    auto decomp = lu_decompose(A);

    auto x = lu_solve(decomp, b);

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_NEAR(x_expected(i), x(i), 1e-5);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
