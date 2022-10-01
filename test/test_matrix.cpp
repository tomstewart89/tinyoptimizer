#include <gtest/gtest.h>

#include "matrix.h"

using namespace tinyoptimizer;

TEST(Matrix, Multiplication)
{
    Matrix<3, 3> A = {3., 5., 8., 4., 7., 9., 2., 5., 10.};
    Matrix<3, 3> B = {6., 3., 2., 3., 7., 5., 8., 9., 1.};

    auto C = A * B;

    EXPECT_DOUBLE_EQ(C(0, 0), 97.);
    EXPECT_DOUBLE_EQ(C(0, 1), 116.);
    EXPECT_DOUBLE_EQ(C(0, 2), 39.);
    EXPECT_DOUBLE_EQ(C(1, 0), 117.);
    EXPECT_DOUBLE_EQ(C(1, 1), 142.);
    EXPECT_DOUBLE_EQ(C(1, 2), 52.);
    EXPECT_DOUBLE_EQ(C(2, 0), 107);
    EXPECT_DOUBLE_EQ(C(2, 1), 131.);
    EXPECT_DOUBLE_EQ(C(2, 2), 39.);
}

TEST(Matrix, ElementwiseMultiplication)
{
    Matrix<3, 3> A = {3., 5., 8., 4., 7., 9., 2., 5., 10.};
    auto B = A * 2.0;

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_DOUBLE_EQ(B(i, j), A(i, j) * 2.0);
        }
    }
}

TEST(Matrix, ElementwiseSubtraction)
{
    Matrix<3, 3> A = {3., 5., 8., 4., 7., 9., 2., 5., 10.};

    auto B = A - 5.0;

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_DOUBLE_EQ(B(i, j), A(i, j) - 5.0);
        }
    }
}

TEST(Matrix, Assignment)
{
    Matrix<3, 3> A = {3., 5., 8., 4., 7., 9., 2., 5., 10.};
    Matrix<3, 3> B = A;

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_DOUBLE_EQ(A(i, j), B(i, j));
        }
    }
}

TEST(Matrix, Addition)
{
    Matrix<3, 3> A = {3., 5., 8., 4., 7., 9., 2., 5., 10.};
    Matrix<3, 3> B = {34, 25., 18., 4.45, 7.2, 319., 32., 55., 1120.};

    auto C = A + B;

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_DOUBLE_EQ(C(i, j), A(i, j) + B(i, j));
        }
    }
}

TEST(Matrix, Subtraction)
{
    Matrix<3, 3> A = {3., 5., 8., 4., 7., 9., 2., 5., 10.};
    Matrix<3, 3> B = {34, 25., 18., 4.45, 7.2, 319., 32., 55., 1120.};

    auto C = A - B;

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_DOUBLE_EQ(C(i, j), A(i, j) - B(i, j));
        }
    }
}

TEST(Matrix, OnesZeros)
{
    Matrix<3, 3> A = Ones<3, 3>() + Ones<3, 3>();
    Matrix<3, 3> B = A * Zeros<3, 3>();

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_DOUBLE_EQ(A(i, j), 2.0);
            EXPECT_DOUBLE_EQ(B(i, j), 0.0);
        }
    }
}

TEST(Matrix, Transpose)
{
    Matrix<3, 2> A = {3., 5., 8., 4., 7., 9.};
    auto B = A.transpose();

    EXPECT_EQ(A.rows, B.cols);
    EXPECT_EQ(A.cols, B.rows);

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            EXPECT_DOUBLE_EQ(A(i, j), B(j, i));
        }
    }
}

TEST(Matrix, References)
{
    Matrix<3, 3> A = {3.25, 5.67, 8.67, 4.55, 7.23, 9.00, 2.35, 5.73, 10.56};
    auto A_view = A.template view<1, 3, 1, 3>();

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_FLOAT_EQ(A_view(i, j), A(i + 1, j + 1));
        }
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
