#pragma once

#include <math.h>
#include <stdlib.h>

#include "matrix.h"

namespace tinyoptimizer
{
template <int N>
struct LUDecomposition
{
    bool singular = false;
    double parity = 1.0;
    Matrix<N, N> LU;
    int permutation[N];

    LUDecomposition(const Matrix<N, N> &mat) : LU(mat)
    {
        permutation[0] = 0;

        for (int i = 1; i < N; ++i)
        {
            permutation[i] = permutation[i - 1] + 1;
        }
    }
};

template <int N>
LUDecomposition<N> lu_decompose(const Matrix<N, N> &mat)
{
    LUDecomposition<N> decomp(mat);

    auto &idx = decomp.permutation;
    auto &LU = decomp.LU;

    // row_scale stores the implicit scaling of each row
    double row_scale[N];

    // Loop over rows to get the implicit scaling information.
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            row_scale[i] = max(fabs(LU(i, j)), row_scale[i]);
        }

        if (row_scale[i] == 0.0)
        {
            decomp.singular = true;
            return decomp;
        }
    }

    // This is the loop over columns of Crout’s method.
    for (int j = 0; j < N; ++j)
    {
        // Calculate beta ij
        for (int i = 0; i < j; ++i)
        {
            for (int k = 0; k < i; ++k)
            {
                LU(i, j) -= LU(i, k) * LU(k, j);
            }
        }

        // Calcuate alpha ij (before division by the pivot)
        for (int i = j; i < N; ++i)
        {
            for (int k = 0; k < j; ++k)
            {
                LU(i, j) -= LU(i, k) * LU(k, j);
            }
        }

        // Search for largest pivot element
        double largest_pivot = 0.0;
        int largest_pivot_idx = j;

        for (int i = j; i < N; i++)
        {
            double this_pivot = fabs(LU(i, j)) / row_scale[i];

            if (this_pivot >= largest_pivot)
            {
                largest_pivot = this_pivot;
                largest_pivot_idx = i;
            }
        }

        if (j != largest_pivot_idx)
        {
            for (int k = 0; k < N; ++k)
            {
                swap(LU(largest_pivot_idx, k), LU(j, k));
            }

            decomp.parity = -decomp.parity;

            swap(idx[j], idx[largest_pivot_idx]);
            row_scale[largest_pivot_idx] = row_scale[j];
        }

        if (j != N)
        {
            for (int i = j + 1; i < N; ++i)
            {
                LU(i, j) /= LU(j, j);
            }
        }
    }

    return decomp;
}

template <int N>
Matrix<N> lu_solve(const LUDecomposition<N> &decomp, const Matrix<N> &b)
{
    Matrix<N> x, tmp;

    auto &idx = decomp.permutation;
    auto &LU = decomp.LU;

    // Forward substitution to solve L * y = b
    for (int i = 0; i < N; ++i)
    {
        double sum = 0.0;

        for (int j = 0; j < i; ++j)
        {
            sum += LU(i, j) * tmp(idx[j]);
        }

        tmp(idx[i]) = b(idx[i]) - sum;
    }

    // Backward substitution to solve U * x = y
    for (int i = N - 1; i >= 0; --i)
    {
        double sum = 0.0;

        for (int j = i + 1; j < N; ++j)
        {
            sum += LU(i, j) * tmp(idx[j]);
        }

        tmp(idx[i]) = (tmp(idx[i]) - sum) / LU(i, i);
    }

    // Undo the permutation
    for (int i = 0; i < N; ++i)
    {
        x(i) = tmp(idx[i]);
    }

    return x;
}

}  // namespace tinyoptimizer