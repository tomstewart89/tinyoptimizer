#pragma once

#include "Arduino.h"

namespace tinyoptimizer
{
template <typename RefType, int Rows, int Cols = 1>
class MatrixRef;

template <typename RefType, int Rows, int Cols = 1>
class MatrixTranspose;

template <typename DerivedType, int Rows, int Cols>
struct MatrixBase
{
   public:
    constexpr static int rows = Rows;
    constexpr static int cols = Cols;

    double &operator()(int i, int j = 0) { return static_cast<DerivedType *>(this)->operator()(i, j); }

    double operator()(int i, int j = 0) const { return static_cast<const DerivedType *>(this)->operator()(i, j); }

    template <typename MatType>
    DerivedType &operator=(const MatrixBase<MatType, Rows, Cols> &mat)
    {
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                static_cast<DerivedType &> (*this)(i, j) = mat(i, j);
            }
        }

        return static_cast<DerivedType &>(*this);
    }

    DerivedType &operator=(double elem)
    {
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                static_cast<DerivedType &> (*this)(i, j) = elem;
            }
        }

        return static_cast<DerivedType &>(*this);
    }

    template <int row_start, int row_end, int col_start, int col_end>
    MatrixRef<DerivedType, row_end - row_start, col_end - col_start> view()
    {
        return MatrixRef<DerivedType, row_end - row_start, col_end - col_start>(static_cast<DerivedType &>(*this),
                                                                                row_start, col_start);
    }

    template <int row_start, int row_end, int col_start, int col_end>
    MatrixRef<const DerivedType, row_end - row_start, col_end - col_start> view() const
    {
        return MatrixRef<const DerivedType, row_end - row_start, col_end - col_start>(
            static_cast<const DerivedType &>(*this), row_start, col_start);
    }

    MatrixTranspose<DerivedType, Cols, Rows> transpose()
    {
        return MatrixTranspose<DerivedType, Cols, Rows>(static_cast<DerivedType &>(*this));
    }

    MatrixTranspose<const DerivedType, Cols, Rows> transpose() const
    {
        return MatrixTranspose<const DerivedType, Cols, Rows>(static_cast<const DerivedType &>(*this));
    }
};

template <int Rows, int Cols = 1>
class Matrix : public MatrixBase<Matrix<Rows, Cols>, Rows, Cols>
{
   public:
    double storage[Rows * Cols];

    double &operator()(int i, int j = 0) { return storage[i * Cols + j]; }
    double operator()(int i, int j = 0) const { return storage[i * Cols + j]; }

    Matrix() = default;

    template <typename DerivedType>
    Matrix(const MatrixBase<DerivedType, Rows, Cols> &mat)
    {
        static_cast<MatrixBase<Matrix<Rows, Cols>, Rows, Cols> &>(*this) = mat;
    }

    template <typename DerivedType>
    Matrix &operator=(const MatrixBase<DerivedType, Rows, Cols> &mat)
    {
        return static_cast<MatrixBase<Matrix<Rows, Cols>, Rows, Cols> &>(*this) = mat;
    }
};

template <int Rows, int Cols = 1>
class Zeros : public MatrixBase<Zeros<Rows, Cols>, Rows, Cols>
{
   public:
    double operator()(int i, int j = 0) const { return 0.0; }

    Zeros() = default;
};

template <int Rows, int Cols = 1>
class Ones : public MatrixBase<Ones<Rows, Cols>, Rows, Cols>
{
   public:
    double operator()(int i, int j = 0) const { return 1.0; }

    Ones() = default;
};

template <typename RefType, int Rows, int Cols>
class MatrixRef : public MatrixBase<MatrixRef<RefType, Rows, Cols>, Rows, Cols>
{
    RefType &parent_;
    const int row_offset_;
    const int col_offset_;

   public:
    explicit MatrixRef(RefType &parent, int row_offset = 0, int col_offset = 0)
        : parent_(parent), row_offset_(row_offset), col_offset_(col_offset)
    {
    }

    double &operator()(int i, int j) { return parent_(i + row_offset_, j + col_offset_); }
    double operator()(int i, int j) const { return parent_(i + row_offset_, j + col_offset_); }

    template <typename MatType>
    MatrixRef &operator=(const MatType &mat)
    {
        return static_cast<MatrixBase<MatrixRef<RefType, Rows, Cols>, Rows, Cols> &>(*this) = mat;
    }
};

template <typename RefType, int Rows, int Cols>
class MatrixTranspose : public MatrixBase<MatrixTranspose<RefType, Rows, Cols>, Rows, Cols>
{
    RefType &parent_;

   public:
    explicit MatrixTranspose(RefType &parent) : parent_(parent) {}

    double &operator()(int i, int j) { return parent_(j, i); }
    double operator()(int i, int j) const { return parent_(j, i); }

    template <typename MatType>
    MatrixTranspose &operator=(const MatType &mat)
    {
        return static_cast<MatrixBase<MatrixTranspose<RefType, Rows, Cols>, Rows, Cols> &>(*this) = mat;
    }
};

template <typename MatAType, typename MatBType, int MatARows, int MatACols, int MatBCols>
Matrix<MatARows, MatBCols> operator*(const MatrixBase<MatAType, MatARows, MatACols> &matA,
                                     const MatrixBase<MatBType, MatACols, MatBCols> &matB)
{
    Matrix<MatARows, MatBCols> ret;

    for (int i = 0; i < MatARows; ++i)
    {
        for (int j = 0; j < MatBCols; ++j)
        {
            if (MatACols > 0)
            {
                ret(i, j) = matA(i, 0) * matB(0, j);
            }

            for (int k = 1; k < MatACols; k++)
            {
                ret(i, j) += matA(i, k) * matB(k, j);
            }
        }
    }
    return ret;
}

template <typename MatAType, typename MatBType, int Rows, int Cols>
Matrix<Rows, Cols> operator+(const MatrixBase<MatAType, Rows, Cols> &matA, const MatrixBase<MatBType, Rows, Cols> &matB)
{
    Matrix<Rows, Cols> ret;

    for (int i = 0; i < Rows; ++i)
    {
        for (int j = 0; j < Cols; ++j)
        {
            ret(i, j) = matA(i, j) + matB(i, j);
        }
    }

    return ret;
}

template <typename MatAType, typename MatBType, int Rows, int Cols>
Matrix<Rows, Cols> operator-(const MatrixBase<MatAType, Rows, Cols> &matA, const MatrixBase<MatBType, Rows, Cols> &matB)
{
    Matrix<Rows, Cols> ret;

    for (int i = 0; i < Rows; ++i)
    {
        for (int j = 0; j < Cols; ++j)
        {
            ret(i, j) = matA(i, j) - matB(i, j);
        }
    }

    return ret;
}

template <typename MatType, int Rows, int Cols>
Matrix<Rows, Cols> operator*(const MatrixBase<MatType, Rows, Cols> &mat, const double k)
{
    Matrix<Rows, Cols> ret;

    for (int i = 0; i < Rows; ++i)
    {
        for (int j = 0; j < Cols; ++j)
        {
            ret(i, j) = mat(i, j) * k;
        }
    }
    return ret;
}

template <typename MatType, int Rows, int Cols>
Matrix<Rows, Cols> operator-(const MatrixBase<MatType, Rows, Cols> &mat, const double k)
{
    Matrix<Rows, Cols> ret;

    for (int i = 0; i < Rows; ++i)
    {
        for (int j = 0; j < Cols; ++j)
        {
            ret(i, j) = mat(i, j) - k;
        }
    }
    return ret;
}

template <typename MatType, int Rows, int Cols>
double norm(const MatrixBase<MatType, Rows, Cols> &mat)
{
    double sum = 0.0;

    for (int i = 0; i < Rows; ++i)
    {
        for (int j = 0; j < Cols; ++j)
        {
            sum += pow(mat(i, j), 2);
        }
    }

    return sqrt(sum);
}

template <typename DerivedType, int Rows, int Cols>
Print &operator<<(Print &strm, const MatrixBase<DerivedType, Rows, Cols> &mat)
{
    strm.print('[');

    for (int i = 0; i < Rows; ++i)
    {
        strm.print('[');

        for (int j = 0; j < Cols; ++j)
        {
            strm.print(mat(i, j));
            strm.print((j == Cols - 1) ? ']' : ',');
        }

        strm.print(i == Rows - 1 ? ']' : ',');
    }
    return strm;
}

}  // namespace tinyoptimizer