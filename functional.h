#pragma once

#include "matrix.h"

namespace tiny_sqp_solver
{
template <int Inputs, int Outputs>
struct Function
{
    virtual Matrix<Outputs> operator()(const Matrix<Inputs> &) const = 0;
};

template <int Inputs, int Outputs>
struct NonMemberFunction : public Function<Inputs, Outputs>
{
    using FunctPtr = Matrix<Outputs> (*)(const Matrix<Inputs> &);

    FunctPtr f_;

   public:
    NonMemberFunction(const FunctPtr &f) : f_(f) {}

    Matrix<Outputs> operator()(const Matrix<Inputs> &input) const override { return (f_)(input); }
};

template <int Inputs, int Outputs, class ObjectType>
class MemberFunction : public Function<Inputs, Outputs>
{
    using FunctPtr = Matrix<Outputs> (ObjectType::*)(const Matrix<Inputs> &) const;

    const ObjectType &obj_;
    FunctPtr f_;

   public:
    MemberFunction(FunctPtr f, const ObjectType &obj) : obj_(obj), f_(f) {}

    Matrix<Outputs> operator()(const Matrix<Inputs> &input) const override { return (obj_.*f_)(input); }
};

};  // namespace tiny_sqp_solver