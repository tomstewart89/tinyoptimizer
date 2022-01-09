#include <sstream>

#include "finite_difference.h"
#include "lu_decompose.h"

namespace tiny_sqp_solver
{
template <int X, int P>
struct EqualityConstrainedQuadraticProblem
{
    Matrix<X, X> Q;
    Matrix<X> c;
    Matrix<X, P> A;
    Matrix<P> Ax_b;
};

template <int X, int P>
Matrix<X> solve(const EqualityConstrainedQuadraticProblem<X, P>& problem)
{
    Matrix<X + P, X + P> KKT;
    Matrix<X + P> kkt;

    kkt.template view<0, X, 0, 1>() = problem.c * -1.0;
    kkt.template view<X, X + P, 0, 1>() = problem.Ax_b * -1.0;

    KKT.template view<0, X, 0, X>() = problem.Q;
    KKT.template view<0, X, X, X + P>() = problem.A;
    KKT.template view<X, X + P, 0, X>() = problem.A.transpose();
    KKT.template view<X, X + P, X, X + P>() = Zeros<P, P>();

    auto decomp = lu_decompose(KKT);

    if (decomp.singular)
    {
        std::stringstream strm;
        strm << "KKT matrix was singular: KKT = " << KKT;
        throw std::runtime_error(strm.str());
    }

    return lu_solve(decomp, kkt).template view<0, X, 0, 1>();
}

template <int X, int P>
struct EqualityConstrainedProblem
{
    std::function<Matrix<1>(const Matrix<X>&)> objective;
    std::function<Matrix<P>(const Matrix<X>&)> constraints;
};

template <int X, int P>
Matrix<X> solve(const EqualityConstrainedProblem<X, P>& problem, const Matrix<X>& initial_guess)
{
    const double step_size = 0.25;
    const double tolerance = 1e-3;
    const int max_iterations = 100;

    Matrix<X> x = initial_guess;

    for (int i = 0; i < max_iterations; ++i)
    {
        EqualityConstrainedQuadraticProblem<X, P> local_approximation;
        local_approximation.c = differentiate<X, 1>(problem.objective, x);
        local_approximation.Q = twice_differentiate<X>(problem.objective, x);
        local_approximation.A = differentiate<X, P>(problem.constraints, x);
        local_approximation.Ax_b = problem.constraints(x);

        auto newton_step = solve(local_approximation);

        x = x + newton_step * step_size;

        if (norm(newton_step) < tolerance)
        {
            break;
        }
    }

    return x;
}

}  // namespace tiny_sqp_solver
