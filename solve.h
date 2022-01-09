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

template <int X, int P, int M>
struct InequalityConstrainedProblem
{
    std::function<Matrix<1>(const Matrix<X>&)> objective;
    std::function<Matrix<P>(const Matrix<X>&)> equality_constraints;
    std::function<Matrix<M>(const Matrix<X>&)> inequality_constraints;
};

template <int X, int P, int M>
Matrix<X> solve(const InequalityConstrainedProblem<X, P, M>& problem, const Matrix<X>& initial_guess)
{
    const double epsilon = 1e-3;
    const double mu = 1.25;
    const double t0 = 0.25;
    double t = t0;

    Matrix<X> x = initial_guess;

    EqualityConstrainedProblem<X, P> barrier_problem;

    barrier_problem.objective = [&t, &problem](const Matrix<X>& x)
    {
        auto cost = problem.objective(x) * t;

        auto residuals = problem.inequality_constraints(x);

        for (int i = 0; i < M; ++i)
        {
            cost(0) -= std::log(-residuals(i));
        }

        return cost;
    };

    barrier_problem.constraints = problem.equality_constraints;

    for (; t < M / epsilon; t *= mu)
    {
        x = solve(barrier_problem, x);
    }

    return x;
}

}  // namespace tiny_sqp_solver
