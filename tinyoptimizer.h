#include "finite_difference.h"
#include "lu_decompose.h"

namespace tinyoptimizer
{

template <int X, int P>
struct LinearQuadraticApproximation
{
    Matrix<X, X> Q;
    Matrix<X> c;
    Matrix<X, P> A;
    Matrix<P> Ax_b;
};

template <int DecisionVars, int EqualityConstraints, int InequalityConstraints>
struct Problem
{
    virtual Matrix<1> objective(const Matrix<DecisionVars>& x) const = 0;

    virtual Matrix<EqualityConstraints> equality_constraints(const Matrix<DecisionVars>& x) const
    {
        return Zeros<EqualityConstraints>();
    }
    virtual Matrix<InequalityConstraints> inequality_constraints(const Matrix<DecisionVars>& x) const
    {
        return Zeros<InequalityConstraints>();
    }
};

template <int X, int P>
Matrix<X> solve(const LinearQuadraticApproximation<X, P>& problem)
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

    return lu_solve(decomp, kkt).template view<0, X, 0, 1>();
}

template <int X, int P>
Matrix<X> solve(const Problem<X, P, 0>& problem, const Matrix<X>& initial_guess)
{
    const double step_size = 0.25;
    const double tolerance = 1e-3;
    const int max_iterations = 100;

    Matrix<X> x = initial_guess;
    MemberFunction<X, 1, Problem<X, P, 0>> objective(&Problem<X, P, 0>::objective, problem);
    MemberFunction<X, P, Problem<X, P, 0>> constraints(&Problem<X, P, 0>::equality_constraints, problem);

    for (int i = 0; i < max_iterations; ++i)
    {
        LinearQuadraticApproximation<X, P> local_approximation;
        local_approximation.c = differentiate<X, 1>(objective, x);
        local_approximation.Q = twice_differentiate<X>(objective, x);
        local_approximation.A = differentiate<X, P>(constraints, x);
        local_approximation.Ax_b = constraints(x);

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
class BarrierProblem : public Problem<X, P, 0>
{
   public:
    const Problem<X, P, M>& problem_;
    double t;

    BarrierProblem(const Problem<X, P, M>& problem) : problem_(problem) {}

    Matrix<1> objective(const Matrix<X>& x) const override
    {
        auto cost = problem_.objective(x) * t;

        auto residuals = problem_.inequality_constraints(x);

        for (int i = 0; i < M; ++i)
        {
            cost(0) -= log(-residuals(i));
        }

        return cost;
    }

    Matrix<P> equality_constraints(const Matrix<X>& x) const override { return problem_.equality_constraints(x); }
};

template <int X, int P, int M>
Matrix<X> solve(const Problem<X, P, M>& problem, const Matrix<X>& initial_guess)
{
    const double epsilon = 1e-3;
    const double mu = 1.25;
    const double t0 = 0.25;

    Matrix<X> x = initial_guess;

    BarrierProblem<X, P, M> barrier_problem(problem);

    for (barrier_problem.t = t0; barrier_problem.t < M / epsilon; barrier_problem.t *= mu)
    {
        x = solve(barrier_problem, x);
    }

    return x;
}

template <int X, int P, int M>
class PhaseOneProblem : public Problem<X + 1, P, M>
{
   public:
    const Problem<X, P, M>& problem_;
    double t;

    PhaseOneProblem(const Problem<X, P, M>& problem) : problem_(problem) {}

    Matrix<1> objective(const Matrix<X + 1>& x) const override
    {
        Matrix<1> cost;
        cost(0) = x(X);
        return cost;
    }

    Matrix<P> equality_constraints(const Matrix<X + 1>& x) const override
    {
        return problem_.equality_constraints(x.template view<0, X, 0, 1>());
    }

    Matrix<M> inequality_constraints(const Matrix<X + 1>& x) const override
    {
        return problem_.inequality_constraints(x.template view<0, X, 0, 1>()) - x(X);
    }
};

template <int X, int P, int M>
Matrix<X> solve_strictly_feasible(const Problem<X, P, M>& problem, const Matrix<X>& initial_guess)
{
    Matrix<X + 1> phase_one_initial_guess;

    phase_one_initial_guess.template view<0, X, 0, 1>() = initial_guess;
    phase_one_initial_guess(X) = 1.0;

    auto residuals = problem.inequality_constraints(initial_guess);

    for (int i = 0; i < M; ++i)
    {
        phase_one_initial_guess(X) = max(phase_one_initial_guess(X), residuals(i));
    }

    return solve(PhaseOneProblem<X, P, M>(problem), phase_one_initial_guess).template view<0, X, 0, 1>();
}

}  // namespace tinyoptimizer
