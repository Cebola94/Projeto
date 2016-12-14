using Krylov, OptimizationProblems, CUTEst, LinearOperators, NLPModels
include("Penalidade_reg_conf.jl")

function st_cg(B, g, Δ; ϵ = 1e-15, max_iter = 0, verbose=false)

    n = length(g);
    p = zeros(n);
    r = -g;
    d = r;

    if max_iter == 0
        max_iter = max(50,2n)
    end
    iter = 0

    if norm(r) < ϵ
        return p
    end

    while iter < max_iter

        iter +=1
        ω = dot(d,B*d);
        R = dot(r,r);
        D = dot(d,d);

        if ω <= 1e-12*D
            P = dot(p,p);
            DP = dot(d,p);
            τ1 = (-DP + sqrt(DP^2 - D*(P - Δ^2)))/D;
            τ2 = (-DP - sqrt(DP^2 - D*(P - Δ^2)))/D;
            p1 = p + τ1*d
            p2 = p + τ2*d
            Q1 = 0.5*dot(p1,B*p1) + dot(g,p1)
            Q2 = 0.5*dot(p2,B*p2) + dot(g,p2)
            if Q1 < Q2
                return p1
            else
                return p2
            end
        end

        α = R/ω;
        p += α*d;
        if norm(p)>Δ
            p -= α*d;
            P = dot(p,p);
            DP = dot(d,p);
            τ = (-DP + sqrt(DP^2 - D*(P - Δ^2)))/D;
            return p + τ*d
        end
        r = r - α*B*d;
        if norm(r) < ϵ*norm(g)
            return p
        end
        M = dot(r,r)/R;
        d = r + M*d;
    end
    return p
end

function P(x,l,u)
  if length(l) == 0 && length(u) == 0
    return x
  elseif length(l) >= length(x)
    for i=1:length(x)
      x[i] = max(l[i],min(x[i], u[i]))
    end
  elseif length(l) < length(x)
    for i=1:length(l)
      x[i] = max(l[i],min(x[i], u[i]))
    end
  end
  return x
end

function penalty_method_caixa(nlp::AbstractNLPModel;tol = 1e-5, max_iter = 1000, max_time = 30, η1 = 1e-2,
                        η2 = 0.66, σ1 = 1.2, σ2 = 0.25, verbose = false)

    exit_flag = 0
    λ_min = 1e-20
    λ_max = 1e20
    n = nlp.meta.nvar
    x = nlp.meta.x0
    l = nlp.meta.lcon
    u = nlp.meta.ucon
    if length(l) == 0 && length(u) == 0
        x, fx, gx, cx, ef, iter, el_time = penalty_method(nlp = nlp,tol = tol, max_iter = max_iter, max_time = max_time, η1 = η1,
                                                          η2 = η2, σ1 = σ1, σ2 = σ2, verbose = verbose)
    end
    f(x) = obj(nlp, x)
    c(x) = cons(nlp, x)
    ∇f(x) = grad(nlp, x)
    J(x) = jac(nlp, x)

    µ = 10.0
    ω = 1 / µ
    Δ = 10.0
    r = 0.5
    iter = 0
    start_time = time()
    elapsed_time = 0.0

    x = P(x,l,u)
    g = ∇f(x)
    cx = c(x)
    cx_ant = cx
    A = J(x)

    m = length(cx)
    λ = min(max(λ_min,µ * cx),λ_max)

    ϕ(x,λ,µ) = f(x) + dot(λ,c(x)) + (0.5 * µ) * norm(c(x), 2)^2
    ϕx = ϕ(x,λ,µ)
    L_x = g + A' * λ
    L_xλ = g + A' * (µ * cx + λ)
    B = LBFGSOperator(n)
    M = B + µ * A' * A
    Proj_x = P(x - L_xλ,l,u)
    ρ = 0
    if verbose
        @printf("%4s  %9s  %9s  %9s  %9s  %9s  %9s  %9s  %9s\n", "iter","||c(x)||", "λ",
        "µ", "Δ", "ρ", "ω","fx","||Proj_x||")
    end

    while norm(Proj_x - x) > tol #|| norm(cx) > tol
      while norm(Proj_x - x) > ω
        d = st_cg(M, L_xλ, Δ)
        ared = ϕx - ϕ(x + d,λ,µ)
        pred = - dot(L_xλ,d) - 0.5 * dot(d, M * d)
        ρ = ared/pred
        if pred <= 0
          exit_flag = -1
          return x, f(x), norm(L_x), norm(cx), exit_flag, iter, elapsed_time
        end
        if ρ < η1
            Δ *= σ2
        elseif ρ >= η1


            x = P(x + d,l,u)
            if ρ > η2
                Δ *= σ1
            end
        end
        if abs(Δ) >= 1e100 || abs(Δ) <= 1e-100
          exit_flag = -2
          elapsed_time = time() - start_time
          return x, f(x), norm(L_x), norm(cx), exit_flag, iter, elapsed_time
        end
        ϕx = ϕ(x,λ,µ)
        A = J(x)
        cx_ant = cx
        cx = c(x)
        g_ant = g
        g = ∇f(x)
        y = g - g_ant
        L_xλ = g + A' * (µ * cx + λ)
        L_x = g + A' * λ
        M = push!(B, d, y) + µ * A' * A
        Proj_x = P(x - L_xλ,l,u)

        if norm(Proj_x - x) <= tol #&& norm(cx) <= tol
          elapsed_time = time() - start_time
          return x, f(x), norm(L_x), norm(cx), exit_flag, iter, elapsed_time
        end

      end

      λ += min(max(λ_min,µ * cx),λ_max)
      if iter == 1 || norm(cx,Inf) <= norm(cx_ant,Inf)
          ω = ω / µ
      else
          µ = min(100 * µ, 1e30)
          ω = 1 / µ
      end

      iter += 1
      if iter >= max_iter
          exit_flag = 1
          break
      end
      elapsed_time = time() - start_time
      if elapsed_time >= max_time
          exit_flag = 2
          break
      end

      verbose && @printf("%4d  %9.1e  %9.1e  %9.1e  %9.1e  %9.1e  %9.1e  %9.1e  %9.1e\n",
      iter, norm(cx), norm(λ), µ, Δ, ρ, ω, f(x),norm(Proj_x - x))

    end

return x, f(x), norm(L_x), norm(cx), exit_flag, iter, elapsed_time

end
