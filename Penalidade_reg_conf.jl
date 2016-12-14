using Krylov, OptimizationProblems, CUTEst, LinearOperators, NLPModels

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

function penalty_method(nlp::AbstractNLPModel;tol = 1e-5, max_iter = 1000, max_time = 30, η1 = 1e-2,
                        η2 = 0.66, σ1 = 1.2, σ2 = 0.25, verbose = false)

    exit_flag = 0
    n = nlp.meta.nvar
    x = nlp.meta.x0
    f(x) = obj(nlp, x)
    c(x) = cons(nlp, x)
    ∇f(x) = grad(nlp, x)
    J(x) = jac(nlp, x)
    H(x,y) = hess_op(nlp, x, y=y)

    µ = 10.0
    ω = 1 / µ
    Δ = 10.0
    r = 0.5
    iter = 0
    start_time = time()
    elapsed_time = 0.0

    g = ∇f(x)
    cx = c(x)
    A = J(x)

    m = length(cx)
    λ = µ * cx

    ϕ(x,λ,µ) = f(x) + dot(λ,c(x)) + (0.5 * µ) * norm(c(x), 2)^2
    ϕx = ϕ(x,λ,µ)
    L_x = g + A' * λ
    L_xλ = g + A' * (µ * cx + λ)
    B = LBFGSOperator(n)
    #M = B + µ * A' * A
    M = H(x,λ) + µ * A' * A
    ρ = 0
    if verbose
        @printf("%4s  %9s  %9s  %9s  %9s  %9s  %9s  %9s  %9s  %9s\n", "iter","||c(x)||", "λ",
        "µ", "Δ", "ρ", "ω","fx", "||L_x||", "||L_xλ||")
    end

    while norm(L_x) > tol || norm(cx) > tol
      cx_km1 = cx
      while norm(L_xλ) > ω
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
            x = x + d
            ϕx = ϕ(x,λ,µ)
            A = J(x)
            cx = c(x)
            g_ant = g
            g = ∇f(x)
            y = g - g_ant
            L_xλ = g + A' * (µ * cx + λ)
            L_x = g + A' * λ
            #M = push!(B, d, y) + µ * A' * A
            M = H(x,λ) + µ * A' * A
            if ρ > η2
                Δ *= σ1
            end
        end
        if abs(Δ) >= 1e50 || abs(Δ) <= 1e-50
          exit_flag = -2
          println("Δ = $Δ")
          return x, f(x), norm(L_x), norm(cx), exit_flag, iter, elapsed_time
        end

        if norm(L_x) <= tol && norm(cx) <= tol

          elapsed_time = time() - start_time
          return x, f(x), norm(L_x), norm(cx), exit_flag, iter, elapsed_time
        end

      end

      λ += µ * cx
      if norm(cx,Inf) <= norm(cx_km1,Inf)
          ω = max(ω * 0.1, tol)
      else
          µ = min(100 * µ, 1e30)
          ω = max(1 / µ, tol)
      end
      ϕx = ϕ(x,λ,µ)
      L_xλ = g + A' * (µ * cx + λ)
      L_x = g + A' * λ
      #M = B + µ * A' * A
      M = H(x,λ) + µ * A' * A

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

      verbose && @printf("%4d  %9.1e  %9.1e  %9.1e  %9.1e  %9.1e  %9.1e  %9.1e  %9.1e  %9.1e\n",
      iter, norm(cx), norm(λ), µ, Δ, ρ, ω, f(x), norm(L_x), norm(L_xλ))

    end

return x, f(x), norm(L_x), norm(cx), exit_flag, iter, elapsed_time

end
