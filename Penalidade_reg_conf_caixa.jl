using Krylov, OptimizationProblems, CUTEst, LinearOperators, NLPModels

function st_cg_caixa(B, g, y, z; ϵ = 1e-15, max_iter = 0, verbose=false)

    n = length(g);
    p = zeros(n);
    r = -g;
    d = r;

    if max_iter == 0
        max_iter = max(50,2n)
    end
    iter = 0

    if norm(r) < ϵ
      println("p1=$p")
        return p
    end

    while iter < max_iter

        iter +=1
        ω = dot(d,B*d);
        R = dot(r,r);
        D = dot(d,d);

        if ω <= 1e-12*D
          p1 = zeros(n)
          p2 = copy(p1)
            P = dot(p,p);
            DP = dot(d,p);
            for i = 1:n
              if p[i] < y[i]
                τ1 = (-DP + sqrt(DP^2 - D*(P - y[i]^2)))/D;
                τ2 = (-DP - sqrt(DP^2 - D*(P - y[i]^2)))/D;
                p1[i] = p[i] + τ1*d[i]
                p2[i] = p[i] + τ2*d[i]
              elseif p[i] > z[i]
                τ1 = (-DP + sqrt(DP^2 - D*(P - y[i]^2)))/D;
                τ2 = (-DP - sqrt(DP^2 - D*(P - y[i]^2)))/D;
                p1[i] = p[i] + τ1*d[i]
                p2[i] = p[i] + τ2*d[i]
              end
            end
            Q1 = 0.5*dot(p1,B*p1) + dot(g,p1)
            Q2 = 0.5*dot(p2,B*p2) + dot(g,p2)
            if Q1 < Q2
              println("p2=$p1")
                return p1
            else
              println("p3=$p2")
                return p2
            end
        end

        α = R/ω;
        p += α*d;
        if 1 > 0
          p -= α*d;
          P = dot(p,p);
          DP = dot(d,p);
          for i = 1:n
            if p[i] < y[i]
                τ = (-DP + sqrt(DP^2 - D*(P - y[i]^2)))/D;
                p[i] = p[i] + τ * d[i]
            elseif p[i] > z[i]
                τ = (-DP + sqrt(DP^2 - D*(P - z[i]^2)))/D;
                p[i] = p[i] + τ * d[i]
            end
          end
          println("p4=$p")
          return p
        end

        r = r - α*B*d;
        if norm(r) < ϵ*norm(g)
          println("p5=$p")
            return p
        end
        M = dot(r,r)/R;
        d = r + M*d;
    end
    println("p6=$p")
    return p
end


function gradient_projection(x, l, u, f, P, M, L_xλ, Δ)
  exit_flag = 0
  n = length(x)
  xp = copy(x)
  y, z = zeros(n),zeros(n)
  t = 1.0
  for i = 1:n
    if -Δ > l[i]
      y[i] = -Δ
    else
      y[i] = l[i]
    end
    if Δ > u[i]
      z[i] = u[i]
    else
      z[i] = Δ
    end
  end
  #println("L_xλ=$L_xλ")
  #println("y=$y")
  #println("z=$z")
  d = st_cg_caixa(M, L_xλ, Δ)
  println("d=$d")
  ti = zeros(n)
  b = 1.0
  for i = 1:n
    if d[i] < 0 && u[i] < Inf
          b = (x[i] - u[i]) / d[i]
    elseif d[i] > 0 && l[i] > -Inf
          b = (x[i] - l[i]) / d[i]
    end
    ti[i] = b
    if b < 1.0
        x[i] = P(x[i] - b * d[i], l[i], u[i])
    else
        x[i] = P(x[i] - d[i], l[i], u[i])
    end
  end
  tj = unique(sort(ti))
  deleteat!(tj, find(tj .== 0))
  l = length(tj)
  p = copy(d)
  for j = 1:l
    for i = 1:n
      if ti[j] < ti[i]
        p[i] = -d[i]
      else
        p[i] = 0
      end
    end
    x_aux = P(x - tj[j] * d,l,u)
    x = x_aux + (t - tj[j]) * p
    f = dot(L_xλ, p) + dot(x_aux, M * p)
    f_2 = dot(p, M * p)
    a = -(f/f_2)
    if f > 0
      t = tj[j]
      x = P(x - t * d,l,u)
      return x, p
    elseif a <= tj[j] && a >= 0
      t = tj[j] + a
      x = P(x - t * d,l,u)
      return x, p
    end
  end

  return x, p
end

# function tamanho_l_u(x, l, u)
#
#   if length(u) < length(l)
#     u_aux = zeros(length(l))
#     for i = 1:length(u)
#       u_aux[i] = u[i]
#     end
#     for i = length(u):length(l)-length(u)
#       u_aux[i] = Inf
#     end
#     u = u_aux
#   elseif length(l) < length(u)
#     l_aux = zeros(length(u))
#     for i = 1:length(l)
#       l_aux[i] = l[i]
#     end
#     for i = length(l):length(u)-length(l)
#       l_aux[i] = -Inf
#     end
#     l = l_aux
#   end
#
#   if length(l) > length(x)
#     x_aux = zeros(length(l))
#     for i = 1:length(x)
#       x_aux[i] = x[i]
#     end
#     for i = length(x)-1:length(l)-length(x)
#       x_aux[i] = 0
#     end
#     x = x_aux
#   end
#
#   if length(l) < length(x)
#     l_aux = zeros(length(x))
#     for i = 1:length(l)
#       l_aux[i] = l[i]
#     end
#     for i = length(l):length(x)-length(l)
#       l_aux[i] = -Inf
#     end
#     l = l_aux
#   end
#
#   if length(u) < length(x)
#     u_aux = zeros(length(x))
#     for i = 1:length(u)
#       u_aux[i] = l[i]
#     end
#     for i = length(u):length(x)-length(u)
#       u_aux[i] = Inf
#     end
#     u = u_aux
#   end
#
#   if length(l) > length(x)
#     x_aux = zeros(length(l))
#     for i = 1:length(x)
#       x_aux[i] = x[i]
#     end
#     for i = length(x)-1:length(l)-length(x)
#       x_aux[i] = 0
#     end
#     x = x_aux
#   end
#   return x, l, u
# end

function penalty_method_caixa(nlp::AbstractNLPModel;tol = 1e-5, max_iter = 1000, max_time = 30, η1 = 1e-2,
                        η2 = 0.66, σ1 = 1.2, σ2 = 0.25, verbose = false)

    start_time = time()
    elapsed_time = 0.0
    exit_flag = 0
    n = nlp.meta.nvar
    x = nlp.meta.x0
    l = nlp.meta.lvar
    u = nlp.meta.uvar
    f(x) = obj(nlp, x)
    c(x) = cons(nlp, x)
    ∇f(x) = grad(nlp, x)
    J(x) = jac(nlp, x)
    H(x,y) = hess_op(nlp, x, y = y)
    P(x,l,u) = min(u, max(x, l))
    x = P(x, l, u)

    µ = 10.0
    ω = 1 / µ
    Δ = 10.0
    r = 0.5
    iter = 0

    g = ∇f(x)
    cx = c(x)
    A = J(x)

    m = length(cx)
    λ = µ * cx

    ϕ(x,λ,µ) = f(x) + dot(λ,c(x)) + (0.5 * µ) * norm(c(x), 2)^2
    ϕx = ϕ(x, λ, µ)
    L_x = g + A' * λ
    L_xλ = g + A' * (µ * cx + λ)
    M = H(x, λ) + µ * A' * A
    ρ = 0
    if verbose
        @printf("%4s  %9s  %9s  %9s  %9s  %9s  %9s  %9s  %9s  %9s\n", "iter","||c(x)||", "λ",
        "µ", "Δ", "ρ", "ω","fx", "||L_x||", "||L_xλ||")
    end

    while norm(P(x - L_x, l, u) - x) > tol || norm(cx) > tol
      cx_km1 = cx
      while norm(P(x - L_xλ,l,u) - x) > ω
        x, d = gradient_projection(x, l, u, f, P, M, L_xλ, Δ)
        println("d=$d")
        ared = ϕx - ϕ(x + d,λ,µ)
        pred = - dot(L_xλ,d) - 0.5 * dot(d, M * d)
        ρ = ared/pred
        if pred <= 0 || ared <= tol
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
            M = H(x,λ) + µ * A' * A
            if ρ > η2
                Δ *= σ1
            end
        end
        if abs(Δ) >= 1e50 || abs(Δ) <= 1e-50
          exit_flag = -2
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
