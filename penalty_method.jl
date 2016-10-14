using Krylov, CUTEst, LinearOperators, NLPModels
function penalty_method(nlp::AbstractNLPModel; α = 0.5, tol = 1e-6, max_iter = 1000, max_time = 60)

    exit_flag = 0
    x0 = nlp.meta.x0
    n = nlp.meta.nvar
    f(x) = obj(nlp, x)
    c(x) = cons(nlp, x)
    ∇f(x) = grad(nlp, x)
    J(x) = jac(nlp, x)
    
    µ = 10.0
    j = 0

    x = copy(x0) # Cópia de x0
    iter = 0
    start_time = time()
    elapsed_time = 0.0

    fx = f(x)
    ∇fx = ∇f(x)
    cx = c(x)
    A = J(x)

    m = length(cx)
    u = ones(m)
    λ = u - (cx / µ)
    L_xλ = ∇fx - A' * λ
    Φ(x,u,µ) = f(x) - dot(u,cx) + (0.5 / µ) * (norm(cx, 2))^2
    B = LBFGSOperator(n)
    M = hess(nlp, x, y = λ) + (1 / µ) * (A' * A)

    while norm(∇fx - A' * u) > tol || norm(cx) > tol

         ϵ_j = µ^(j + 1)
         while norm(L_xλ) > ϵ_j

             d,stats = cg(M, -L_xλ)
             t = 1.0
             while Φ(x + t * d,u,µ) > Φ(x,u,µ) + α*t*dot(L_xλ,d)
                 t = t*0.9
                 if t <= tol
                    break
                 end
             end
             s = t * d
             x = x + s
             y = ∇f(x) - ∇fx
             A = J(x)
             M = push!(B, s, y) + (1 / µ) * (A' * A)
             fx = f(x)
             ∇fx = ∇f(x)
             cx = c(x)
             λ += u - (cx / µ)
             L_xλ = ∇fx - A' * λ
         end

         η = µ^(0.1 + 0.9 * j)
         if cx = []
              if (∇fx) <= η
                 u = λ
                 j += 1
              else
                 µ = max(min(0.1, sqrt(µ)) * µ, 1e-10)
                 j = 0
              end
         else if norm(cx) <= η
              u = λ
              j += 1
         else
              µ = max(min(0.1, sqrt(µ)) * µ, 1e-10)
              j = 0
         end

         iter = iter + 1
         if iter >= max_iter
             exit_flag = 1
             break
         end
         elapsed_time = time() - start_time
         if elapsed_time >= max_time
             exit_flag = 2
             break
         end
    end
    return x, fx, cx, norm(∇fx), exit_flag, iter, elapsed_time
end 
