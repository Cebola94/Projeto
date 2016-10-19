using Krylov, CUTEst, LinearOperators, NLPModels
function penalty_method(nlp::AbstractNLPModel; α = 0.5, tol = 1e-5, max_iter = 1000, max_time = 60, verbose = false)

    exit_flag = 0
    x = nlp.meta.x0
    n = nlp.meta.nvar
    f(x) = obj(nlp, x)
    c(x) = cons(nlp, x)
    ∇f(x) = grad(nlp, x)
    J(x) = jac(nlp, x)
<<<<<<< HEAD

=======
    
>>>>>>> 724d990486943c309c10a53814bf1bd0ebacbe90
    μ = 10.0
    j = 0

    iter = 0
    start_time = time()
    elapsed_time = 0.0

    fx = f(x)
    ∇fx = ∇f(x)
    cx = c(x)
    A = J(x)

    m = length(cx)
    u = ones(m)
    λ = u - (cx / μ)
    L_xλ = ∇fx - A' * λ
    L_x = ∇fx - A' * u
    Φ(x,u,µ) = f(x) - dot(u,c(x)) + (0.5 / μ) * (norm(c(x), 2))^2
    B = InverseLBFGSOperator(n)
    M = hess(nlp, x, y = λ) + (1 / μ) * (A' * A)
    ϵ_j = μ^(j + 1)
    η = μ^(0.1 + 0.9 * j)
<<<<<<< HEAD

    while norm(L_x) > tol || norm(cx) > tol

        while norm(L_xλ) > ϵ_j

            d,stats = cg(M, -L_xλ)
            t = 1.0

            while Φ(x + t * d,u,μ) > Φ(x,u,μ) + α * t * dot(L_xλ,d)
                t = t * 0.9
                if t <= 1e-16
                    break
                end
            end
            if Φ(x + t * d,u,μ) > Φ(x,u,μ) + α * t * dot(L_xλ,d)
                x1, d1 = SOC(Φ, x, d, L_xλ, J, u, c, μ)
                A = J(x1)
                cx = c(x1)
                ∇fx = ∇f(x1)
                λ = u - (cx / μ)
                L_xλ = ∇fx - A' * λ
                if dot(L_xλ,d1) <= ϵ_j
                    t = 1.0
                    x, d = x1, d1
                end
            else
                x = x + t * d
                A = J(x)
                ∇fx = ∇f(x)
                cx = c(x)
                λ = u - (cx / μ)
                L_xλ = ∇fx - A' * λ
            end
            M = push!(B, t * d, ∇f(x) - ∇fx) + (1 / μ) * (A' * A)
        end
        #println("cx=$(norm(cx)),L_x=$(norm(L_x))")
        L_x = ∇fx - A' * u
        if norm(L_xλ) <= ϵ_j
            ϵ_j = ϵ_j*0.1
        else
            ϵ_j = μ^(j + 1)
        end
        η = μ^(0.1 + 0.9 * j)
        if norm(cx) <= η
            u = λ
            j += 1
        else
            μ = max(min(0.1, sqrt(μ))*μ, 1e-25)
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
    return x, f(x), norm(cx), norm(L_x), exit_flag, iter, elapsed_time
=======
    
    while norm(L_x) > tol || norm(cx) > tol
        
        while norm(L_xλ) > ϵ_j

             d,stats = cg(M, -L_xλ)
             L_xλ_d = dot(L_xλ,d)
             t = 1.0
             while Φ(x + t * d,u,μ) > Φ(x,u,μ) + α*t*L_xλ_d
                 t = t*0.9
                 if t <= tol
                    break
                 end
             end
             s = t * d
             x = x + s
             y = ∇f(x) - ∇fx
             A = J(x)
             M = push!(B, s, y) + (1 / μ) * (A' * A)
             fx = f(x)
             ∇fx = ∇f(x)
             cx = c(x)
             λ = u - (cx / μ)
             L_xλ = ∇fx - A' * λ
             L_x = ∇fx - A' * u
         end
         
         ϵ_j = μ^(j + 1)
         η = μ^(0.1 + 0.9 * j)
        
         if norm(cx) <= η
              u = λ
              j += 1
         else
              μ = max(min(0.1, μ^2), 1e-25)
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
    return x, fx, norm(cx), norm(L_x), exit_flag, iter, elapsed_time
>>>>>>> 724d990486943c309c10a53814bf1bd0ebacbe90
end
