using Krylov, CUTEst, LinearOperators, NLPModels

function SOC(Φ, x, d, L_xλ, A, u, c, μ)
    η1,η2 = 1e-10,0.9
    t = 1.0
    Φ_x = Φ(x, u, μ)
    cond = true
    while cond == true
        if Φ(x + t * d, u ,μ) <= Φ_x + 0.5 * t * L_xλ_d
            d = t * d
            x = x + d
            cond = false
        elseif t == 1
            d1,_ = cg((A * A'),c(x + d))
            if Φ(x + d + d1, u, μ) <= Φ_x + 0.5 * t * L_xλ_d
                d = d + d1
                x = x + d
                cond = false
            else
                t = max(η1 * t,η2 * t)
            end
        else
                t = max(η1 * t,η2 * t)
        end
    end
    return x
end

function penalty_method(nlp::AbstractNLPModel; α = 0.5, tol = 1e-5, max_iter = 1000, max_time = 30, verbose = false)
    
    exit_flag = 0
    x = nlp.meta.x0
    f(x) = obj(nlp, x)
    c(x) = cons(nlp, x)
    ∇f(x) = grad(nlp, x)
    J(x) = jac_op(nlp, x)
    
    µ = 10.0
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
    λ = u - (cx / µ)
    L_xλ = ∇fx - A' * λ
    L_x = ∇fx - A' * u

    ϕ(x,u,µ) = f(x) - dot(u,cx) + (0.5 / µ) * (norm(cx, 2))^2
    B = LBFGSOperator(nlp.meta.nvar)
    M = B + (1 / µ) * (A' * A)
    ϵ_j = µ^(j + 1)
    η = µ^(0.1 + 0.9 * j)

    if verbose
        @printf("%4s  %9s  %9s  %9s\n", "λ", "µ", "iter", "fx")
    end
   
    while norm(L_x) > tol || norm(cx) > tol
 
        while norm(L_xλ) > ϵ_j        
            d,_ = cg(M, -L_xλ)
            L_xλ_d = dot(L_xλ, d)
            if L_xλ_d >= 0
                exit_flag = -1
                return x, f(x), norm(L_x),norm(cx), exit_flag, iter, elapsed_time
            end        
            x = SOC(F, x, d, L_xλ_d, A, u, c, µ)
            A = J(x)
            cx = c(x)
            ∇fx_ant = ∇fx
            ∇fx = ∇f(x)
            y = ∇fx - ∇fx_ant
            λ = u - (cx / µ)
            L_xλ = ∇fx - A' * λ
            L_x = ∇fx - A' * u
            M = push!(B, d, y) + (1 / µ) * (A' * A)
            
            if η >= 1e2
                µ = max(min(0.1, sqrt(µ))*µ, 1e-25)
                j = 0
            end
            if norm(cx) <= η            
                u = λ
                j += 1
            else
                µ = max(min(0.1, sqrt(µ))*µ, 1e-25)
                j = 0
            end
            η = µ^(0.1 + 0.9 * j)
        end

        if norm(L_xλ) <= ϵ_j
            ϵ_j *= 0.1
        else
            ϵ_j = µ^(j + 1) 
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

        verbose && @printf("%4s  %9s  %9s  %9s\n", λ, µ, iter, f(x))

    end

return x, f(x), norm(L_x),norm(cx), exit_flag, iter, elapsed_time

end
