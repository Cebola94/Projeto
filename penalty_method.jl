using Krylov, OptimizationProblems, CUTEst, LinearOperators, NLPModels

function SOC(Φ, x, d, L_xλ_d, A, u, c, μ)
    η1,η2 = 0.2,0.9
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
            d1 = -A'* d1
            if Φ(x + d + d1, u, μ) <= Φ_x + 0.5 * t * L_xλ_d
                d = d + d1
                x = x + d
                cond = false
            else
                t *= 0.9
                if t <= 1e-25
                    cond = false
                end
            end
        else
                t *= 0.9
                if t <= 1e-25
                    cond = false
                end
        end
    end
    return x, d
end

<<<<<<< HEAD
function penalty_method(nlp::AbstractNLPModel; α = 0.5, tol = 1e-5, max_iter = 1000, max_time = 60, verbose = false)

    k_max = 30
=======
function penalty_method(nlp::AbstractNLPModel; α = 0.5, tol = 1e-5, max_iter = 50, max_time = 60, verbose = false)
    
    k_max = 10
>>>>>>> 66b3567352c2bf52f129626cb4f5dadb745003e0
    exit_flag = 0
    x = nlp.meta.x0
    f(x) = obj(nlp, x)
    c(x) = cons(nlp, x)
    ∇f(x) = grad(nlp, x)
<<<<<<< HEAD
    J(x) = jac(nlp, x)

=======
    J(x) = jac_op(nlp, x)
    
>>>>>>> 66b3567352c2bf52f129626cb4f5dadb745003e0
    µ = 0.99
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

    ϕ(x,u,µ) = f(x) - dot(u,c(x)) + (0.5 / µ) * (norm(c(x), 2))^2
    B = LBFGSOperator(nlp.meta.nvar)
    M = B + (1 / µ) * (A' * A)
    ϵ_j = µ^(j + 1)*1e5
    η = µ^(0.1 + 0.9 * j)

    if verbose
        @printf("%4s  %9s  %9s  %9s  %9s  %9s  %9s  %9s\n", "iter","u","||c(x)||","µ","η","fx", "||L_x||", "||L_xλ||")
    end
<<<<<<< HEAD

    while norm(L_x) > tol || norm(cx) > tol
        k = 0
        while norm(L_xλ,2) > ϵ_j
            d,_ = cg(M, -L_xλ)
            L_xλ_d = dot(L_xλ, d)
            t = 1.0
=======
   
    while norm(L_x) > tol || norm(cx) > tol
        k = 0
        while norm(L_xλ,2) > ϵ_j        
            d,_ = cg(M, -L_xλ)
            L_xλ_d = dot(L_xλ, d)
            t = 1.0
            if L_xλ_d >= 0
                exit_flag = -1
                return x, f(x), norm(L_x),norm(cx), exit_flag, iter, elapsed_time
            end  
>>>>>>> 66b3567352c2bf52f129626cb4f5dadb745003e0
            x, s = SOC(ϕ, x, d, L_xλ_d, A, u, c, µ)
            A = J(x)
            cx = c(x)
            ∇fx_ant = ∇fx
            ∇fx = ∇f(x)
            y = ∇fx - ∇fx_ant
            λ = u - (cx / µ)
            L_xλ = ∇fx - A' * λ
            L_x = ∇fx - A' * u
            M = push!(B, s, y) + (1 / µ) * (A' * A)
            k += 1
            if k >= k_max
                break
            end
<<<<<<< HEAD

        end

=======
            
        end 
>>>>>>> 66b3567352c2bf52f129626cb4f5dadb745003e0
        if k >= k_max
            exit_flag = -2
            break
        end
<<<<<<< HEAD

=======
>>>>>>> 66b3567352c2bf52f129626cb4f5dadb745003e0
        if η >= 1e2
            µ = max(min(0.1, sqrt(µ)) * µ, 1e-25)
            j = 0
        end
<<<<<<< HEAD
        if norm(cx) <= η
=======
        if norm(cx) <= η            
>>>>>>> 66b3567352c2bf52f129626cb4f5dadb745003e0
            u = λ
            j += 1
        else
            µ = max(min(0.1, sqrt(µ)) * µ, 1e-25)
            j = 0
        end
        η = µ^(0.1 + 0.9 * j)
        ϵ_j = µ^(j + 1)*1e5
<<<<<<< HEAD

=======
        
>>>>>>> 66b3567352c2bf52f129626cb4f5dadb745003e0
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

        verbose && @printf("%4d  %9.1e  %9.1e  %9.1e  %9.1e  %9.1e  %9.1e  %9.1e\n",
        iter, norm(u), norm(cx), µ, η, f(x), norm(L_x), norm(L_xλ))

    end

return x, f(x), norm(L_x),norm(cx), exit_flag, iter, elapsed_time

end
