using Krylov, CUTEst, LinearOperators, NLPModels
function penalty_method(nlp::AbstractNLPModel; α = 0.5, mem = 5, tol = 1e-5, max_iter = 1000, max_time = 60)

    exit_flag = 0
    x0 = nlp.meta.x0
    n = nlp.meta.nvar
    f(x) = obj(nlp, x)
    c(x) = cons(nlp, x)
    ∇f(x) = grad(nlp, x)
    J(x) = jac(nlp, x)
    μ = 0.1
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
    λ = u - (cx / μ)
    L_xλ = ∇fx - A' * λ
    Φ(x,u,μ) = f(x) - dot(u,cx) + (0.5 / μ) * (norm(cx, 2))^2
    B = LBFGSOperator(n)
    M = hess(nlp, x, y = λ) + (1 / μ) * (A' * A)

    while norm(L_xλ) > tol || norm(cx) > tol

        ϵ_j = μ^(j + 1)
        while norm(L_xλ) > ϵ_j

            d,stats = cg(M, -L_xλ)
            ∇fx_dot_d = dot(L_xλ,d)
            t = 1.0
            while Φ(x + t*d,u,μ) > Φ(x,u,μ) + α*t*∇fx_dot_d
                t = t*0.9
                if t >= tol
                    break
                end
            end

            s = t * d
            x = x + s
            y = ∇f(x) - ∇fx
            A = J(x)
            B = push!(B, s, y)
            M = B + (1 / μ) * (A'*A)
            fx = f(x)
            ∇fx = ∇f(x)
            cx = c(x)
            λ = u - (cx / μ)
            L_xλ = ∇fx - A' * λ
            η = μ^(0.1 + 0.9 * j)
            ϵ_j = μ^(j + 1)

            if norm(cx) <= η
                u = λ
            else
                μ = max(min(0.1, sqrt(μ)) * μ,1e-8)
                j+=1
            end

        end
        #println("μ=$μ")

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
