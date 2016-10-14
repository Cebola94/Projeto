function atualiza_filtro(F, x, f, h)
    n = length(F)
    remover = []
    for i = 1:n
        # Se x é dominado, não o adicione
        if f(F[i]) <= f(x) && h(F[i]) <= h(x)
            return false # false pra indicar que x não foi adicionado
        elseif f(x) <= f(F[i]) && h(x) <= h(F[i])
            # Por outro lado, se x domina alguém, remova este outro ponto.
            push!(remover, i)
        end
    end
    deleteat!(F, remover)
    push!(F, x)
    return true
end


function newton_method_filter(nlp::AbstractNLPModel; tol = 1e-5, max_iter = 1000, max_time = 60)
    exit_flag = 0
    x0 = nlp.meta.x0
    n = nlp.meta.nvar
    f(x) = obj(nlp, x)
    c(x) = cons(nlp, x)
    ∇f(x) = grad(nlp, x)
    J(x) = jac(nlp, x)
    H(x) = hess(nlp,x)
    Filtro = Any[x0]

    x = copy(x0) # Cópia de x0
    iter = 0
    start_time = time()
    elapsed_time = 0.0
    fx = f(x)
    ∇fx = ∇f(x)
    B = H(x)
    cx = c(x)
    Jx = J(x)
    h(x) = norm(c(x), 1)

    n = length(x0)
    m = length(cx)
    λ = zeros(m)

    ∇Lx = ∇fx + Jx'*λ

    Φ(x,μ) = f(x) + μ*norm(c(x), 1)
    while norm(∇Lx) > tol || norm(c(x), 1) > tol
        dtil = -[B Jx'; Jx zeros(m,m)]\[∇Lx; cx]
        d = dtil[1:n]
        dλ = dtil[n+1:n+m]
        Jxd = Jx*d

        t = 1.0
        while atualiza_filtro(Filtro, x + t*d, f, h)
            t = t*0.9
        end
        x = x + t*d

        fx = f(x)
        ∇fx = ∇f(x)
        B = H(x)
        cx = c(x)
        Jx = J(x)
        λ += t*dλ
        ∇Lx = ∇fx + Jx'*λ
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
    return x, fx, norm(∇Lx), cx, exit_flag, iter, elapsed_time # Precisamos retornar o ponto encontrado
end
