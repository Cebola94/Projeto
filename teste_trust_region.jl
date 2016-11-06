function more_sorensen(B, g, Δ)
    n = length(g)
    d,_ = cg(B,-g)
    λlow, λ = 0.0, 1.0
    I = opEye(n)
    if norm(d) <= Δ
        return d, λ
    end

    d,_ = cg(B + λ*I,-g)
    while norm(d) > Δ
        λlow = λ
        λ *= 10.0
        d,_ = cg(B + λ*I,-g)
    end

    λhigh = λ
    λ = (λlow + λhigh)/2
    d,_ = cg(B + λ*I,-g)
    while abs(norm(d) - Δ) > 1e-3
        if norm(d) > Δ
            λlow = λ
        else
            λhigh = λ
        end
        λ = (λlow + λhigh)/2
        if abs(λlow - λhigh) <= 1e-10
            break
        end
        d,_ = cg(B + λ*I,-g)
    end

    return d, λ
end



using Krylov, OptimizationProblems, CUTEst, LinearOperators, NLPModels
function more_sorensen_method(nlp::AbstractNLPModel; maxiter = 10000, maxtime = 30, tol = 1e-5,
                            η1 = 1e-2, η2 = 0.66, σ1 = 1.2, σ2 = 0.25, verbose = false)

    exit_flag = 0
    n = nlp.meta.nvar
    x = nlp.meta.x0
    f(x) = obj(nlp, x)
    ∇f(x) = grad(nlp, x)
    H(x) = hess(nlp, x)
    Δ = 10.0

    iter = 0
    start_time = time()
    elapsed_time = 0.0

    fx = f(x)
    g = ∇f(x)
    B = H(x)

    m(d) = dot(g,d) + 0.5 * dot(d, B * d)

    if verbose
        @printf("%4s  %8s  %8s  %8s  %9s  %12s  %12s\n", "iter","||d||","λ","ρ","Δ","fx","||∇fx||")
    end

    while norm(g) > tol
        d, λ = more_sorensen(B, g, Δ)
        ared = fx - f(x + d)
        pred = fx - m(d)
        ρ = ared/pred
        if ρ < η1
            Δ *= σ1
            d, λ = more_sorensen(B, g, Δ)
            x = x + d
            fx = f(x)
            g = ∇f(x)
            B = H(x)
        elseif ρ >= η1
            x = x + d
            fx = f(x)
            g = ∇f(x)
            B = H(x)
            if ρ > η2
                Δ *= σ2
            end
        end

        iter += 1
        if iter >= maxiter
            exit_flag = 1
            break
        end
        elapsed_time = time() - start_time
        if elapsed_time >= maxtime
            exit_flag = 2
            break
        end

        verbose && @printf("%2d  %10.1e  %10.1e  %10.1e  %10.1e  %9.1e  %10.1e\n",iter,norm(d),λ,ρ,Δ,fx, norm(g))
    end

return x, fx, norm(g), exit_flag, iter, elapsed_time

end


function compara_metodos(met, prob; tol = 1e-5, itermax = 10000, etmax = 30)

    m=length(prob)
    n=length(met)

    fx = zeros(m,n)
    iter = zeros(m,n)
    et = zeros(m,n)

    fxfac = zeros(m,n)
    etfac = zeros(m,n)

    fbest = 0

    Pf = zeros(m,n)

    rtol = 1e-3
    atol = 1e-6
    println("Número de problemas nesse teste = $m")
    @printf("%4s    %8s     %24s\n","Número", "Problema", "Nosso")
    for i=1:m #loop problemas

        nlp = MathProgNLPModel(prob[i]())
        @printf("%4s",i)
        @printf("   %8s", prob[i])
            for j=1:n #loop métodos

                x, fx[i,j], nlx, ef, iter[i,j], et[i,j] = met[j](nlp) #executa método e guarda resultados

                if iter[i,j] < itermax && et[i,j] < etmax #se método convergiu, x é factível
                    fxfac[i,j] = fx[i,j]
                    etfac[i,j] = et[i,j]
                    Pf[i,j] = iter[i,j] + et[i,j]
                else #senão, guarda valores absurdos pra não perder relação de método i e problema j e não atrapalhar valor mínimo
                    fxfac[i,j] = Inf
                    etfac[i,j] = Inf
                end
            end

        fbest = minimum(fxfac[i,:]) #melhor valor de função para o problema i, considerando os pontos factíveis
        for j = 1:n #loop métodos
            if fxfac[i,j] == Inf #se não convergiu, falhou
                Pf[i,j] = -1
            elseif fxfac[i,j] > fbest + rtol*abs(fbest) + atol #se não satisfaz, falhou
                Pf[i,j] = -1
            end
            @printf("    %6d\n", Pf[i, j])
        end

    end

    return Pf #retorna matriz de performance de colunas "problemas" e linhas "métodos"
    #se P[i,j] = -1 então o método j falhou no problema i
end


using OptimizationProblems
problems = [arwhead,bdqrtic,broydn7d,brybnd,chainwoo,chnrosnb_mod,cosine,curly,curly10,curly20,curly30,dixmaane,dixmaanf,dixmaang,dixmaanh,dixmaani,dixmaanj,dixmaank,dixmaanl,dixmaanm,dixmaann,dixmaano,dixmaanp,edensch,eg2,engval1,extrosnb,fletcbv2,fletcbv3_mod,genhumps,genrose,genrose_nash,indef_mod,liarwhd,morebv,ncb20b,nondquar,penalty2,penalty3,rosenbrock,schmvett,scosine,sinquad,sparsine,sparsqur,srosenbr,tointgss,woods]
mtds = [more_sorensen_method];

P = compara_metodos(mtds, problems);


using BenchmarkProfiles, Plots

performance_profile(P, ["Nosso"])
