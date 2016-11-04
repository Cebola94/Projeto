#Recebe um vetor contendo os métodos a serem comparados e a um vetor contendo os problemas de teste

function compara_metodos(met, prob; tol = 1e-5, itermax = 1000, etmax = 60)

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

    for i=1:m #loop problemas

        nlp = CUTEstModel(prob[i])
        c = nlp.counters
        try
            for j=1:n #loop métodos

                x, fx[i,j], ncx, nlx, ef, iter[i,j], et[i,j] = met[j](nlp) #executa método e guarda resultados

                if iter[i,j] < itermax && et[i,j] < etmax && ncx <= tol #se método convergiu, x é factível
                    fxfac[i,j] = fx[i,j]
                    etfac[i,j] = et[i,j]
                    Pf[i,j] = c.neval_obj + c.neval_grad + c.neval_cons + c.neval_jcon + c.neval_jgrad + c.neval_jac + c.neval_jprod + c.neval_jtprod + c.neval_hess + c.neval_hprod + c.neval_jhprod
                else #senão, guarda valores absurdos pra não perder relação de método i e problema j e não atrapalhar valor mínimo
                    fxfac[i,j] = Inf
                    etfac[i,j] = Inf
                end

                reset!(nlp)
            end

        finally
            cutest_finalize(nlp)
        end

        fbest = minimum(fxfac[i,:]) #melhor valor de função para o problema i, considerando os pontos factíveis
        for j = 1:n #loop métodos
            if fxfac[i,j] == inf #se não convergiu, falhou
                Pf[i,j] = -1
            elseif fxfac[i,j] > fbest + rtol*abs(fbest) + atol #se não satisfaz, falhou
                Pf[i,j] = -1
            end
        end

    end

    return Pf #retorna matriz de performance de colunas "problemas" e linhas "métodos"
    #se P[i,j] = -1 então o método j falhou no problema i
end
