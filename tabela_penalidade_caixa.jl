using Krylov, OptimizationProblems, CUTEst, LinearOperators, NLPModels
include("Penalidade_reg_conf_caixa.jl")
problems = CUTEst.select(max_var=2,max_con=4,only_equ_con=false)
println("Número de problemas:$(length(problems))")
open("log_penalidade_caixa.txt", "w") do file
    str = @sprintf("%8s  %10s  %10s  %10s  %7s  %7s  %7s  %7s  %7s  %7s  %10s\n","Problema", "f(x)", "||g(x)||", "||c(x)||","iter","ef","nf","ng","nH","nHv","tempo")
    print(str)
    print(file,str)
    for p in problems
      #println(p)
        nlp = CUTEstModel(p)
        c = nlp.counters
        try
            x, fx, ngx, ncx, ef, iter, el_time = penalty_method_caixa(nlp)
            str = @sprintf("%8s  %10.1e  %10.1e  %10.1e  %7d  %7d  %7d  %7d  %7d  %7d  %10.8f\n",p,fx,ngx,ncx,iter,ef,c.neval_obj,c.neval_grad,c.neval_hess,c.neval_hprod,el_time)
            print(str)
            print(file,str)
        finally
            finalize(nlp)
        end
    end
end
