using FactCheck, NLPModels, CUTEst, LinearOperators, Krylov

include("Penalidade_reg_conf_caixa.jl")

context("Método Penalidade\n") do
    nlp = ADNLPModel(x->(x[1] - 1)^2 + 100*(x[2] - x[1]^2)^2, [-1.2; 1.0], c=x->[x[1]+x[2]-1], lcon=[0.5], ucon=[1.6])

    facts("ROSENBROCK com restrição de igualdade") do
      x,fx,gx,cx,ef,num_iter,el_time = penalty_method_caixa(nlp)
      println("num_iter=$num_iter, x=$x")
      finalize(nlp)
      #@fact gx --> roughly(0.0, 1e-5)
      #@fact cx --> roughly(0.0, 1e-5)
      @fact ef --> 0
      @fact num_iter --> less_than(1000)
    end

    nlp = ADNLPModel(x->(x[1] - 1)^2 + 100*(x[2] - x[1]^2)^2, [-1.2; 1.0], c=x->[x[1]+2*x[2]-2.2], lcon=[0.5], ucon=[1.0])

    facts("ROSENBROCK com restrição de igualdade") do
      x,fx,gx,cx,ef,num_iter,el_time = penalty_method_caixa(nlp)
      println("num_iter=$num_iter, x=$x")
      finalize(nlp)
      #@fact gx --> roughly(0.0, 1e-5)
      #@fact cx --> roughly(0.0, 1e-5)
      @fact ef --> 0
      @fact num_iter --> less_than(1000)
    end
end
