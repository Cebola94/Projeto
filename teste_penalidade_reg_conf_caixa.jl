using FactCheck, NLPModels, CUTEst, LinearOperators, Krylov

include("Penalidade_reg_conf_caixa.jl")

context("Método Penalidade\n") do
  context("Problemas pequenos") do

    facts("HS7") do
        nlp = CUTEstModel("HS7")
        x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp)
        finalize(nlp)
        @fact gx --> roughly(0.0, 1e-5)
        @fact cx --> roughly(0.0, 1e-5)
        @fact ef --> 0
        @fact num_iter --> less_than(1000)
    end

    facts("HS9") do
      nlp = CUTEstModel("HS9")
      x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp)
      finalize(nlp)
      @fact gx --> roughly(0.0, 1e-5)
      @fact cx --> roughly(0.0, 1e-5)
      @fact ef --> 0
      @fact num_iter --> less_than(1000)
    end

    facts("BT1") do
      nlp = CUTEstModel("BT1")
      x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp)
      finalize(nlp)
      @fact gx --> roughly(0.0, 1e-5)
      @fact cx --> roughly(0.0, 1e-5)
      @fact ef --> 0
      @fact num_iter --> less_than(1000)
    end
  end

  context("Problemas médios\n") do
    facts("HS79") do
      nlp = CUTEstModel("HS79")
      x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp)
      finalize(nlp)
      @fact gx --> roughly(0.0, 1e-5)
      @fact cx --> roughly(0.0, 1e-5)
      @fact ef --> 0
      @fact num_iter --> less_than(1000)
    end

    facts("BT8") do
      nlp = CUTEstModel("BT8")
      x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp)
      finalize(nlp)
      @fact gx --> roughly(0.0, 1e-5)
      @fact cx --> roughly(0.0, 1e-5)
      @fact ef --> 0
      @fact num_iter --> less_than(1000)
    end

    facts("HS77") do
      nlp = CUTEstModel("HS77")
      x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp)
      finalize(nlp)
      @fact gx --> roughly(0.0, 1e-5)
      @fact cx --> roughly(0.0, 1e-5)
      @fact ef --> 0
      @fact num_iter --> less_than(1000)
    end

    facts("ORTHREGB") do
      nlp = CUTEstModel("ORTHREGB")
      x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp)
      finalize(nlp)
      @fact gx --> roughly(0.0, 1e-5)
      @fact cx --> roughly(0.0, 1e-5)
      @fact ef --> 0
      @fact num_iter --> less_than(1000)
    end
  end

  context("Problemas que podem falhar\n") do
    facts("MARATOS") do
      nlp = CUTEstModel("MARATOS")
      x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp)
      finalize(nlp)
      @fact gx --> roughly(0.0, 1e-5)
      @fact cx --> roughly(0.0, 1e-5)
      @fact ef --> 0
      @fact num_iter --> less_than(1000)
    end

    nlp = ADNLPModel(x->(x[1] - 1)^2 + 100*(x[2] - x[1]^2)^2, [-1.2; 1.0], c=x->[x[1]+x[2]-2], lcon=[0.0], ucon=[0.0])

    facts("ROSENBROCK com restrição de igualdade") do
      x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp)
      finalize(nlp)
      @fact gx --> roughly(0.0, 1e-5)
      @fact cx --> roughly(0.0, 1e-5)
      @fact ef --> 0
      @fact num_iter --> less_than(1000)
    end

    nlp = ADNLPModel(x->(x[1] - 1)^2 + 100*(x[2] - x[1]^2)^2, [-1.2; 1.0], c=x->[x[1]+x[2]-1.1], lcon=[0.0], ucon=[0.0])

    facts("ROSENBROCK com restrição de igualdade") do
      x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp)
      finalize(nlp)
      @fact gx --> roughly(0.0, 1e-5)
      @fact cx --> roughly(0.0, 1e-5)
      @fact ef --> 0
      @fact num_iter --> less_than(1000)
    end
  end
end
