using FactCheck, NLPModels, CUTEst, LinearOperators, Krylov

include("penalty_method.jl")

context("Método Penalidade") do
  context("Problemas pequenos") do
    facts("HS6") do
        nlp = CUTEstModel("HS6")
        x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp)
        @fact x --> roughly(ones(nlp.meta.nvar), 1e-3)
        @fact fx --> roughly(0.0, 1e-6)
        @fact gx --> roughly(0.0, 1e-6)
        @fact cx --> roughly(0.0, 1e-6)
        @fact ef --> 0
        @fact num_iter --> less_than(1000)
        @fact el_time --> less_than(1)
        cutest_finalize(nlp)
    end

    facts("HS7") do
        nlp = CUTEstModel("HS7")
        x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp)
        cutest_finalize(nlp)
        @fact x --> roughly([0.0,1.73205], 1e-3)
        @fact fx --> roughly(-1.73205, 1e-6)
        @fact gx --> roughly(0.0, 1e-6)
        @fact cx --> roughly(0.0, 1e-6)
        @fact ef --> 0
        @fact num_iter --> less_than(1000)
        @fact el_time --> less_than(1)
    end

    facts("HS8") do
      nlp = CUTEstModel("HS8")
      x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp)
      cutest_finalize(nlp)
      @fact x --> roughly([4.601,1.95], 1e-3)
      @fact fx --> roughly(-1.0, 1e-6)
      @fact gx --> roughly(0.0, 1e-6)
      @fact cx --> roughly(0.0, 1e-6)
      @fact ef --> 0
      @fact num_iter --> less_than(1000)
      @fact el_time --> less_than(1)
    end

    facts("BT1") do
      nlp = CUTEstModel("BT1")
      x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp)
      cutest_finalize(nlp)
      @fact x --> roughly([1.0;0.0], 1e-3)
      @fact fx --> roughly(-1.0, 1e-6)
      @fact gx --> roughly(0.0, 1e-6)
      @fact cx --> roughly(0.0, 1e-6)
      @fact ef --> 0
      @fact num_iter --> less_than(1000)
      @fact el_time --> less_than(1)
    end
  end

  context("Problemas médios") do
    facts("HS79") do
      nlp = CUTEstModel("HS79")
      x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp)
      cutest_finalize(nlp)
      @fact x --> roughly([1.19,1.36,1.47,1.67,1.63], 1e-3)
      @fact fx --> roughly(0.078, 1e-6)
      @fact gx --> roughly(0.0, 1e-6)
      @fact cx --> roughly(0.0, 1e-6)
      @fact ef --> 0
      @fact num_iter --> less_than(1000)
      @fact el_time --> less_than(1)
    end

    facts("BT8") do
      nlp = CUTEstModel("BT8")
      x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp)
      cutest_finalize(nlp)
      @fact x --> roughly([1.0,0.0,0.0,0.0,0.0], 1e-3)
      @fact fx --> roughly(1.0, 1e-6)
      @fact gx --> roughly(0.0, 1e-6)
      @fact cx --> roughly(0.0, 1e-6)
      @fact ef --> 0
      @fact num_iter --> less_than(1000)
      @fact el_time --> less_than(1)
    end

    facts("HS77") do
      nlp = CUTEstModel("HS77")
      x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp)
      cutest_finalize(nlp)
      @fact x --> roughly([1.16,0.61,1.38,1.50,1.18], 1e-3)
      @fact fx --> roughly(0.24, 1e-6)
      @fact gx --> roughly(0.0, 1e-6)
      @fact cx --> roughly(0.0, 1e-6)
      @fact ef --> 0
      @fact num_iter --> less_than(1000)
      @fact el_time --> less_than(1)
    end

    facts("ORTHREGB") do
      nlp = CUTEstModel("ORTHREGB")
      x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp)
      cutest_finalize(nlp)
      @fact fx --> roughly(0.0, 1e-6)
      @fact gx --> roughly(0.0, 1e-6)
      @fact cx --> roughly(0.0, 1e-6)
      @fact ef --> 0
      @fact num_iter --> less_than(1000)
      @fact el_time --> less_than(1)
    end
  end

  context("Problemas que podem falhar") do
    facts("MARATOS") do
      nlp = CUTEstModel("MARATOS")
      x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp)
      cutest_finalize(nlp)
      @fact x --> roughly([1.0,0.0], 1e-3)
      @fact fx --> roughly(-1.0, 1e-6)
      @fact gx --> roughly(0.0, 1e-6)
      @fact cx --> roughly(0.0, 1e-6)
      @fact ef --> 0
      @fact num_iter --> less_than(1000)
      @fact el_time --> less_than(1)
    end
    x0 = 10*ones(2)
    f(x) = x[1]^2 + x[2]^2
    c(x) = x[1] + x[2] - 1
    nlp = SimpleNLPModel(f, x0, c=c, lcon=[0.0], ucon=[0.0])

    facts("Limite de Iterações") do
      x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp,max_iter = 1)
      @fact ef --> 1
      @fact num_iter --> 1
    end

    facts("Limite de Tempo") do
      x,fx,gx,cx,ef,num_iter,el_time = penalty_method(nlp,max_time = 0)
      @fact ef --> 2
      @fact el_time --> greater_than(0)
    end
  end
end
