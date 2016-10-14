using FactCheck, NLPModels, CUTEst, LinearOperators

include("penalty_method.jl")
include("metodo_filtro.jl")

context("Método Penalidade") do
  context("Testes sem restrições") do
    facts("Rosenbrock") do
        nlp = CUTEstModel("ROSENBR")
        x,fx,cx,gx = penalty_method(nlp)
        cutest_finalize(nlp)
        @fact x --> roughly(ones(2), 1e-3)
        @fact fx --> roughly(0.0, 1e-6)
        @fact gx --> roughly(0.0, 1e-6)
        @fact ef --> 0
        @fact nf --> less_than(1000)
    end

    facts("Brownbs") do
        nlp = CUTEstModel("BROWNBS")
        x,fx,gx,ef,nf = penalty_method(nlp)
        cutest_finalize(nlp)
        @fact x --> roughly([1e6;2*1e-6], 1e-3)
        @fact fx --> roughly(0.0, 1e-6)
        @fact gx --> roughly(0.0, 1e-6)
        @fact ef --> 0
        @fact nf --> less_than(1000)
    end

    facts("Beale") do
      nlp = CUTEstModel("BEALE")
      x,fx,gx,ef,nf = penalty_method(nlp)
      cutest_finalize(nlp)
      @fact x --> roughly([3;0.5], 1e-3)
      @fact fx --> roughly(0.0, 1e-6)
      @fact gx --> roughly(0.0, 1e-6)
      @fact ef --> 0
      @fact nf --> less_than(1000)
    end
  end

  context("Testes com restrições de igualdade") do
    facts("HS009") do
      nlp = CUTEstModel("HS9")
      x,fx,gx,ef,nf = penalty_method(nlp)
      cutest_finalize(nlp)
      #@fact x --> roughly(r, 1e-3)
      @fact fx --> roughly(-0.5, 1e-6)
      @fact gx --> roughly(0.0, 1e-6)
      @fact ef --> 0
      @fact nf --> less_than(1000)
    end

    facts("HS026") do
      nlp = CUTEstModel("HS26")
      x,fx,gx,ef,nf = penalty_method(nlp)
      cutest_finalize(nlp)
      #@fact x --> roughly(r, 1e-3)
      @fact fx --> roughly(0.0, 1e-6)
      @fact gx --> roughly(0.0, 1e-6)
      @fact ef --> 0
      @fact nf --> less_than(1000)
    end

    facts("HS027") do
      nlp = CUTEstModel("HS27")
      x,fx,gx,ef,nf = penalty_method(nlp)
      cutest_finalize(nlp)
      #@fact x --> roughly(r, 1e-3)
      @fact fx --> roughly(0.04, 1e-6)
      @fact gx --> roughly(0.0, 1e-6)
      @fact ef --> 0
      @fact nf --> less_than(1000)
    end
  end

  context("Testes com falhas") do
    facts("HS027") do
      nlp = CUTEstModel("HS27")
      x,fx,gx,ef,nf = penalty_method(nlp)
      cutest_finalize(nlp)
      @fact x --> roughly(r, 1e-3)
      @fact fx --> roughly(1.0, 1e-6)
      @fact gx --> roughly(0.0, 1e-6)
      @fact ef --> 0
      @fact nf --> less_than(1000)
    end
  end
end

context("Método Filtro") do
  context("Testes sem restrições") do
    facts("Rosenbrock") do
        nlp = CUTEstModel("ROSENBR")
        x,fx,cx,gx = newton_method_filter(nlp)
        cutest_finalize(nlp)
        @fact x --> roughly(ones(2), 1e-3)
        @fact fx --> roughly(0.0, 1e-6)
        @fact gx --> roughly(0.0, 1e-6)
        @fact ef --> 0
        @fact nf --> less_than(1000)
    end

    facts("Brownbs") do
        nlp = CUTEstModel("BROWNBS")
        x,fx,gx,ef,nf = newton_method_filter(nlp)
        cutest_finalize(nlp)
        @fact x --> roughly([1e6;2*1e-6], 1e-3)
        @fact fx --> roughly(0.0, 1e-6)
        @fact gx --> roughly(0.0, 1e-6)
        @fact ef --> 0
        @fact nf --> less_than(1000)
    end

    facts("Beale") do
      nlp = CUTEstModel("BEALE")
      x,fx,gx,ef,nf = newton_method_filter(nlp)
      cutest_finalize(nlp)
      @fact x --> roughly([3;0.5], 1e-3)
      @fact fx --> roughly(0.0, 1e-6)
      @fact gx --> roughly(0.0, 1e-6)
      @fact ef --> 0
      @fact nf --> less_than(1000)
    end
  end

  context("Testes com restrições de igualdade") do
    facts("HS009") do
      nlp = CUTEstModel("HS9")
      x,fx,gx,ef,nf = newton_method_filter(nlp)
      cutest_finalize(nlp)
      #@fact x --> roughly(r, 1e-3)
      @fact fx --> roughly(-0.5, 1e-6)
      @fact gx --> roughly(0.0, 1e-6)
      @fact ef --> 0
      @fact nf --> less_than(1000)
    end

    facts("HS026") do
      nlp = CUTEstModel("HS26")
      x,fx,gx,ef,nf = newton_method_filter(nlp)
      cutest_finalize(nlp)
      #@fact x --> roughly(r, 1e-3)
      @fact fx --> roughly(0.0, 1e-6)
      @fact gx --> roughly(0.0, 1e-6)
      @fact ef --> 0
      @fact nf --> less_than(1000)
    end

    facts("HS027") do
      nlp = CUTEstModel("HS27")
      x,fx,gx,ef,nf = newton_method_filter(nlp)
      cutest_finalize(nlp)
      #@fact x --> roughly(r, 1e-3)
      @fact fx --> roughly(0.04, 1e-6)
      @fact gx --> roughly(0.0, 1e-6)
      @fact ef --> 0
      @fact nf --> less_than(1000)
    end
  end

  context("Testes com falhas") do
    facts("HS027") do
      nlp = CUTEstModel("HS27")
      x,fx,gx,ef,nf = newton_method_filter(nlp)
      cutest_finalize(nlp)
      @fact x --> roughly(r, 1e-3)
      @fact fx --> roughly(1.0, 1e-6)
      @fact gx --> roughly(0.0, 1e-6)
      @fact ef --> 0
      @fact nf --> less_than(1000)
    end
  end
end
