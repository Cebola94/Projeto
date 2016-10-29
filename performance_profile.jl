problems = ["HS7","HS27","MARATOS","HS6","HS28","HS48","HS8","HS79","HS42","HS77","BT2"]
mtds = [penalty_method,metodo_abel];
np = length(problems)
nm = length(mtds)
P = zeros(np, nm)

@printf("%8s   %8s   %8s\n", "Problema", "Nosso", "Abel")
for p = 1:np
        nlp = CUTEstModel(problems[p])
        c = nlp.counters
        @printf("%8s", problems[p])
    try
        for i = 1:nm
            x, fx, ngx, optimal, ef, iter, eltime = mtds[i](nlp);
                if norm(ngx) >= 1e-5 || norm(optimal) >= 1e-5
                    P[p, i] = -1
                else
                    P[p, i] = c.neval_obj + c.neval_grad + c.neval_jac
                end
            @printf("  %8d", P[p, i])
        end
    finally
        cutest_finalize(nlp)
    end
    @printf("\n")
end
