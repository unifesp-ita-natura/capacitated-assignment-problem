# Calibração do SA contra o MIP (HiGHS)

**Date:** 2026-09-28
**Related:** branch `fine-tuning`; experimento `experiments/sa_calibration/`

## Task

Refazer a calibração dos parâmetros do Simulated Annealing usando o ótimo do MIP
(HiGHS) como referência de gap, em vez do BKV do próprio SA usado no relatório anterior,
e podar as cópias que a sessão anterior deixou em `src/fine-tuning/` (reconstrução
`standalone/cap/`, `repo.py` resolvendo imports por cópia, uma cópia paralela do SA).

## Outcome

A calibração agora mede o gap do SA contra o ótimo provado do MIP (HiGHS fechou nas 4
instâncias em segundos). A configuração recomendada (`lhs_32`) fica a 1,96% do ótimo em
média na revalidação, contra 23,8% do default do repositório. O ganho vem principalmente
de gastar o orçamento de iterações e de usar uma penalidade baixa; o valor dessa
penalidade ainda precisa ser validado em instâncias maiores. Detalhes em
`experiments/sa_calibration/REPORT.md`.

## What changed

- `src/fine-tuning/` inteiro saiu do repositório (movido para `.trash/` local, fora do git):
  eram cópias do código de `src/` ou scripts que dependiam delas. O que era genuinamente
  novo (α por orçamento, previsão de iterações, traço) foi reimplementado no lugar certo.
- `src/solver/heuristics/simulated_annealing.py`: `callback` opcional em `solve`, métricas
  `zmax`/`zmin`/`std_load`/`time`/`feasible` no resultado, `cooling_rate_for_budget` e
  `iterations_for_schedule`; `_balance_objective` passou a usar o novo `_per_cycle_mean`.
  Tudo retrocompatível (os testes existentes passam sem mudança).
- `src/calibration/` (novo): instâncias, MIP de referência, gap, espaço de busca
  (LHS + Optuna), executor paralelo, análise, I/O e gráficos. Ver `docs/calibration.md`.
- `experiments/sa_calibration/` + `configs/experiments/sa_calibration/`: driver e YAMLs.
- `tests/experiments/sa_calibration/`: testes da instrumentação do SA, do gap, da config,
  do executor, da análise, do MIP (solver mockado), do I/O e das instâncias.

## Notes

- Escopo combinado com o usuário: só o SA foi modificado entre os arquivos existentes; MIP,
  `main.py`, `persistence`, `config`, `generate`, `forecasting`, `pyproject.toml`,
  `Makefile` e `.gitignore` ficaram intactos. Consequências: a sequência de chamadas de
  `src/calibration/instances.py` repete a de `src/main.py` (sem copiar funções); o Optuna é
  instalado com `uv run --with optuna` (não há extra `calibration`); o `.gitignore` de
  `outputs/` é local ao experimento.
- O índice `docs/agent-log/README.md` não foi atualizado nesta entrada (fora do escopo
  combinado); o usuário atualiza.
- Em 80 setores o ótimo do MIP é o mesmo para qualquer `capacity_multiplier` viável (o
  churn é que prende); `tight80` usa 1,03 × o mínimo viável para que o As-Is comece acima
  da capacidade.
- Não testados, por exigirem um gancho de agenda no SA (fora do escopo): resfriamento em
  patamares e resfriamento adaptativo por taxa de aceitação.
