# SA: energia de pico e desvio, avaliação incremental, comparação de solvers e escala

**Date:** 2026-10-19
**Related:** commit `ed87162` (branch `fine-tuning`); plano
`docs/plans/2026-10-19-correcoes-e-proximos-passos.md`; experimento
`experiments/sa_calibration/`

## Task

Seguir a calibração do Simulated Annealing (ver `2026-09-28-sa-calibration-highs.md`) com
três perguntas: (1) uma energia de pico e desvio, em vez da amplitude, dá uma carga diária
melhor? (2) o Gurobi e o HiGHS chegam ao mesmo ótimo, e com que tolerância? (3) o SA
continua competitivo em 400 e 800 setores, onde o MIP é mais caro?

## Outcome

Na instância de 800 setores gerada, o Gurobi fecha o ótimo em cerca de 20 s, mais rápido e
exato do que o SA de 200 mil iterações (gap médio de 8,3% em 27 s com a energia de
amplitude). O SA só se justificaria numa escala em que o MIP não fecha, e a instância
gerada usa no máximo 3,7% da capacidade dos CDs, então essa pergunta continua em aberto
(item 9 do plano). A energia de pico e desvio, calibrada, reduz o desvio da carga em 1 a
3%, mas piora o gap da amplitude (11,0% contra 8,3% em 800 setores) e leva cerca de
1,8 vez o tempo. A energia de amplitude continua sendo o padrão.

## What changed

- `src/solver/heuristics/simulated_annealing.py`: `energy_mode` (`"range"` ou
  `"peak_std_breach"`, com `w_peak`, `w_std`, `w_range`) e avaliação incremental dos
  vizinhos (`_evaluate_move`, `_candidate_evaluation`, `_resync`). O padrão não muda e as
  trajetórias do modo `range` são idênticas às do SA anterior.
- `src/calibration/`: opções de tolerância e limites do MIP por solver (`mip.py`),
  modo de energia e pesos na análise e no executor, e `comparison.py` para comparar
  Gurobi, HiGHS e SA.
- `experiments/sa_calibration/compare_solvers.py` e as configs `peak_std.yaml`,
  `solver_comparison.yaml` e `scale800*.yaml` (`configs/experiments/sa_calibration/`).
- Testes novos em `tests/solver/heuristics/` e `tests/experiments/sa_calibration/`.
- O plano de correções e próximos passos (`docs/plans/`), que consolida o que ficou em aberto.
- `docs/calibration.md` foi removido nesse commit e substituído por um índice (ver Notes).

## Notes

- **Duas mudanças no mesmo commit.** A avaliação incremental só acelera, e as trajetórias
  são idênticas às da avaliação completa (testes com hash de atribuições). Quem altera o
  gap é a energia `peak_std_breach`. Ao comparar rodadas, mudar uma de cada vez.
- **A energia nova precisa de calibração própria.** Aplicada com os parâmetros calibrados
  para a amplitude (`lhs_32`, ρ ≈ 146), ela dá gap de 14,6% a 38,9% e 0/8 execuções viáveis
  em `tight40` (`outputs/solver_comparison/`). Calibrada (`tpe_b0_04`, ρ ≈ 311) fica em 7,4%
  (400 setores) e 11,0% (800 setores).
- **Gap de 800 setores cai com mais iterações:** 8,3% com 200 mil e 4,2% com 1 milhão
  (energia de amplitude), em 155 s.
- **Tolerância do MIP.** Com 1%, o Gurobi parou 0,26% acima do ótimo em `base40` e o HiGHS
  devolveu o ótimo. A comparação usa `mip_rel_gap = 1e-4` como referência.
- **Avaliação incremental e CDs múltiplos.** `_evaluate_move` soma o delta só no CD
  dominante do setor. Se um setor passar a ter carga dividida entre CDs (branch
  `feat/sa-cd-shares`), o delta precisa atualizar todos eles e os hashes de referência
  dos testes têm de ser regravados.
- **Documentação.** Esta entrada, o índice e a atualização de `docs/calibration.md`,
  `experiments/sa_calibration/README.md` e `docs/simulated_annealing_explicado.md` foram
  escritos em 2026-10-27, a partir do commit, do plano e das saídas locais de
  `experiments/sa_calibration/outputs/` (fora do git). Os números acima vêm dessas saídas;
  nenhum foi remedido.
- Pendências conhecidas: `energy_mode` ainda não está em `src/solver/README.md`, e a
  tabela de itens do plano ainda lista o item 16 (documentação) como aberto.
