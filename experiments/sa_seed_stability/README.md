# sa_seed_stability

**Name:** `sa_seed_stability` — driver `experiments/sa_seed_stability/run.py`, config
`configs/experiments/sa_seed_stability/full.yaml`, helpers in `src/pipeline/stability.py`,
outputs in `experiments/sa_seed_stability/outputs/` (fora do git).

**Status:** `exploratory`

## Goal

A melhoria que o SA entrega sobre o As-Is é estável entre seeds e entre cenários de
previsão, ou a recomendação depende de um sorteio de sorte?

## Hypothesis

CV do objetivo entre seeds abaixo de 2%; variação entre cenários maior que entre seeds;
melhoria realizada (demanda real) abaixo da prevista, porque o SA também otimiza o erro de
previsão.

## Design note

- Pipeline de `configs/pipeline.yaml`: 20 ciclos de histórico, 6 de horizonte, churn 20%,
  base completa (sem `max_sectors`).
- 3 cenários de previsão: `hierarchical_sector`, `hierarchical_management`, `naive_mean`.
  Prevê-se uma vez por cenário; o SA roda com as seeds 1–10 na mesma instância.
- Cada atribuição é avaliada com a mesma métrica (amplitude diária média por ciclo) na
  demanda **prevista** e na **real** do horizonte (mesma grade cidade-dia da previsão, itens
  reais no lugar dos previstos; dias previstos sem pedido valem 0). Melhoria = queda
  percentual em relação ao As-Is na mesma demanda.
- Retomável: `runs.csv` recebe uma linha por (cenário, seed); `--overwrite` recomeça.

```bash
uv run --extra forecast python -m experiments.sa_seed_stability.run
```

## Result

10 seeds por cenário (30 execuções). Média ± desvio (mín–máx):

| cenário | melhoria prevista (%) | melhoria realizada (%) | objetivo To-Be previsto | objetivo To-Be realizado |
|---|---|---|---|---|
| hierarchical_sector | 26.7 ± 2.4 (23.5–30.6) | 18.1 ± 2.6 (13.4–21.9) | 34 877 ± 1 139 (CV 3.3%) | 107 605 ± 3 437 (CV 3.2%) |
| hierarchical_management | 26.3 ± 2.4 (23.2–29.3) | 17.7 ± 3.7 (9.1–22.0) | 35 134 ± 1 159 (CV 3.3%) | 108 077 ± 4 809 (CV 4.5%) |
| naive_mean | 26.6 ± 2.5 (23.3–31.1) | 18.6 ± 2.9 (13.6–24.3) | 41 627 ± 1 400 (CV 3.4%) | 106 956 ± 3 775 (CV 3.5%) |
| todos | 26.5 ± 2.4 (23.2–31.1) | 18.1 ± 3.0 (9.1–24.3) | — | 107 546 ± 3 935 (CV 3.7%) |

- **A recomendação não é um sorteio de sorte no sinal:** em 30/30 execuções a melhoria
  realizada é positiva (9–24%). Mas a magnitude varia: o CV do objetivo entre seeds é
  ~3–4,5%, **acima** dos 2% esperados (hipótese refutada), e a melhoria realizada oscila
  ±3 pontos percentuais (pior caso 9,1%, `hierarchical_management`, seed 1).
- **A previsão pesa mais que a seed na média, não na dispersão:** a melhoria prevista
  (~26,5%) é ~8 pontos acima da realizada (~18%), nos três cenários. Entre cenários a média
  da melhoria realizada difere menos de 1 ponto, bem dentro do ruído das seeds; então, com
  10 seeds, não há evidência de que um cenário de previsão seja melhor que outro para
  decidir a atribuição (o `naive_mean` empata com o hierárquico).
- As seeds correlacionam entre cenários (a seed 1 é ruim em todos, a 3–4 são boas): a
  qualidade do ótimo local depende da seed mais que do cenário.
- **Leitura para o cliente:** esperar ~18% de melhoria real (faixa plausível 13–22%), não
  os ~26% do relatório sobre a previsão. Rodar várias seeds e ficar com a de menor
  objetivo previsto é barato; ver ressalvas.

## Caveats

- O campo `seconds` saiu 0 em todas as execuções (`AnnealingResult.time` não foi
  preenchido nesse caminho); não usar esta rodada para tempos.
- Escolher a melhor seed pelo objetivo **previsto** não garante a melhor realizada
  (correlação não medida aqui); vale checar antes de recomendar "melhor de N".
- Uma janela de horizonte só (últimos 6 ciclos); a variação entre janelas não foi medida.
- O objetivo realizado (~107 mil) é ~3× o previsto: a previsão suaviza picos reais, então
  só a melhoria percentual (não o nível absoluto) é comparável.
