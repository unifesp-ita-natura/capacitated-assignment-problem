# Calibração do Simulated Annealing — CAP / desafio Natura

Código usado na calibração dos parâmetros do SA, pronto para rodar.

## O que tem aqui

```
simulated_annealing.py      substitui src/solver/heuristics/simulated_annealing.py
calibration/repo.py         resolve os imports (usa src.* no repo, cap.* standalone)
calibration/instances.py    monta as instâncias (espelha src/main.py até os inputs do solver)
calibration/calibrate.py    screen (grade) | refine (TPE) | bkv (execuções de referência)
calibration/finalize.py     run (revalidação) | trace (energia por iteração)
calibration/report.py       tabela final + gráfico de convergência
standalone/cap/             reconstrução do repo para rodar sem clonar nada
results/                    os CSVs e o PNG gerados na calibração
```

## Parada por estagnação desabilitada

O `simulated_annealing.py` daqui é o módulo do repositório com três mudanças,
todas compatíveis com o código existente:

1. `AnnealingParams.stagnation_enabled` — a parada por estagnação só dispara se
   `stagnation_window > 0`, `stagnation_tolerance > 0` **e**
   `stagnation_window < max_iterations`. `_stagnated` devolve `False` nos demais
   casos, sem precisar mexer no laço.
2. `calibration_params(max_iterations=200_000, **overrides)` — fábrica que já
   entrega os parâmetros com a estagnação desligada (`stagnation_window =
   max_iterations`, `stagnation_tolerance = 0.0`). É o que o harness usa em
   toda execução.
3. `solve(..., trace=True)` devolve `result.trace` com
   `(energia_corrente, melhor_energia, temperatura)` por iteração, e há dois
   auxiliares de agenda: `iterations_for_schedule(params)` e
   `cooling_rate_for_budget(n, T0, T_min)`.

```python
from src.solver.heuristics.simulated_annealing import calibration_params, solve

params = calibration_params(
    initial_temperature=5000.0,
    cooling_rate=0.999,
    min_temperature=0.001,
    sector_bias=0.80,
    destination_bias=0.80,
    penalty_coefficient=1e6,
)
assert not params.stagnation_enabled
result = solve(**instance_kwargs, params=params, rng=random.Random(42))
```

O default do repositório (`AnnealingParams()`) continua com a estagnação ligada:
nada do comportamento de produção muda por instalar este arquivo.

## Como rodar

**No repositório** — copie `simulated_annealing.py` para
`src/solver/heuristics/` e a pasta `calibration/` para a raiz do repo:

```bash
uv run python calibration/calibrate.py screen  --n-configs 60 --n-seeds 5
uv run python calibration/calibrate.py bkv     --n-seeds 3
uv run python calibration/calibrate.py refine  --screen results/screen.csv --batches 3
uv run python calibration/finalize.py run      --n-seeds 8
uv run python calibration/finalize.py trace    --config cal_T0_5000
uv run python calibration/report.py
```

`repo.py` detecta `src` automaticamente; se não achar, cai para `cap`.

**Sem o repositório** (foi assim que os resultados em `results/` foram gerados):

```bash
PYTHONPATH=standalone python3 calibration/calibrate.py screen --n-configs 60 --n-seeds 5
```

Dependências: `numpy`, `pandas`, `scipy` (TPE), `matplotlib` (gráfico). Não
precisa de Optuna nem de solver com licença.

## Instâncias

`calibrate.INSTANCE_SPECS` define três:

| nome | setores | horizonte | `capacity_multiplier` |
|---|---|---|---|
| `base40` | 40 | 2 ciclos | 1.0 |
| `tight40` | 40 | 2 ciclos | 0.003 |
| `base80` | 80 | 3 ciclos | 1.0 |

`tight40` existe porque com `CD_CAPACITIES` real e poucas dezenas de setores a
carga por CD fica em 2–6% da capacidade: `B(X)` nunca tem CD violado, `F(X)`
devolve todas as combinações, e `sector_bias`/`destination_bias` ficam sem
efeito. Com `0.003` o As-Is já viola um CD (`P_cap = 3.739`) e esses parâmetros
passam a estar ligados.

## Métrica

Gap percentual contra o **BKV** (melhor objetivo viável conhecido por
instância), não contra o ótimo do MIP — o ambiente onde isso rodou não tinha
Gurobi nem HiGHS. `score_from_rows` conta execução inviável como gap de 100%,
para que uma configuração não "ganhe" reportando um objetivo baixo numa solução
que estoura o churn. Quando você tiver um solver disponível, troque
`current_bkv` pelo objetivo do MIP e a métrica vira o gap que você pediu.

BKV usados: `base40` 5.056 · `tight40` 4.982 · `base80` 10.753.

## Resultado da revalidação (8 seeds × 3 instâncias)

| config | T0 | α | T_min | β | γ | ρ | gap médio | desvio | iterações | s/exec |
|---|---|---|---|---|---|---|---|---|---|---|
| `cal_no_bias` | 622 | 0.999 | 0.0129 | 0.50 | 0.50 | 1e7 | 10,30% | 4,64 | 10.779 | 1,17 |
| `cal_T0_5000` | 5000 | 0.999 | 0.001 | 0.80 | 0.80 | 1e6 | 10,72% | 3,63 | 15.418 | 1,67 |
| `cal_T0_500` | 500 | 0.999 | 0.001 | 0.65 | 0.80 | 1e6 | 11,40% | 4,40 | 13.116 | 1,42 |
| `default_repo` | 1000 | 0.99 | 0.01 | 0.80 | 0.70 | 1e6 | 18,06% | 3,35 | 1.146 | 0,13 |

As três calibradas são indistinguíveis entre si (diferença menor que um desvio);
todas batem o default por ~7-8 pontos percentuais, e a diferença entre elas e o
default é quase inteiramente orçamento: 15.418 contra 1.146 iterações.

## Duas coisas que o gráfico de convergência mostra

`results/report_convergencia.png`: a melhor energia para de cair por volta da
iteração 6.000, quando T ≈ 12. Daí até a 15.418 a busca está congelada —
`min_temperature = 0.001` é ordens de grandeza menor do que o necessário e só
gasta tempo. Vale testar `min_temperature` na faixa de 1 a 10, ou usar o tempo
que sobra para várias iterações por nível de temperatura.

E `max_iterations = 200_000` nunca é atingido: com resfriamento geométrico o
número de iterações é `ln(T0/T_min) / -ln(α)`, que com α = 0.999 dá ~15 mil.
Para usar o teto de verdade é preciso α ≈ 0.99993 (`cooling_rate_for_budget`) —
é a config `cal_long` do `finalize.py`, que não chegou a ser medida aqui
(~18 s por execução em 40 setores, ~31 s em 80).
