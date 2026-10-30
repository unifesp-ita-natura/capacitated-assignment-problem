# Analisar blocos de features na hierarquia

Este experimento compara o modelo completo com variantes que retiram um
bloco de features de cada vez. O histórico, parâmetros, cadastro consultado,
cenários, pontos avaliados e demais blocos ficam iguais. O objetivo é medir
efeito fora do treino nos folds, não interpretar importância de splits como
prova de melhoria ou piora.

## Executar

Na raiz do repositório, com o ambiente virtual preparado:

```powershell
.\.venv\Scripts\python.exe -m experiments.hierarchical_level.ablation configs/experiments/hierarchical_level/management_city_ablation.yaml
```

O YAML usa a hierarquia região → gerência → cidade, 20 ciclos fechados de
treino e os mesmos nove alvos 202604 a 202612. `paths.reference_run` aponta
para uma execução completa com a mesma base e configuração. `paths.run_dir`
precisa ser uma pasta nova.

Os YAMLs de ablação existentes usam `comparison: historical` para conferir
as três saídas aprendidas contra os arquivos de referência antigos.
Manifestos anteriores ao campo `comparison` são interpretados como históricos
somente nessa verificação. Uma referência histórica não é aceita como se
fosse uma execução padrão. O backtest usual agora compara apenas média móvel
pura de cinco ciclos, compartilhamento e reconciliação; essa mudança não
altera os números dos arquivos de referência.

O executor reproduz o modelo completo e verifica MAE, WMAPE, Bias, pontos,
falta de cobertura e folds contra a referência. Compartilha as tabelas de
features entre variantes e usa `_learned_predictions()` do mesmo fluxo
produtivo. Não recalcula as médias móveis, que não dependem dos blocos do
LightGBM. Elas continuam disponíveis na execução de referência. Apenas os
cenários históricos observados são suportados neste executor.

## Variantes

| Variante | Features retiradas do LightGBM |
|---|---|
| `full` | Nenhuma |
| `without_history` | Lags, médias móveis, desvio e tendência da demanda |
| `without_calendar` | Mês, número do ciclo, durações e descritores das aberturas |
| `without_structure` | Quantidade de filhos, atividade recente dos filhos e mudança na quantidade |
| `without_hierarchy` | Participações 5/10, média do pai e participação anterior do maior filho |
| `without_intermittency` | Frequência de compra, distância desde compra, média positiva e tamanho do histórico |
| `without_city_recent_shares` | Participações 1/2/3 da cidade no pai |

`node` e `parent` continuam categóricos em todas as variantes. O bloco é
retirado em todos os LightGBMs internos que o utilizam. As previsões ETS,
as referências históricas para regularização, a distribuição e a reconciliação
continuam presentes. Retirar o calendário do LightGBM não retira o ajuste
de duração do ETS. Retirar as features `hierarchy` não remove a hierarquia
nem a regra de distribuição.

## Arquivos gerados

A pasta principal contém configuração, features, manifesto, folds,
`comparison.csv`, `metrics_by_cycle.csv`, `ablation_effects.csv` e
`ablation_effects_by_cycle.csv`. As tabelas têm a coluna `variant`.

Em `variants/<variante>/` ficam o YAML efetivo, catálogo de features,
`predictions.csv`, métricas principais/completas e `fitted_models.json`.
Para reproduzir só uma variante no executor normal, copie seu YAML e altere
`paths.run_dir` para uma pasta nova; o executor normal também calculará as
médias móveis configuradas.

As métricas principais de cidade incluem somente demanda real positiva.
Zeros históricos continuam no treino; compradores novos sem previsão contam
como erro. Os diagnósticos `*_all_points.csv` mantêm todos os pontos.

## Interpretar

Nos arquivos de efeitos, `mae_delta = MAE_sem_bloco − MAE_completo`.
Um valor negativo indica melhora ao retirar o bloco. `mae_reduction_pct`
positivo indica redução relativa do erro. O sinal de `bias_pct_delta` não
basta para decidir melhoria: compare a distância do Bias a zero.

Observe os nove ciclos individualmente. Uma melhora global concentrada em
poucos ciclos não prova benefício estável. As comparações reutilizam alvos
de desenvolvimento: qualquer seleção de features continua exploratória e
precisa ser confirmada em ciclos posteriores. Este teste identifica efeito
conjunto de blocos; não identifica causalmente a responsabilidade de cada
variável nem demonstra que todas as suas combinações terão o mesmo resultado.

Os resultados da primeira execução estão no
[relatório de análise dos blocos](../reports/management-city-feature-ablation.md).
A rodada concluída usa 20 ciclos; a execução adicional com 10 não foi autorizada.
