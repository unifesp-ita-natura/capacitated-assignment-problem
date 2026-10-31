# Previsão por gerência e cidade

Este experimento prevê itens por gerência–estado–cidade–ciclo. Setores da
mesma gerência que atendem a mesma cidade são somados antes do treinamento.
Uma cidade em duas gerências gera duas séries. Estado distingue municípios
homônimos; a região também integra a identidade organizacional da gerência.
O shape e os executores anteriores de level não são alterados.

## Configuração e execução

Arquivo: `configs/experiments/hierarchical_level/management_city.yaml`.

```yaml
settings:
  levels: [region, management, city]
  city_grouping: management
  history_cycles: 20
```

`city_grouping: sector` continua sendo o padrão das configurações anteriores.
Para `management`, a gerência deve aparecer imediatamente antes de `city`;
não se pode colocar setor entre gerência e cidade porque a cidade agregada
pode receber demanda de vários setores.

Com o ambiente do projeto preparado, na raiz do repositório:

```powershell
.\.venv\Scripts\python.exe -m experiments.hierarchical_level.run configs/experiments/hierarchical_level/management_city.yaml
```

A execução usa `data/base_tratada_v2.csv`, os mesmos nove alvos de 202604 a
202612 e 20 ciclos globais encerrados antes da primeira abertura de cada alvo.
Os parâmetros de aprendizado e os seis blocos de features são os do teste
anterior com participações recentes. Para repetir, escolha outra pasta em
`paths.run_dir`; resultados existentes não são sobrescritos.

O exemplo mantém região como nível inicial para comparar com a arquitetura
anterior. É possível usar `levels: [management, city]` para colocar o ETS
diretamente na gerência; essa é outra configuração, não o teste descrito aqui.

## Modelos comparados

- Distribuição: ETS prevê o total regional; LightGBM compartilhado prevê
  participação da gerência na região e da cidade na gerência. Os scores são
  normalizados entre irmãos antes de distribuir o total superior.
- Reconciliação: ETS regional e LightGBM Tweedie para os volumes de gerências
  e cidades; mínimos quadrados ponderados ajustam todos os níveis em conjunto.
- Referência padrão: média móvel pura de cinco observações em cada nó,
  sem distribuição por participações e sem reconciliação.

O fluxo gera somente esses três candidatos. As regras de treino, features
e avaliação permanecem iguais. Não há modo histórico nem parâmetro para
alterar a janela da média móvel. Resultados antigos permanecem arquivados,
e sua implementação pode ser consultada no commit `69b9ece`.

A regularização de cidades com poucas compras continua presente. `share_1`,
`share_2`, `share_3`, `share_5` e `share_10` agora medem participação da cidade
na gerência. Lags, médias e frequência de compra usam a série agregada, não
a previsão antiga somada depois. Nenhuma feature usa a demanda do alvo.

## Calendário e cadastro

`aggregate_nodes()` em `src/forecasting/hierarchy_data.py` soma os itens uma
única vez e resume as janelas com pesos iguais por setor distinto dentro do
nó. Duração é a média das durações dos setores, não a união de todos os dias
de abertura. Abertura do nó é a menor abertura dos setores e os demais
descritores mantêm mínimo, máximo e médias. As janelas históricas do alvo
são tratadas como planejamento conhecido, conforme o guia principal.

O cadastro da consulta continua sendo construído exclusivamente do passado.
Uma cidade agregada conhecida pode receber previsão mesmo sem compra no
alvo. Cidades novas com compra continuam contando como erro e falta de
cobertura. A avaliação de vínculos conhecidos usa o último cadastro
organizacional disponível, como nos testes anteriores.

## Avaliação e arquivos

`comparison.csv` e `metrics_by_cycle.csv` avaliam cidade somente com demanda
agregada real positiva naquele ciclo. Os níveis superiores usam seus totais
completos. Os arquivos `*_all_points.csv` preservam o diagnóstico com zeros.
As previsões mantêm todos os pontos e `included_in_primary_metrics` informa
quais entram na análise principal.

Os arquivos de configuração, features, modelos ajustados, folds e manifesto
são salvos junto dos resultados. `city_identity_keys` registra explicitamente
`CD_RE`, `CD_GV`, `estado`, `cidade` no manifesto e em `features.json`. As
previsões de cidade têm esses identificadores e não um setor artificial.

Não compare diretamente o MAE com o MAE por setor–cidade: a unidade avaliada
mudou. Compare primeiro os modelos e médias móveis neste mesmo experimento.
Para comparar arquiteturas, é preciso somar as previsões antigas no mesmo
nível gerência–cidade e igualar os pontos avaliados.

Veja também [o guia da hierarquia](hierarchical-level.md).

Para testar se algum bloco de features prejudica os modelos, veja o
[guia de remoção controlada de features](hierarchical-feature-ablation.md).

## Resultados históricos

A comparação antiga entre dez e vinte ciclos de treino foi concluída em
27/10/2026. Os YAMLs exclusivos desse estudo foram retirados do fluxo atual;
os parâmetros salvos com as execuções e o código do commit `69b9ece`
continuam preservados. Dez ciclos pioraram o MAE nas duas abordagens. Veja o
[relatório de comparação dos históricos](../reports/management-city-history-length.md)
para métricas, baselines e conferência da população avaliada.

### Primeira execução

Os nove ciclos foram executados com o catálogo antigo em 27/10/2026.
Na cidade ativa, compartilhamento
teve MAE 230,74, WMAPE 49,79% e Bias −17,89%; reconciliação teve MAE 235,01,
WMAPE 50,35% e Bias −16,36%. A melhor média móvel por WMAPE foi MA10 com
distribuição histórica: MAE 233,22 e WMAPE 49,97%.

As previsões antigas também foram somadas no novo nível antes de comparar.
O novo compartilhamento melhorou levemente MAE, mas piorou WMAPE e Bias;
a reconciliação piorou as três métricas. Consulte o
[relatório com resultados e comparação no mesmo nível](../reports/management-city-hierarchy.md).
