# Hierarquia por gerência e cidade: resultados

Data: 27/10/2026. Execução:
`experiments/hierarchical_level/outputs/2026-10-27_management_city_v01/`.

## Desenho do experimento

Prever itens por região → gerência → cidade, somando os setores da mesma
gerência que atendem o mesmo município. A identidade da cidade é
`CD_RE`, `CD_GV`, `estado`, `cidade`; não há setor como nível intermediário.
`aggregate_nodes()` em `src/forecasting/hierarchy_data.py` aplica os agrupamentos
definidos por `level_keys()` em `hierarchy_config.py` antes do treinamento.

Mesma base (hash confirmado) e mesmos nove folds da execução anterior com
participações recentes: alvos 202604 a 202612, histórico de 20 ciclos fechados
antes da primeira abertura global de cada alvo, mesmos hiperparâmetros e
blocos de features. `city_grouping: sector` continua sendo o padrão anterior.
Configuração nova: `configs/experiments/hierarchical_level/management_city.yaml`.

Compartilhamento corresponde a `distribution_ets_lgbm`: ETS na região e
LightGBMs compartilhados de participação na gerência e cidade. Reconciliação
corresponde a `reconciliation_ets_lgbm`: ETS regional e LightGBMs Tweedie de
volume, ajustados conjuntamente por mínimos quadrados ponderados não negativos.
São gerados por `_learned_predictions()` em `src/forecasting/hierarchy.py`.

## População e avaliação

Foram 15 candidatos e 1.099.200 linhas de previsões. Cada candidato tem:

- 72 pontos regionais, oito regiões;
- 333 pontos de gerência, 37 gerências;
- 72.875 pontos de gerência-cidade-ciclo, 8.523 identidades distintas;
- 38.945 pontos de cidade com demanda positiva, cobrindo 7.115 identidades;
- 33.930 pontos de cidade com demanda zero excluídos das métricas principais.

Entre as observações ativas, 726 não tinham previsão no cadastro histórico;
seu volume conta como erro. A cobertura de volume ativo é 99,7772%.
`summarize()` em `hierarchy_evaluation.py` aplica o filtro somente depois de
gerar as previsões. Treino, cadastro e referências históricas mantêm zeros.
Os arquivos `*_all_points.csv` preservam o diagnóstico completo.

## Resultado principal em gerência-cidade

| Alternativa | MAE | WMAPE | Bias |
|---|---:|---:|---:|
| Média móvel 10 + distribuição histórica | 233,22 | 49,9738% | −12,9157% |
| ETS + compartilhamento LightGBM | 230,74 | 49,7910% | −17,8930% |
| ETS + LightGBM reconciliado | 235,01 | 50,3531% | −16,3590% |

A média móvel acima é a melhor variante pelo WMAPE entre as janelas 3, 5,
10 e 20 e as abordagens direta, distribuída e reconciliada. A menor MAE entre
as médias móveis é 232,93 da distribuição MA20, que tem WMAPE 50,1251% e
Bias −5,4041%. A escolha da janela é exploratória sobre o próprio teste.

O compartilhamento supera ligeiramente a melhor média móvel em WMAPE
(0,1828 ponto percentual, aproximadamente 0,37% de redução relativa) e em
MAE, mas subestima mais. A reconciliação perde para ambos em MAE e WMAPE.
Não foi demonstrada significância estatística.

## Resultados dos níveis superiores na nova arquitetura

| Nível | Alternativa | MAE | WMAPE | Bias |
|---|---|---:|---:|---:|
| Região | Média móvel 5 + reconciliação | 40.052,58 | 11,6730% | −7,0367% |
| Região | Compartilhamento | 38.659,38 | 11,2670% | −6,5225% |
| Região | Reconciliação | 38.930,87 | 11,3461% | −5,2988% |
| Gerência | Média móvel 5 + distribuição histórica | 11.915,37 | 16,0610% | −7,1685% |
| Gerência | Compartilhamento | 11.859,48 | 15,9856% | −6,5225% |
| Gerência | Reconciliação | 11.826,98 | 15,9418% | −5,2988% |

As médias móveis são selecionadas por menor WMAPE dentro de cada nível.
Mudar os filhos muda features de estrutura e a reconciliação conjunta; por
isso os resultados de gerência e região não precisam ser iguais aos antigos.

## Comparação justa com a arquitetura anterior

Não se comparou o MAE 220,31 por setor-cidade diretamente ao MAE 230,74 por
gerência-cidade. As previsões antigas foram somadas primeiro no novo nível,
incluindo todos os zeros, e só depois filtradas pela demanda agregada positiva.
As mesmas 38.945 combinações de gerência-cidade-ciclo, a mesma demanda real,
o mesmo hash de base e os mesmos folds foram verificados nas duas abordagens.

| Abordagem | MAE antigo agregado | MAE novo | WMAPE antigo agregado | WMAPE novo | Bias antigo agregado | Bias novo |
|---|---:|---:|---:|---:|---:|---:|
| Compartilhamento | 231,41 | 230,74 | 49,6925% | 49,7910% | −17,7013% | −17,8930% |
| Reconciliação | 233,46 | 235,01 | 49,9230% | 50,3531% | −11,8214% | −16,3590% |

O compartilhamento melhora um pouco a MAE, mas piora WMAPE e Bias. A nova
reconciliação piora as três métricas. Portanto, o treinamento agregado por
gerência-cidade não trouxe melhora geral sobre somar a arquitetura anterior,
neste teste. Mudaram simultaneamente o nível das séries e a profundidade da
hierarquia; o resultado não isola causalmente cada mudança.

O vínculo setor-gerência da base inteira foi usado exclusivamente para
agregar previsões arquivadas na comparação. Foi confirmado que cada setor
pertence à mesma gerência/região em toda a base, evitando ambiguidades de
vínculo. Esse procedimento não gerou previsões nem features com dados futuros.

## Arquivos e validação

Na pasta da execução ficam configurações, catálogo de features, modelos
ajustados, previsões, folds e métricas. Arquivos adicionais:

- `checks.json`: métricas recalculadas, chaves únicas, previsões não negativas,
  somas coerentes e equivalência dos dados/folds;
- `architecture_comparison.csv`: comparação das duas arquiteturas no mesmo nível;
- `architecture_comparison_checks.json`: verificações de população e demanda.

Validação: 23 testes passaram, com 96% de cobertura combinada dos módulos de
configuração, dados e avaliação (97%, 92% e 97%, respectivamente). Ruff passou,
as funções verificadas mantiveram complexidade Radon A e `git diff --check`
não apontou erros. Os testes anteriores e o shape foram preservados.

Para reproduzir, veja [o guia de gerência-cidade](../guides/hierarchical-management-city.md)
e escolha uma pasta de saída nova no YAML.
