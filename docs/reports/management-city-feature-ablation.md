# Features na hierarquia: análise por retirada de blocos

Data: 27/10/2026. Execução:
`experiments/hierarchical_level/outputs/2026-10-27_management_city_ablation_v01/`.

## Método e verificações

Modelo região → gerência → cidade, com os mesmos 20 ciclos globais fechados
de treinamento, nove alvos 202604 a 202612, hiperparâmetros, sementes,
cadastro e janelas de abertura da referência. Foram avaliados o modelo completo
e seis variantes, retirando apenas um bloco de preditores de cada vez.

As métricas principais de cidade usam os mesmos 38.945 pontos com demanda
positiva, excluindo 33.930 pontos sem demanda. Os 726 pontos ativos sem
previsão continuam contando como erro. Treino, cadastro e referências históricas
mantêm zeros. As tabelas de features são compartilhadas entre as variantes,
mas as matrizes entregues ao LightGBM respeitam a seleção de cada variante.

`learned_only()` em `experiments/hierarchical_level/ablation.py` chama
`_learned_predictions()` de `src/forecasting/hierarchy.py`, o mesmo caminho
dos modelos existentes. O executor pula somente o recálculo das médias
móveis, que não dependem da seleção de features do LightGBM.

O controle completo reproduziu ponto a ponto as previsões e os valores reais
da referência `2026-10-27_management_city_v01`, além das métricas e dos folds.
A seleção registrada foi conferida contra as features dos modelos efetivamente
ajustados. Foram 7 variantes, 3 candidatos aprendidos por variante e 9 origens.

## Resultado no nível gerência-cidade

MAE menor é melhor; as demais configurações permanecem iguais.

| Bloco retirado do LightGBM | MAE compartilhamento | MAE reconciliação |
|---|---:|---:|
| Nenhum: modelo completo | 230,74 | 235,01 |
| Histórico: lags, médias, desvio e tendência | 230,50 | 235,11 |
| Calendário e duração | 230,84 | 237,29 |
| Estrutura: quantidades de filhos e alterações | 230,63 | 235,05 |
| Hierarquia: participações 5/10, média do pai e maior filho | 231,02 | 235,33 |
| Intermitência | 230,79 | 234,63 |
| Participações recentes da cidade: 1/2/3 | 230,56 | 235,07 |

### Compartilhamento

- Retirar o bloco histórico deu o menor MAE: 230,50 contra 230,74, redução
  de apenas 0,106%, com melhora em 5/9 ciclos. WMAPE passou de 49,7910% para
  49,7373%; Bias permaneceu praticamente igual, −17,8930% para −17,8970%.
- Retirar `share_1`, `share_2`, `share_3` melhorou 7/9 ciclos, mas somente
  0,078% no MAE agregado: 230,56. WMAPE 49,7471% e Bias −17,8877%.
- Retirar estrutura melhorou 6/9 ciclos, com redução de somente 0,047% no MAE.
- Retirar as participações históricas 5/10 e outros preditores do bloco
  `hierarchy` piorou 7/9 ciclos. O bloco continua fornecendo informação útil
  sob estes parâmetros, embora o efeito agregado seja pequeno.

As três pequenas melhorias sugerem redundância ou ajustes desnecessários
no contexto deste teste; não demonstram que cada variável desses blocos seja
prejudicial. O ganho máximo de MAE ficou abaixo de 0,25 item por cidade.

### Reconciliação

- Retirar intermitência teve o menor MAE: 234,63 contra 235,01, redução de
  0,164%, com melhora em 5/9 ciclos. WMAPE passou de 50,3531% para 50,2749%
  e Bias de −16,3590% para −14,7759%.
- Retirar calendário piorou MAE em 0,971%, para 237,29, e agravou Bias
  para −23,9743%. Não há evidência para remover esse bloco no objetivo ativo.
- As demais retiradas pioraram ligeiramente o MAE da cidade.

Os melhores resultados de reconciliação ainda ficam acima da MA20 distribuída
de referência (MAE 232,93) e do compartilhamento completo (230,74).

## Efeito relevante nos níveis superiores

As retiradas afetam todos os LightGBMs internos que consomem o bloco, não
apenas o de cidade. Remover os preditores históricos na reconciliação gerou:

| Nível | MAE completo | MAE sem bloco histórico | Redução relativa | Ciclos com melhora |
|---|---:|---:|---:|---:|
| Região | 38.930,87 | 35.555,34 | 8,67% | 6/9 |
| Gerência | 11.826,98 | 11.114,15 | 6,03% | 7/9 |
| Cidade ativa | 235,01 | 235,11 | piora de 0,04% | 3/9 |

Nos agregados, WMAPE passou de 11,3461% para 10,3623% na região e de
15,9418% para 14,9810% na gerência. Bias dos agregados passou de −5,2988%
para −2,4584%. Na cidade ativa, o Bias melhorou de −16,3590% para −14,1469%,
apesar da pequena piora de MAE e WMAPE (50,4476%).

Portanto, a reconciliação sem o bloco histórico é uma candidata a testar
quando o objetivo combina agregados e cidades. Não é a menor MAE de cidade,
e esta comparação não define automaticamente o peso dos diferentes objetivos.

## O que este teste não remove

Retirar `history` significa retirar lags, médias móveis, desvio e tendência
como preditores do LightGBM. Continuam presentes os 20 ciclos de treino,
o ETS, os demais preditores históricos (como participações), as referências
de regularização e os identificadores categóricos `node`/`parent`.

Retirar calendário do LightGBM não desliga o ajuste de duração do ETS.
Retirar `hierarchy` não remove a árvore organizacional, a distribuição nem
a reconciliação. Os identificadores categóricos, que são muito utilizados,
não foram retirados neste experimento. O arquivo de contagem de splits é
descritivo, não uma medida de dano ou benefício.

## Diagnóstico com todos os pontos

O critério principal continua restrito às cidades com compra, como solicitado.
Os diagnósticos completos mostram que os objetivos podem divergir: retirar
calendário na reconciliação piorou o MAE ativo de 235,01 para 237,29, mas
reduziu o MAE com zeros de 200,39 para 198,09. Retirar intermitência melhorou
o MAE ativo, mas piorou o completo de 200,39 para 200,65. Esses resultados
não justificam trocar o critério após ver os números; foram preservados
apenas como diagnóstico do comportamento fora do recorte principal.

## Conclusão e limites

No nível cidade, não foi identificado um bloco responsável por uma grande
piora: os ganhos de retirada foram menores que 0,2% no MAE. Não há base para
retirar indiscriminadamente features. Participações recentes e estrutura
parecem pouco úteis ao compartilhamento neste recorte; intermitência teve
pequeno efeito negativo na reconciliação. O sinal mais forte foi o bloco
histórico na reconciliação dos agregados, com pouco efeito no MAE da cidade.

Os nove alvos já foram usados no desenvolvimento. A escolha da melhor
variante é exploratória, sem significância estatística demonstrada e sem
validação em novos ciclos. Não foram testadas combinações de retiradas nem
cada variável individualmente. O efeito de retirar dois blocos não pode ser
obtido somando os ganhos das retiradas isoladas. As configurações principais
dos modelos foram preservadas.

O teste adicional com dez ciclos teve sua execução recusada na confirmação
de permissão; não foi iniciado nem repetido. Portanto, estes resultados não
explicam diretamente a piora observada ao reduzir o treino para dez ciclos.

## Arquivos e reprodução

Na pasta principal da execução:

- `analysis_summary.csv`: métricas de cidade, efeitos e contagem de ciclos;
- `comparison.csv` e `metrics_by_cycle.csv`: todos os níveis e variantes;
- `ablation_effects.csv` e `ablation_effects_by_cycle.csv`: diferenças contra o controle;
- `analysis_checks.json`: conferência de controle, população e features ajustadas;
- `full_city_share_split_counts.csv`: uso das features em splits do modelo completo;
- `config.yaml`, `features.json`, `folds.csv`, `manifest.json`: parâmetros e proveniência.

Cada `variants/<variante>/` contém sua configuração, features, previsões,
métricas principais/completas e metadados dos modelos ajustados. Para
reproduzir, veja o [guia da análise](../guides/hierarchical-feature-ablation.md).

Validação: 25 testes passaram, com 93% de cobertura do novo executor;
Ruff passou, todas as funções verificadas ficaram em Radon A e
`git diff --check` passou. Os algoritmos principais e o shape não foram alterados.
