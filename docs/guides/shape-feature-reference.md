# Features do shape

O modelo `lightgbm` de shape permite selecionar estas features em
`learning.features`. Elas não afetam os modelos `uniform` e `sector`.

| Nome no YAML | Informação | Origem |
| --- | --- | --- |
| `CD_RE` | Região, tratada como categoria | Base de demanda |
| `CD_GV` | Gerência de vendas, tratada como categoria | Base de demanda |
| `weekday` | Dia da semana, segunda=0 até domingo=6, tratado como categoria | Data do dia previsto |
| `month` | Mês, janeiro=1 até dezembro=12, tratado como categoria | Data do dia previsto |

Feriados não estão implementados nesta etapa. Não é necessário um CSV de calendário.

O LightGBM sempre recebe `cd_setor`, `relative_position` e `n_days`. A posição
relativa é `(offset + 0,5) / n_days`, contando `offset=0` na abertura. Use
`features: []` para avaliar somente essas três entradas básicas.

`src/forecasting/shape_learning.py → feature_table` constrói essas colunas.
`historical_windows` leva os últimos códigos de região e gerência conhecidos
no treino para os ciclos avaliados: não usa o cadastro futuro do holdout. Setores
novos e códigos não vistos são tratados como categorias ausentes. Região e
gerência precisam existir na base quando solicitadas pelo YAML. Códigos
conflitantes dentro de um setor–ciclo geram erro na preparação do painel.

## Configuração

O arquivo `configs/experiments/compare_shapes/calendar_features.yaml` compara
os dois modelos anteriores, LightGBM básico, LightGBM só com dia da semana
e LightGBM com as quatro features:

```yaml
- name: lightgbm_organization_calendar
  model: lightgbm
  learning:
    features: [CD_RE, CD_GV, weekday, month]
    n_estimators: 100
    learning_rate: 0.05
    num_leaves: 15
    min_child_samples: 30
```

Para medir o ganho só do dia da semana, compare o candidato `lightgbm_weekday`
(`features: [weekday]`) com `lightgbm_position` na mesma execução.
Mantenha o restante dos parâmetros e o split iguais. Para avaliar as outras
features, adicione-as gradualmente, com nomes distintos para os candidatos.

Na raiz do repositório:

```bat
.venv\Scripts\python.exe -m experiments.compare_shapes.run configs/experiments/compare_shapes/calendar_features.yaml
```

O teste usa uma pasta nova (`shape_calendar_v01`), sem sobrescrever `shape_v01`.
No dashboard, clique **Atualizar pastas** e selecione a nova execução. Os modelos,
features e parâmetros efetivos ficam registrados em `run_manifest.json`.
O pacote LightGBM é necessário e já está instalado neste ambiente; em outro
ambiente, use as dependências opcionais `forecast` do projeto.

## Como a curva é aprendida

`shape_learning.py → predict_learned_shape` usa os ciclos de treino com total
positivo para aprender `actual_share × n_days`, uma intensidade relativa diária.
Cada linha recebe peso `1 / n_days`, dando peso total igual a cada curva de treino.
A API nativa do LightGBM evita depender do adaptador de scikit-learn.

As intensidades previstas são limitadas a valores não negativos e normalizadas
por setor–ciclo para somar 1. Se todas forem zero, a curva vira uniforme e o
fallback é reportado. Não há treinamento com demanda dos ciclos avaliados, nem
ajuste automático de hiperparâmetros no holdout. Durante a avaliação, o total
real do ciclo apenas transforma as participações previstas em itens, como nos
outros modelos de shape.
