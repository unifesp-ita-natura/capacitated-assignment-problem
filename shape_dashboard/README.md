# Shape dashboard

Front local e independente para explorar as pastas produzidas pelo backtest de
shape. Não lê, executa ou altera o fluxo de level. Não precisa de internet,
Node.js ou dependências extras de gráficos: usa Python/pandas e SVG no navegador.

## Executar

Na raiz do repositório, com o ambiente virtual existente:

```bat
.venv\Scripts\python.exe -m shape_dashboard.serve
```

O navegador abre em **http://127.0.0.1:8002/**. Deixe o terminal aberto;
Ctrl+C encerra o servidor. Se a porta estiver ocupada, acrescente `--port 8003`.

A raiz padrão é `experiments/compare_shapes/outputs/`. Para outra raiz:

```bat
.venv\Scripts\python.exe -m shape_dashboard.serve --runs caminho\da\pasta\de\execucoes
```

Passe a **pasta que contém as execuções**, não o caminho do predictions.csv.
Por exemplo, `outputs/shape_v01/predictions.csv` aparecerá como `shape_v01`
quando a raiz for `outputs/`. Clique em **Atualizar pastas** após novos testes.
Pastas sem predictions.csv não aparecem. Se ainda não houver resultados:

```bat
.venv\Scripts\python.exe -m experiments.compare_shapes.run configs/experiments/compare_shapes/sector_day.yaml
```

## Filtros e gráficos

1. Selecione uma pasta de resultados (o cenário de teste).
2. Selecione um ciclo e a origem da previsão. Origens diferentes não são somadas.
3. Mantenha todos os modelos para comparar curvas ou escolha somente um.
4. Digite/selecione um setor; campo vazio inclui todos os setores.
5. Alterne entre participação diária (%) e itens por dia.

- **Real versus previsto**: distribuição da demanda por dia relativo à abertura
  da janela do setor (Dia 1 é a abertura). Passe o mouse
  nos pontos para ver os números.
- **Acumulado**: mostra se o modelo antecipa ou atrasa a demanda no ciclo.
- **Erro diário**: previsto − real; positivo indica superestimação naquele dia.
- **Métricas do recorte**: mesmas definições de `compare_shapes`, calculadas nos
  filtros atuais. São erros dos pontos individuais, antes de agregar setores nos
  gráficos. Por isso, erros entre setores podem se cancelar no gráfico e ainda
  aparecer nas métricas.
- **Ponto a ponto**: primeiras 200 linhas do recorte por modelo/setor/data.
  **Baixar pontos filtrados** exporta todas as linhas filtradas em CSV.
- **Ranking completo e parâmetros**: comparison.csv e run_manifest.json
  originais, sem filtros; também é possível baixar config_snapshot.yaml.

Com todos os setores, o gráfico alinha as aberturas e soma os itens por dia
relativo, mesmo quando os setores abriram em datas diferentes. A participação exibida
é o total diário dividido pelo total real do recorte: uma distribuição
ponderada pelo volume. Para estudar a curva de um setor, filtre-o.
As métricas de participação continuam dando peso igual às curvas, como no
backtest. Curvas de total zero não têm participação definida; exibem “—”.

Os gráficos e métricas usam pontos diários comuns entre os modelos selecionados.
Os arquivos do backtest atual têm a mesma cobertura. O download contém todos
os pontos que atendem aos filtros. `items_pred` já vem do backtest como
`share_pred × total_real_do_ciclo`; isso avalia o shape isoladamente. O bias total
tende a zero por conservação, mesmo quando o modelo erra bastante em cada dia.

## Implementação

`shape_dashboard/serve.py → api → shape_dashboard/data.py → view` prepara
os dados do recorte; `app.js → render/plots` desenha a interface. O servidor
mantém um cache de até quatro CSVs, filtra antes de enviar os dados e escuta
somente em localhost. Os arquivos dos testes são apenas lidos.
