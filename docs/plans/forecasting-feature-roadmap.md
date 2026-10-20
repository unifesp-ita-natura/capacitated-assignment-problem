# Plano de testes: novas features para a previsão de demanda

Roteiro incremental para tentar superar o `naive:mean`, que hoje é o melhor
candidato (`experiments/compare_forecasters`). Cada etapa é **um experimento,
uma variável, um commit**, na ordem em que o retorno esperado cai e o custo
sobe.

A regra que governa o plano: **nada entra no modelo final sem passar pela
Etapa 0.** Sem ela, todo ganho medido é ganho aparente.

## Onde estamos

| fato medido | valor | de onde vem |
|---|---|---|
| melhor candidato hoje | `naive:mean`, MAE 1,851 | `compare_forecasters/README.md` |
| LightGBM atual | MAE 1,900 (2º), 606 pontos perdidos | idem |
| oráculo que conhece a média exata do setor | MAE 1,658 | idem |
| **espaço total em disputa** | **~10%** | 1,851 → 1,658 |
| autocorrelação do desvio em relação à média do setor | −0,065 | idem |
| variância entre setores / dentro do setor | 21% / 79% | idem |
| erro relativo por setor / por CD | 41,0% / 12,5% | idem |

Duas leituras que o plano assume como verdadeiras até que um experimento as
derrube: (a) no nível do setor há pouquíssimo a ganhar além do nível médio, e
(b) a métrica que importa para o MIP é por CD-dia, não por setor.

---

## Etapa 0 — Protocolo de avaliação (pré-requisito, não é feature)

**Por que primeiro.** O README do `compare_forecasters` já registra a falha:
a grade de `num_leaves` foi pontuada nos mesmos folds do resultado. Com um
espaço em disputa de ~10% e diferenças candidatas de ~3%, selecionar e medir
no mesmo lugar produz ganho fantasma. Se as etapas seguintes rodarem sobre o
protocolo atual, não saberemos distinguir feature boa de sorte.

**0.1 — Holdout final.** Reservar os 2 ciclos mais recentes do painel, que
**nenhum experimento deste plano pode tocar**. Toda seleção acontece nos folds
anteriores; o holdout é aberto uma única vez, no fim, para reportar o número
honesto no paper.

**0.2 — Métrica por CD-dia ao lado da MAE por setor.** É o nível em que a
restrição de capacidade age. Exige a junção `setor → rota → CD` (563 de 564
rotas do GI apontam para um CD único — ver
`docs/context/natura-client-answers.md`), que substitui a aproximação atual de
"CD dominante". Pode eleger um vencedor diferente, e isso por si só já é um
resultado publicável.

**0.3 — Múltiplas sementes.** Toda comparação envolvendo LightGBM roda com 5
sementes e reporta média e desvio. Sem isso não dá para afirmar que uma
diferença de 3% não é ruído do próprio boosting.

**Critério de saída:** `compare_forecasters` re-executado sob o novo protocolo,
com os números atuais reafirmados ou corrigidos.

---

## Etapa 1 — Pedidos e volumes no painel

**Experimento:** `panel_order_counts`

**O que varia:** `build_item_panel` passa a agregar também
`total_pedidos_mascarado` e `total_volumes_mascarado`; o LightGBM ganha
`lag_1..4` dessas duas séries e a razão itens/pedido do ciclo anterior.

**O que fica fixo:** split, alvo, hiperparâmetros, conjunto de candidatos.

**Hipótese:** ganha. Medido em 6.705 pares (setor, ciclo→ciclo seguinte),
`pedidos_{k-1}` correlaciona **0,230 (Spearman)** com os itens do ciclo
seguinte, contra **0,180** de `itens_{k-1}`. Pedido ≈ consultora ativa, que é
estável; itens/pedido é volátil. O painel hoje descarta o preditor melhor.

**Ressalva registrada antes de rodar:** depois de dividir pela média do
setor, a correlação cai para −0,08 nas duas séries. Então o ganho esperado é
em **estimar o nível com menos ruído**, não em prever movimento. Se o ganho
aparecer no `naive:mean` equivalente (média dos pedidos × itens por pedido) e
não no LightGBM, a conclusão é sobre o painel, não sobre o modelo.

**Critério de sucesso:** MAE no subconjunto comum abaixo de 1,851.

**Custo:** baixo. Nenhum dado novo.

**Status: concluída em 2026-09-22 — feature mantida, hipótese derrubada.**
O LightGBM caiu de 1892,0 para 1873,2 itens (−1,0%), melhorando em 4 dos 5
folds, mas **não** bateu o `naive:mean` (1834,3). O teste pré-registrado
falhou: um preditor ingênuo montado sobre pedidos (média de pedidos × média
da razão itens/pedido) dá 1841,4 e perde para a média simples de itens, então
o ganho **não** vem de "pedidos medem o nível com menos ruído". A ablação
mostra que nenhuma série responde sozinha pelo ganho — orders, volumes e a
razão entram todas numa faixa de 4 itens. Leitura mais provável, a ser testada
na Etapa 4: com 7 folhas o modelo está limitado por capacidade, e colunas
correlacionadas extras dão mais pontos de corte, não informação nova.

O critério das 5 sementes foi abandonado nesta etapa por um motivo técnico: a
configuração não tem componente estocástico — `feature_fraction` e
`bagging_fraction` estão no padrão 1,0 e não há bagging — então toda semente
treina o mesmo modelo. Sementes só passam a informar na Etapa 4. Ver
`experiments/panel_order_counts/README.md`.

**Duas consequências para as etapas seguintes.** A Etapa 5 (fator de ciclo)
ganha peso: o modelo pooled vence o `naive:mean` justamente no fold em que os
setores mais se afastam das próprias médias. E a Etapa 2 (janela curta) vale
mais do que o custo sugere: o primeiro fold é o único em que a feature piora
e é também o que dá ao `naive:mean` quase toda a sua margem — e é exatamente
o fold que a janela de 6 ciclos joga fora.

---

## Etapa 2 — Janela móvel curta

**Experimento:** `lightgbm_short_window`

**O que varia:** `rolling_windows` de `[3, 6]` para `[2, 3]`.

**O que fica fixo:** tudo o mais, incluindo o que a Etapa 1 tiver aprovado.

**Hipótese:** recupera cobertura sem perder precisão. Contei as linhas de
treino disponíveis por fold sob a configuração atual:

| fold | treina nos ciclos | linhas de treino |
|---|---|---|
| 1 | 202601–202606 | **0** |
| 2 | 202601–202607 | 571 |
| 3 | 202601–202608 | 1.143 |
| 4 | 202601–202609 | 1.715 |
| 5 | 202601–202610 | 2.286 |
| 6 | 202601–202611 | 2.857 |

O `rolling_mean_6` exige 6 ciclos anteriores completos, então o fold 1 treina
em **zero linhas** e devolve predição vazia — são os 606 pontos perdidos. Uma
janela de 2–3 ciclos devolve o fold 1 e aumenta as linhas de treino em todos os
demais.

**Critério de sucesso:** `n_missing` cai para perto de zero **e** a MAE não
piora. Se a MAE piorar, a informação também é útil: significa que a média longa
é que estava carregando o modelo.

**Custo:** trivial — uma linha de config.

**Status: concluída em 2026-09-22 — critério atendido, com uma ressalva que
importa.** `n_missing` foi de 606 para **zero**: o modelo passa a prever os
3.689 pontos, a mesma cobertura do `naive:mean`. E a MAE no subconjunto comum
melhorou de 1873,2 para 1862,5. Mas o ganho inteiro vem de **um único fold**:
no 202607, o mais pobre em dados, encurtar a janela vale 176 itens; nos três
folds do meio, com treino farto, a janela de 6 ciclos é **melhor**, em média
41 itens. O mecanismo real não é "janela curta é melhor", e sim "uma média de
6 ciclos é impagável cedo e ligeiramente melhor tarde".

**Consequência para a Etapa 6:** com o segundo ano no painel todos os folds
viram fartos — o regime em que `[3, 6]` venceu aqui. A janela longa pode
voltar, e a janela de um ano inteiro fica viável pela primeira vez. A escolha
de `[2, 3]` é provisória e deve ser reavaliada lá. Ver
`experiments/lightgbm_short_window/README.md`.

---

## Etapa 3 — Alvo relativo

**Experimento:** `relative_target`

**O que varia:** o LightGBM passa a treinar em `items / rolling_mean_k` e a
predição é remultiplicada pela média do setor.

**Hipótese:** ganha, e é a mudança estruturalmente mais importante do plano.
Hoje o modelo precisa aprender 633 níveis diferentes usando uma categórica de
633 valores com `num_leaves: 7` — uma árvore de 7 folhas faz 6 cortes no total
e não consegue separar 633 setores. Na prática o `cd_setor` quase não é usado e
o modelo vira uma função global dos lags, competindo com o `naive:mean` sem a
única coisa que o `naive:mean` usa. Normalizar o alvo tira essa tarefa da
árvore.

**Por que razão e não subtração:** a variação absoluta cresce com o tamanho do
setor (1,609 → 2,400 itens do menor ao maior quartil) enquanto a relativa fica
perto de 0,45.

**Critério de sucesso:** ganho sobre a Etapa 2, com a mesma cobertura.

**Status: concluída em 2026-09-22 — hipótese confirmada.** 1876,9 → 1859,5
itens (−0,93%), com a mesma cobertura. A distância acumulada até o
`naive:mean` caiu de 57,7 para **19,6 itens** em três etapas. Só 3 dos 6 folds
melhoram, mas os dois ganhos são grandes (+61,8 e +50,2) e as perdas pequenas
— exceto no 202606, o fold mais pobre em dados, onde a razão piora 47,2 e
sozinho responde pela maior parte do déficit restante. **O padrão da Etapa 2 se
repete: toda mudança ajuda onde há dados e atrapalha onde eles são escassos.**
O denominador usado foi a média corrente do setor (o que o `naive:mean` prevê),
e não `rolling_mean_k` como o plano dizia, porque assim uma razão ajustada em
1,0 reproduz o benchmark exatamente. Ver `experiments/relative_target/README.md`.

---

## Etapa 4 — Configuração honesta do LightGBM

**Experimento:** `lightgbm_tuning`

**Por que só agora:** afinar hiperparâmetros antes de acertar alvo e features é
afinar o modelo errado.

**O que varia:** `n_estimators: 2000` com early stopping sobre um fold interno
de validação, `learning_rate: 0.02`, `feature_fraction` e `bagging_fraction` em
0,8, `lambda_l2` varrido, `min_child_samples` varrido. Grade completa, 5
sementes, seleção **nos folds de seleção apenas**.

**Hipótese:** ganho pequeno, de 1 a 3%. A configuração atual (100 árvores a
lr 0,05, sem early stopping, sem regularização explícita, uma semente) pode
estar subtreinada e ninguém está medindo isso — mas a grade de sensibilidade já
feita mostrou preferência monótona por modelos menores, o que é evidência de
que há pouca estrutura a ajustar.

**Status: concluída em 2026-09-22 — o benchmark caiu.** O modelo passou de
1859,5 para **1803,9** itens e bateu o `naive:mean` (1840,0) pela primeira
vez, em **todas as 5 sementes** (média 1806,8, dp 1,8). Mas o ganho veio quase
todo de um lugar só: **só o early stopping vale 46,9 dos 55,6 itens**; todo o
resto da grade junto soma 8,7. A configuração antiga não estava mal ajustada —
estava **parando cedo demais**, e ninguém media isso porque não havia conjunto
de validação.

**Defeito da grade, registrado:** `bagging_fraction` deu resultados idênticos
em 0,7 e 1,0 — 48 linhas, 24 resultados distintos. O LightGBM ignora esse
parâmetro com `bagging_freq: 0`, que ficou no padrão. Re-testado direito, o
bagging não ajuda (0,7 → 1808,3 contra 1803,9 sem ele), então a conclusão não
muda — mas a grade publicada testou quatro eixos, não cinco.

**A pergunta da Etapa 1, respondida:** com capacidade de verdade, os pedidos e
caixas ainda valem **8,1 itens** (1806,8 contra 1814,9, faixas de semente sem
sobreposição) — menos da metade dos 18,8 que valiam com o modelo estrangulado.
O palpite estava metade certo: metade do ganho da Etapa 1 era capacidade,
metade era informação.

**Correção ao próprio plano:** o roadmap pôs a afinação em quarto lugar com o
argumento de que "afinar antes de acertar o alvo é afinar o modelo errado".
Estava certo sobre a ordem e errado sobre o tamanho — o maior ganho isolado de
todo o plano estava numa regra de parada que ninguém tinha medido. Ver
`experiments/lightgbm_tuning/README.md`.

---

## Etapa 5 — Fator de ciclo comum a todos os setores

**Experimento:** `cycle_factor`

**O que varia:** decompor `items = nível_do_setor × fator_do_ciclo` e prever o
fator como uma série própria, em vez de deixá-lo implícito no `opening_month`.

**Hipótese:** ganha, e talvez mais do que qualquer feature de setor. Os itens
totais por ciclo, indexados pela média, variam assim:

```
c01:0,62  c02:0,92  c03:0,95  c04:1,00  c05:1,02  c06:1,13
c07:0,87  c08:1,04  c09:1,04  c10:1,20  c11:1,16  c12:1,05
```

São **±20% atingindo todos os setores ao mesmo tempo** — quase certamente
revista e promoção, que o cliente confirmou serem definidas na abertura do
ciclo e não terem calendário disponível. Prever uma série de 12 pontos é um
problema muito mais fácil que prever 633 séries.

**Observação que motiva a etapa:** `cycle_number` e `opening_month` são
constantes dentro de um ciclo e crescem monotonicamente. Com janela expansiva,
o ciclo-alvo sempre tem `cycle_number` maior que qualquer valor visto no treino
— e árvore não extrapola. Essas duas features, como estão, no máximo codificam
"o ciclo mais recente".

**Status: concluída em 2026-09-22 — hipótese REJEITADA, e com folga.** Todas as
quatro formas de prever o fator perdem para o `naive:mean`: a melhor
(`mean3`) dá 1926,0 contra 1840,0. O motivo não é que o efeito de ciclo não
exista — é que **ele não é previsível**. A autocorrelação lag-1 da série de
fatores é **+0,110**. E o fator fica sistematicamente acima de 1, então aplicá-lo
infla toda previsão, o que a MAE pune; por isso a variante `all`, que é uma
correção constante de ~1,17, é a **pior** das quatro.

**Correção ao próprio plano:** a justificativa desta etapa usava a série de
*itens totais por ciclo* (0,62 … 1,20). Aquilo é outra quantidade — mistura o
efeito de ciclo com mudanças de quais setores foram observados. Medido
corretamente, contra o que a história de cada setor esperava, o efeito é menor e
praticamente sem memória.

**Um bug de estimador que vale lembrar:** a primeira implementação estimava o
fator como média das razões, e a série abria em **3,51** — parecia um salto
espetacular de demanda, era artefato de dividir por uma média construída com um
ou dois ciclos. Razão de somas cortou a dispersão por cinco (σ 0,701 → 0,138).
Ver `experiments/cycle_factor/README.md`.

---

## Etapa 6 — Segundo ano de histórico

**Experimento:** `two_year_panel`

**O que varia:** o painel passa a ser construído a partir de
`Unifesp_Demanda_v2` (dez/2024 a set/2026) em vez de `base_tratada.csv`
(dez/2025 a set/2026). Destrava `lag_19` (mesmo ciclo do ano anterior) e médias
móveis anuais.

**Por que é seguro:** a frase do cliente "incluí dados aleatórios a partir de
Dezembro/24" foi verificada e significa *amostra aleatória de pedidos reais*.
Para os 643 setores presentes nos dois períodos, a média do setor no período
antigo correlaciona **0,858** com a do período novo (Spearman 0,931), e os
totais mensais repetem a mesma forma nos dois anos. Detalhes em
`docs/context/natura-client-answers.md`.

**Hipótese:** é a maior fonte de sinal não testada, porque é a única etapa que
traz informação que hoje não existe no painel. Todas as outras reorganizam o
que já está lá.

**Riscos a controlar:** os totais de 2025 são sistematicamente maiores que os de
2026 no mesmo mês — pode ser queda real ou taxa de amostragem diferente entre
períodos. O experimento precisa testar as duas hipóteses antes de usar o nível
absoluto. O calendário de 2024–2025 também **não tem bloco/sub-bloco**, o que
não atrapalha a previsão mas impede reconstruir a política da época.

**Custo:** médio. Exige reconstruir o pipeline de carga.

**Status: concluída em 2026-09-22 — o maior ganho do plano.** No painel de dois
anos (728 setores × 32 ciclos, 12 folds) o modelo dá **2468,9 contra 2775,4** do
`naive:mean` — **11,0% melhor**. O lag de 19 ciclos vale **64,1 itens (2,5%)**
sozinho, e é a primeira feature do plano inteiro que traz informação que o painel
não continha.

**Duas ressalvas que precisam viajar junto com o número.** Primeiro, é **outro
painel**: os níveis de erro não se comparam com nenhuma etapa anterior, só as
linhas da tabela entre si. Segundo, parte da vitória é só recência — com 20+
ciclos, o `naive:mean` mistura um período de nível mais alto. Medindo: a média
dos 6 últimos ciclos dá 2677,6, então **~98 dos 306 itens de vantagem são
recência** e ~209 são o modelo. Citar os 11% sem essa decomposição seria
exagerar.

**Repetir o mesmo ciclo do ano anterior é desastroso** (`seasonal_naive`,
3884,7). O valor do ano passado serve como *feature que o modelo pondera*, não
como previsão. Ver `experiments/two_year_panel/README.md`.

---

## Etapa 3 × Etapa 4 — as duas somam?

**Experimento:** `target_x_tuning`

**O que varia:** alvo (nível vs. razão) cruzado com a configuração (sem afinar
vs. vencedora da etapa 4), no painel de um ano. As linhas sem afinar reproduzem
as etapas 2 e 3, então são o controle.

**Por que fazer:** a grade da etapa 4 tinha `target: ratio` no bloco `fixed`, ou
seja, as duas etapas foram **empilhadas mas nunca cruzadas**. A etapa 3 mediu o
alvo em razão em 17,4 itens contra um modelo **sem afinar**, e ninguém verificou
se esse valor sobrevive à afinação — um modelo que treina muito mais tempo
poderia decorar os 633 níveis e dispensar a ajuda do alvo.

**Hipótese (registrada antes):** o alvo em razão importa **mais** depois de
afinar. A função dele é impedir que o modelo gaste capacidade reaprendendo o
nível de 633 setores, e afinar aumentou o orçamento de rodadas — que é
exatamente o que permite a um alvo em nível decorar esses níveis.

**Status: concluída em 2026-09-22 — hipótese confirmada, com folga.**

| | alvo em nível | alvo em razão | vantagem da razão |
|---|---|---|---|
| sem afinar | 1876,95 | 1859,55 | **17,4** |
| afinado (5 sementes) | 1862,39 ± 3,52 | **1806,84 ± 1,75** | **55,6** |
| ganho de afinar | 14,6 | **52,7** | |

`naive:mean` = 1839,98 nos mesmos 3.689 pontos.

**As duas mudanças interagem, e a interação é maior que qualquer efeito isolado.**
Afinar vale 14,6 itens no alvo em nível e 52,7 no alvo em razão: **38,1 itens só
existem com os dois juntos.** O número de manchete da etapa 4 não é propriedade
da afinação, é propriedade do par.

**Afinar sozinho não bate o benchmark:** o braço afinado+nível dá 1862,4, ou seja
22,4 itens *pior* que o `naive:mean`, e as cinco sementes perdem. Se a grade da
etapa 4 tivesse rodado no alvo em nível, a etapa inteira teria sido lida como
fracasso.

**O controle reproduz a etapa 3 exatamente:** os dois braços sem afinar dão
1876,95 e 1859,55 contra os 1877,0 e 1859,5 publicados, então os braços afinados
estão no mesmo painel, nos mesmos folds e na mesma base.

**Consequência para o plano:** as etapas deste roadmap **não são aditivas e
precisam parar de ser reportadas como se fossem**. Em um único dia apareceram
duas interações apontando na mesma direção — a configuração da etapa 4 inverte de
sinal em outro painel (`tuned_two_year`) e inverte de sinal em outro alvo (este).
Qualquer experimento futuro que mexa no orçamento de boosting precisa manter o
alvo em razão fixo, ou vai medir a coisa errada.

---

## Etapa 4 × Etapa 6 — as duas somam?

**Experimento:** `tuned_two_year`

**O que varia:** a configuração (etapa 3 sem afinar vs. a vencedora da etapa 4)
cruzada com as features do ano anterior (presentes ou não), tudo no painel de
dois anos. As linhas sem afinar reproduzem a etapa 6, então ela é o controle.

**Hipótese (registrada antes):** somam, mas não integralmente. O ganho da etapa
4 veio quase todo da parada antecipada, e um painel com 32 ciclos dá mais linhas
por fold — exatamente o regime onde um orçamento longo deveria render *mais*.

**Status: concluída em 2026-09-22 — hipótese refutada.** Afinar **piora**:
2446,5 contra 2408,5 do controle (com ano anterior) e 2500,9 contra 2476,2 (sem).
Custa **38,1 itens**, e as três sementes de cada braço afinado perdem para o seu
controle — não é sorteio. A mesma configuração valia **−36,1 itens** no painel de
um ano; mudar de painel inverte o sinal.

**O botão culpado é a parada antecipada**, justamente o herói da etapa 4.
Decompondo um fator por vez: parada antecipada **+29,1**, taxa 0,02 **+16,0**,
`feature_fraction` 0,7 **−6,1** (o único que ajuda, contra a hipótese registrada).

**E a causa não é o ciclo sacrificado, é o orçamento.** Com orçamentos fixos e
sem parada nenhuma — todos treinando em *todos* os ciclos — o erro piora de forma
monótona: 100 → 2408,5, 200 → 2411,0, 400 → 2424,0, 800 → 2434,2, 1600 → 2443,3.
O braço com parada antecipada deu 2437,6, que interpola em **~1100 rodadas**: a
regra escolheu um orçamento dez vezes maior que o ótimo, e o ciclo perdido do
treino custou aproximadamente nada.

**Consequência para o plano:** a baseline de dois anos fica **sem afinar**, e o
plano precisa parar de assumir que um botão validado uma vez continua validado.
Hiperparâmetro achado num painel não transfere para outro painel do mesmo
problema. Fica uma pergunta aberta que vale um experimento: por que a regra de
parada erra tanto? Ou o conjunto de validação (um ciclo, ~728 linhas) é ruidoso
demais, ou a métrica discorda do placar — o boosting para no L2 da **razão**
enquanto o experimento ranqueia por MAE de **itens**.

---

## Etapa 7 — Calendário do ciclo

**Experimento:** `cycle_calendar_features`

**O que varia:** entram `Qtde dias` (duração real do ciclo, nem sempre 21),
número de dias úteis na janela e número de feriados nacionais.

**Hipótese:** ganho pequeno mas barato e defensável. Um ciclo com um dia útil a
menos vende menos, e isso é sabido antes de o ciclo abrir — é das poucas
features genuinamente conhecidas com antecedência.

**Nota de implementação:** usar um calendário nacional externo, **não** a coluna
`feriado` da base, que só registra feriados em dias que tiveram pedido —
encontrei feriados marcados em apenas 6 dos 12 ciclos.

---

## Etapa 8 — Features de setor e cold start

**Experimento:** `sector_covariates`

**O que varia:** substituir a categórica de 633 níveis por descritores
numéricos: média histórica, coeficiente de variação, quartil de tamanho, número
de rotas, número de CDs que atendem o setor, UF dominante, prazo de transporte
mediano (`PRAZO TOTAL` do GI: mediana 7 dias, máximo 20).

**Hipótese:** ganho modesto no agregado, ganho real em setores com pouco
histórico. Uma árvore de 7 folhas consegue cortar em variáveis numéricas; em
633 níveis categóricos, não. É também o que dá sentido ao "cross-learning"
prometido pelo modelo pooled.

**Sobre as variáveis do IBGE** (densidade populacional, média salarial, taxa de
desemprego): entram **aqui e rotuladas como cold start**, não como tentativa de
bater o baseline. O motivo é estrutural: elas são praticamente constantes na
janela de 9 meses, então só podem explicar diferença *entre* setores — os 21%
— e para isso a média histórica do próprio setor já é a medição direta daquilo
que elas são apenas um proxy indireto. O caso em que ganham é o setor novo ou
rezoneado, que não tem média histórica.

**Viabilidade da junção geográfica, já medida:** um setor cobre **mediana de 11
cidades, até 142** — então dado por cidade exige ponderar por participação de
itens. Por UF é limpo: apesar de 385 setores aparecerem em mais de um estado,
**no setor mediano 99,9% dos itens ficam num único estado**, e só 24 setores são
genuinamente divididos. Se usar IBGE, usar por UF.

**Fora do plano:** taxa de inflação. O alvo é contagem de itens, não
faturamento; a série é nacional, logo idêntica para os 633 setores dentro de um
ciclo; e seriam 12 observações colineares com qualquer tendência temporal.

---

## Critério de parada

**Atualização de 2026-09-22: o critério não foi acionado.** A Etapa 4 bateu o
`naive:mean` no painel de um ano (1803,9 contra 1840,0, em todas as sementes) e
a Etapa 6 bateu com folga no painel de dois anos (2468,9 contra 2775,4). O que
segue continua valendo como critério para o caso de as próximas etapas
estagnarem, e a migração para a métrica por CD-dia segue sendo recomendada de
qualquer forma, porque é o nível em que o MIP opera.

Se as etapas restantes terminarem sem ganho adicional sobre o `naive:mean` no
holdout final, a conclusão honesta é que **no nível do setor não há mais
estrutura a extrair**, e o esforço deve migrar para:

1. a métrica por CD-dia, onde os erros se cancelam parcialmente (41,0% → 12,5%)
   e onde o MIP realmente opera; e
2. o lado da otimização, tratando a incerteza da demanda explicitamente
   (margem de segurança na capacidade) em vez de tentar eliminá-la na previsão.

Isso não é fracasso: com autocorrelação de −0,065 nos desvios e um oráculo a
apenas 10% de distância, "o histórico do próprio setor é o melhor preditor" é um
resultado legítimo — e é, inclusive, o que a Natura já faz hoje, segundo o
próprio cliente.

## Pedidos ao cliente que podem alterar o plano

- **Taxa de amostragem da base.** Bloqueia a calibração da capacidade em valor
  absoluto (a demanda observada roda a 10–30% dos limites informados).
- **Contagem de consultoras por zoneamento.** O cliente disse ter e não enviou.
  É conceitualmente o denominador da demanda, e entraria na Etapa 8 com
  prioridade alta.
- **Ciclos de estratégia.** O cliente negou ter calendário de promoções, mas
  mencionou "ciclos de estratégia". Se existirem em forma tabular, vão direto
  para a Etapa 5, que é onde o efeito de ±20% mora.
