# Curva de `num_leaves` e o cruzamento Etapa 4 × Etapa 6

2026-09-22. Sessão seguinte a `2026-09-22-roadmap-stages-3-to-6.md`.

## O que foi pedido

Duas perguntas, nesta ordem: mudar o número de folhas do LightGBM ajuda? E
as etapas 4 (afinação) e 6 (painel de dois anos) somam quando aplicadas
juntas?

## Curva de `num_leaves` (painel de um ano, 5 sementes por ponto)

A grade da etapa 4 só ofereceu 7, 15 e 31, então a preferência dela pelo
menor valor não distinguia um ótimo de uma borda. Varrendo 2 a 63 na
configuração vencedora: 7 é um **ótimo interior**, mas o fundo é raso —
de 3 a 7 a variação total é 1,6 item contra desvios de 1,3 a 2,2 entre
sementes, ou seja, um platô e não um ranking. Acima de 10 a degradação é
monótona e acelera: 31 folhas devolvem 25 dos 36,1 itens que todo o plano
havia ganhado sobre o `naive:mean`. Tabela completa em
`experiments/lightgbm_tuning/README.md`.

Conclusão prática: `num_leaves` é o único hiperparâmetro do conjunto que
merece ser protegido, e a resposta à pergunta original é "não ajuda, mas
errar nele destrói".

## Cruzamento Etapa 4 × Etapa 6 — hipótese refutada

Experimento novo: `tuned_two_year`. A configuração afinada **perde** para a
não afinada no painel de dois anos, por 38,1 itens com as features do ano
anterior e 24,7 sem elas, nas três sementes. A mesma configuração valia
−36,1 itens no painel de um ano.

Duas hipóteses minhas caíram em sequência, e vale registrar porque o
padrão é instrutivo:

1. **"O culpado é o `feature_fraction`, que dilui a coluna lag 19."**
   Errado — decompondo um fator por vez, a amostragem de colunas é o único
   botão dos três que *ajuda* (−6,1). A parada antecipada custa +29,1 e a
   taxa 0,02 custa +16,0.
2. **"A parada antecipada machuca porque sacrifica o ciclo mais recente do
   treino."** Também errado. Com orçamentos fixos e sem parada nenhuma —
   todos treinando em todos os ciclos — o erro piora monotonicamente com o
   número de rodadas (100 → 2408,5 até 1600 → 2443,3). O braço com parada
   antecipada interpola em ~1100 rodadas. A regra escolheu um orçamento dez
   vezes maior que o ótimo; o ciclo perdido custou aproximadamente nada.

## O que isso muda no plano

A baseline de dois anos fica **sem afinar**. E a afirmação da etapa 4 de que
a parada antecipada "não é bem uma escolha de hiperparâmetro" foi corrigida
no README dela: é uma escolha, e depende do painel. Hiperparâmetro validado
num painel não transfere para outro painel do mesmo problema — o que é a
segunda vez que este plano descobre que um resultado depende do regime de
dados, depois do achado transversal das etapas 1, 2, 3 e 5.

Pergunta aberta que vale um experimento: por que a regra de parada erra por
uma ordem de grandeza? Ou o conjunto de validação (um ciclo, ~728 linhas) é
ruidoso demais para parar em cima dele, ou a métrica discorda do placar — o
boosting para no L2 da razão enquanto o experimento ranqueia por MAE de
itens.

## Erro de processo desta sessão

Rodei duas varreduras em paralelo numa máquina de 8 núcleos; a carga foi a
17 e as duas travaram. Tasso mandou serializar. Também perdi um lote inteiro
por rodar um script do scratchpad como arquivo, o que põe o diretório do
script no `sys.path` em vez da raiz do repositório — resolvido com
`PYTHONPATH`, mas o lote anterior tinha saído vazio e só o log denunciou.
