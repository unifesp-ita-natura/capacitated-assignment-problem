# O que a Natura já nos respondeu (thread de e-mail, 25/08 – 17/09/2026)

Registro do que o cliente afirmou sobre o negócio e os dados, para não
precisarmos reabrir a caixa de e-mail toda vez que uma premissa do modelo
depender de um fato de negócio. **Tudo aqui é afirmação do cliente**, não
verificação nossa — onde a base contradiz o e-mail, a seção "Dúvidas em
aberto" registra o conflito.

Interlocutores: Patricia Ueda, Ana Moraes e Daniel Bernardo Siqueira
(Advanced Analytics / Data & Analytics Hub). Canal oficial é e-mail, com os
três sempre em cópia; Luiza Santos centraliza o contato pelo nosso lado.
Reuniões remotas a cada 4 semanas.

## Estrutura do negócio

- **Ciclo:** a revista Natura tem ciclos numerados de 1 a 19 por ano, com
  duração média de 21 dias. `CICLOS` na base é `aa_ciclo` + `nm_ciclo`.
- **Setor comercial:** grupo de ~1.500 consultoras. **Uma consultora escolhe
  a qual setor quer pertencer**, e a regra é comercial, não geográfica — por
  isso o mesmo setor aparece associado a mais de um estado. O cliente não
  tem a regra documentada.
- **Pedidos:** cada consultora faz seus próprios pedidos ao longo de todo o
  ciclo, desde que atinja o valor mínimo. **A concentração acontece no
  fechamento do ciclo**, quando a consultora já prospectou e acumulou os
  pedidos dos clientes.
- **Relação CD ↔ setor:** definida por **zoneamento (conjunto de CEPs)**, não
  pela malha comercial. Por isso um mesmo setor pode ser atendido por mais de
  um CD, e um CD em SP (ex.: Cabreúva) pode atender o Nordeste. As duas
  malhas — comercial e logística — não seguem a mesma lógica.

## Capacidade dos CDs (resposta de 09/09)

Limites de separação, em **itens por dia**:

| CD | limite (itens/dia) | CD | limite (itens/dia) |
|---|---|---|---|
| 2700 | 350.000 | 5500 | 140.000 |
| 2800 | 44.000 | 5600 | 200.000 |
| 5100 | 1.000.000 | 5700 | 1.200.000 |
| 5300 | 120.000 | 5800 | 400.000 |
| 5400 | 350.000 | | |

São os nove CDs que aparecem em `data/base_tratada.csv`.

O cliente descreve a capacidade como "variável por dia" mas oscilando pouco
entre semanas e meses, e **essencialmente fixa**: o CD tem time fixo
capacitado, então aumentar ou reduzir não é simples. A única válvula de
escape citada é operar em domingos e feriados, e "esses casos são raros".
**Consequência para o modelo:** tratar a capacidade do CD como restrição
rígida está alinhado com o cliente — Luiza confirmou essa premissa por
e-mail em 04/09 e não houve objeção.

## Escopo de setores

Três números circulando (≈800 no case, 693 únicos na base de demanda, 633 na
aba Calendário FV do simulador). Resposta: **alguns setores passam por
rezoneamento ao longo do ano e deixam de existir; a estrutura atual é a do
calendário.** Ou seja, **633 é o número correto** — e é exatamente quantos
setores `data/base_tratada.csv` contém.

## Previsão de demanda — o que o cliente sabe

- **Hoje eles preveem usando apenas o histórico do setor.** Não há modelo
  com drivers externos.
- Perguntados sobre drivers (salário médio, densidade populacional,
  desemprego, feriados, promoções), a única variável adicional que disseram
  possuir é a **quantidade de consultoras cadastradas por zoneamento** — com
  a ressalva de que um zoneamento pode conter mais de um setor.
- **Não usam Nielsen nem Scanntech.**
- **Não há calendário de promoções.** As promoções ocorrem "conforme
  necessidade do negócio, com a abertura do ciclo"; a orientação deles foi
  "se basear nas sazonalidades". Existem "ciclos de estratégia", não
  compartilhados.

Isso limita bastante o espaço de features externas e é consistente com o
resultado do experimento `compare_forecasters`, em que o histórico do próprio
setor (`naive:mean`) venceu todos os modelos.

## Datas de abertura dentro do mesmo bloco/sub-bloco

Observamos setores no mesmo bloco/sub-bloco com aberturas defasadas em ~1
dia. Resposta: **de modo geral as datas deveriam ser iguais** dentro do mesmo
bloco e sub-bloco. As diferenças vêm da coluna "TO BE" (simulações que o
próprio Daniel faz) ou de exceções, como um ciclo aberto retroativamente.

## Arquivos entregues (pasta `dados/reais/`, espelho do Drive)

| arquivo | conteúdo | uso |
|---|---|---|
| `Unifesp_Demanda.csv` | v1: 200.000 linhas, 01/12/2025 a 08/09/2026, **sem** rota/cidade/estado | superada |
| `Unifesp_Demanda_v2.xlsx` | v2: 400.000 linhas, 01/12/2024 a 17/09/2026, **com** rota, cidade e estado | **fonte oficial de demanda** |
| `Simulador Bloco e Subbloco_v2.xlsx` | aba `Calendário FV` (12.022 linhas = 633 setores × 19 ciclos, com bloco, sub-bloco, abertura, fechamento) | fonte de bloco/sub-bloco e datas |
| `Bases_setores_cd.xlsx` | abas `Setor vs Zoneamento` e `GI` (Gestão de Itinerário) | malha logística e lead time |
| `Calendario_ciclos.xlsx` | calendário por setor 2024–2026, **sem** bloco/sub-bloco | histórico estendido |
| `Dados a partir das bases de demanda/` | séries agregadas e gráficos produzidos pelo próprio time | derivados, não fonte |

Duas advertências explícitas do cliente:

1. **A aba "Base Demanda" do simulador deve ser ignorada** — extração aleatória
   feita só para manter as fórmulas do Excel ativas. (São 294 MB de XML; ignorar
   também economiza bastante trabalho.)
2. **O histórico estendido foi descrito como "dados aleatórios a partir de
   Dezembro/24"** — ver a verificação abaixo, que muda a leitura dessa frase.

## O que a verificação dos dados mostrou (22/09/2026)

### "Dados aleatórios" significa *amostra*, não *números inventados*

Esta era a dúvida mais cara do projeto, porque decidia se o segundo ano de
histórico pode ou não ser usado. Três testes sobre `Unifesp_Demanda_v2`,
comparando o período novo (a partir de 01/12/2025, que o cliente sempre tratou
como real) com o período antigo (dez/2024 a nov/2025, o "aleatório"):

| teste | período antigo | período novo |
|---|---|---|
| correlação pedidos × itens | 0,814 | 0,842 |
| correlação volumes × itens | 0,951 | 0,955 |
| % de registros em sábado / domingo | 11,4 / 10,5 | 11,4 / 10,4 |
| variância entre setores | 10,1% | 11,3% |

E o teste decisivo: para os 643 setores com pelo menos 20 registros nos dois
períodos, **a média do setor no período antigo correlaciona 0,858 com a média
do mesmo setor no período novo** (Spearman 0,931). Números gerados
aleatoriamente não preservam a ordenação de tamanho dos setores ao longo de um
ano. Os totais mensais também repetem a mesma forma nos dois anos (jan/fev
baixos, mar/abr no pico, ago baixo).

**Leitura:** "aleatório" aqui quer dizer *amostragem aleatória de pedidos
reais* — coerente com a descrição original da base como "uma amostra dos
pedidos". O histórico de 2024–2025 é utilizável, e a recomendação de
`experiments/compare_forecasters` de revisitar sazonalidade com um segundo ano
**está liberada**.

### A base é uma amostra, e isso quebra a comparação com a capacidade

Somando os itens por CD e por dia na v2 e comparando com os limites informados:

| CD | capacidade (itens/dia) | pico observado | mediana | pico / capacidade |
|---|---|---|---|---|
| 2700 | 350.000 | 80.672 | 25.375 | 23,0% |
| 2800 | 44.000 | 63.968 | 10.734 | **145,4%** |
| 5100 | 1.000.000 | 191.873 | 55.421 | 19,2% |
| 5300 | 120.000 | 21.200 | 3.482 | 17,7% |
| 5400 | 350.000 | 37.816 | 8.906 | 10,8% |
| 5500 | 140.000 | 38.401 | 7.060 | 27,4% |
| 5600 | 200.000 | 63.051 | 12.372 | 31,5% |
| 5700 | 1.200.000 | 207.268 | 44.505 | 17,3% |
| 5800 | 400.000 | 92.966 | 23.026 | 23,2% |
| 3000 | **não informada** | 73.176 | 3.054 | — |

A demanda observada roda a 10–30% da capacidade informada, porque é uma
amostra: os limites estão em itens reais por dia e a base não está. **A
restrição de capacidade não pode ser calibrada contra estes números em valor
absoluto** sem conhecer a taxa de amostragem. O CD 2800 estourando 145% do
próprio limite é o sintoma mais claro de que as duas escalas não conversam.
O CD 3000 aparece na demanda e não tem limite informado (ele não está em
`data/base_tratada.csv`, que tem só os nove CDs com capacidade).

### A rota resolve o mapeamento setor → CD

A aba `GI` tem 8.909 linhas com `CENTRO` (o CD), `UF`, `ZONEAMENTO PARTIDA`,
`NOME CIDADE`, `ROTA`, `PRAZO DE TRP` e `PRAZO TOTAL`. Duas descobertas:

- `ZONEAMENTO PARTIDA` só tem **9 valores distintos** — é o zoneamento de
  *origem*, ou seja, do próprio CD. Não serve para localizar o setor.
- **`ROTA` determina o CD:** das 564 rotas do GI, 563 apontam para um único
  centro. Das 711 rotas da nossa base, 562 estão no GI.

Então o caminho de junção correto é **setor → rota → CD**, e ele já traz o lead
time junto (`PRAZO TOTAL`: mediana 7 dias, mínimo 2, máximo 20). A aba
`Setor vs Zoneamento` (53.897 linhas, 620 dos nossos 633 setores) é
**muitos-para-muitos**: um setor cobre dezenas de zoneamentos de CEP, o que é
esperado, já que a malha comercial não segue a logística.

Isso explica por que um setor aparece sob vários CDs na base (204 setores em 1
CD, 272 em 2, 115 em 3, 37 em 4 e 5 em 5): a divisão vem da geografia das
rotas, **não é decisão do otimizador**. A aproximação atual de
`compare_forecasters`, de atribuir cada setor ao seu CD dominante, pode agora
ser substituída pela divisão real por rota.

## Dúvidas ainda em aberto

1. **Qual é a taxa de amostragem da base?** É a pergunta que bloqueia a
   calibração da restrição de capacidade. Se soubermos que a base é, digamos,
   20% dos pedidos, os limites passam a ser comparáveis. Sem isso, resta
   modelar capacidade em termos relativos (equilibrar carga) em vez de respeitar
   tetos absolutos — o que enfraquece bastante a proposta.
2. **Por que o CD 2800 ultrapassa o próprio limite mesmo numa amostra?** Ou o
   limite de 44k está errado, ou a amostragem não é uniforme entre CDs.
3. **Qual a capacidade do CD 3000?** Ele aparece na demanda e não na lista.
4. **Os limites são teto rígido ou target médio?** A resposta diz "variável por
   dia (...) porém a capacitação depende da demanda do período", e ao mesmo
   tempo que o time é fixo.
5. **Como a demanda de um ciclo se distribui pelos dias?** Temos `data_pedido`,
   então dá para medir em vez de perguntar — mas vale confirmar com o cliente se
   a curva observada (concentrada no fechamento) é a que a operação usa para
   planejar.
6. **As 149 rotas da nossa base que não estão no GI** ficam sem lead time e sem
   CD derivado. Vale checar se são rotas descontinuadas antes de perguntar.
7. **O calendário de 2024–2025 não tem bloco/sub-bloco**, então não dá para
   reconstruir como a política atual se comportou naquele período — o que limita
   usar o histórico estendido como linha de base de comparação, ainda que ele
   sirva para previsão.
