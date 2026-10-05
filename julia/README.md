# Julia (matheurísticas: Relax-and-Fix / Fix-and-Optimize)

Referência em Julia/JuMP de duas técnicas de matheurística — **Relax-and-Fix**
e **Fix-and-Optimize** — pensadas como uma terceira família de solvers ao lado
de `src/solver/mip/` (Pyomo, exato) e `src/solver/heuristics/` (Simulated
Annealing). Julia é um ecossistema separado (gerenciador de pacotes próprio),
por isso mora numa pasta própria na raiz do repo — mesmo padrão do `formal/`
(Lean 4).

## Estado atual: referência, ainda não adaptado

`relax_and_fix_and_optimize.jl` **não resolve o problema de atribuição
setor -> Bloco/Subloco** deste repositório. Ele resolve um problema diferente
de dimensionamento de lotes de produção (SKUs, capacidade semanal, troca de
configuração/setup entre produtos — variáveis `x` = produção, `I` = estoque,
`Φ` = setup ativo, `η` = transição de setup). Foi trazido como ponto de
partida pela técnica (a estrutura de janelas do Relax-and-Fix / Fix-and-
Optimize), não pelo modelo em si.

**O que falta para se tornar um solver de verdade deste repo:** reescrever as
seções `@variable`/`@constraint`/`@objective` para usar a formulação real
(`docs/` e `src/solver/mip/block_assignment.py`) — conjuntos `S`, `D`, `A`,
`C`; variáveis `x[s,d]`, `z_max`, `z_min`; objetivo `min z_max - z_min`;
restrições de atribuição única, capacidade por CD e churn — mantendo a lógica
de janelas deslizantes (congelar/descongelar blocos de variáveis binárias) que
já está escrita.

Também tem um caminho absoluto específico da máquina de origem
(`pasta_instancias = "C:/Users/Cliente/Desktop/..."`) que precisa virar
parâmetro/config antes de rodar em qualquer outro lugar.

## O que são as duas técnicas, resumidamente

- **Relax-and-Fix:** resolve o MIP em janelas de tempo deslizantes. Dentro da
  janela atual, as variáveis binárias (setup) são exigidas inteiras; fora
  dela, ficam relaxadas (contínuas entre 0 e 1). Resolve, congela um pedaço da
  janela com os valores encontrados, desliza a janela adiante, repete. Reduz o
  tamanho do problema binário resolvido de cada vez.
- **Fix-and-Optimize:** depois que o Relax-and-Fix já tem uma solução inteira
  completa (viável), passa de novo pelo horizonte descongelando uma janela por
  vez (permitindo o solver reotimizar aquele pedaço), fixa de novo com o
  resultado, e desliza adiante. É uma busca local em cima da solução inicial.

## Como rodar (quando adaptado)

Requer Julia, o pacote `JuMP.jl` e um solver (`Gurobi.jl` no script atual —
precisa de licença; poderia trocar por um solver livre registrado no JuMP,
mesmo espírito do `solver` configurável do `src/solver/mip/`).

```bash
julia relax_and_fix_and_optimize.jl
```
