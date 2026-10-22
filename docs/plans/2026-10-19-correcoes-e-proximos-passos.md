# Correções sugeridas e próximos passos (SA, MIP e calibração)

Documento de decisão (2026-10-19). Nada aqui está implementado. Consolida o que ficou em
aberto depois da rodada descrita em
`docs/agent-log/2026-10-19-sa-pico-desvio-solvers-escala.md`.

Cada item traz o problema, onde está, a sugestão e uma estimativa de prioridade:

- **Alta:** pode produzir resultado errado ou bloquear a próxima conclusão.
- **Média:** atrapalha a reprodutibilidade ou a interpretação.
- **Baixa:** melhoria.

## Resumo

| # | Item | Tipo | Prioridade |
|---|---|---|---|
| 1 | Tolerância do MIP não é passada nem traduzida por solver | correção | Alta |
| 2 | Troca silenciosa de Gurobi por HiGHS | correção | Alta |
| 3 | Limite de memória por `RLIMIT_AS` | correção | Alta |
| 4 | Previsão automática em `full.yaml` não reproduz as instâncias | correção | Média |
| 5 | ARIMA falha em silêncio | investigação | Média |
| 6 | Retomada de CSV com colunas diferentes | correção | Média |
| 7 | Extras de instalação e testes que falham sem eles | correção | Média |
| 8 | Nomes únicos para arquivos de teste | convenção | Baixa |
| 9 | Instância apertada em escala (633 ou 800 setores) | experimento | Alta |
| 10 | Orçamento do SA proporcional ao número de setores | experimento | Média |
| 11 | ρ e T0 normalizados pela escala da energia | experimento | Média |
| 12 | Avaliar a energia de pico e desvio pelo critério certo | experimento | Média |
| 13 | Mais de uma instância por tamanho e aperto | experimento | Baixa |
| 14 | Outros esquemas de resfriamento | experimento | Baixa |
| 15 | Taxa de amostragem da base de demanda × capacidade | dado do cliente | Média |
| 16 | Pendências de documentação | documentação | Baixa |

Ordem sugerida: 9 e 15 primeiro, porque decidem se o SA é necessário em escala; depois 1–3
(afetam qualquer resultado do MIP); o resto conforme a necessidade.

## Correções no código

### 1. Tolerância do MIP não é passada nem traduzida por solver (Alta)

- **Onde:** `src/solver/mip/block_assignment.py:55` (`_solve_options`).
- **Problema:**
  - para o Gurobi só vão `MemLimit` e `TimeLimit`, e para o HiGHS nada; cada solver para com
    a tolerância padrão dele;
  - os nomes da opção diferem (`MIPGap` no Gurobi, `mip_rel_gap` no HiGHS), e uma tolerância
    com o nome errado é ignorada sem erro.
- **Efeito medido:** com tolerância de 1%, o Gurobi parou 0,26% acima do ótimo em base40 e
  0,04% em base80; o HiGHS devolveu o ótimo.
- **Sugestão:**
  - um parâmetro `mip_rel_gap` em `solve_block_assignment` e `run_block_assignment_mip`,
    traduzido por solver;
  - a tradução já existe em `src/calibration/mip.py` (`solve_kwargs`) e pode ser movida para
    o módulo do MIP e reaproveitada pelos dois caminhos;
  - um teste que confira a opção efetivamente entregue a cada solver.

### 2. Troca silenciosa de Gurobi por HiGHS (Alta)

- **Onde:** `src/solver/mip/block_assignment.py:96` (`_resolve_solver`), usado em
  `solve_block_assignment` (linha 113).
- **Problema:** sem licença ou instalação do Gurobi, o código usa o HiGHS e só registra um
  aviso no log. `run_block_assignment_mip` grava o solver efetivo no resultado, mas
  `solve_block_assignment` não: quem chamou não sabe qual solver rodou.
- **Sugestão:**
  - devolver também o solver efetivo, ou
  - aceitar `allow_fallback: bool = True`, que levante erro quando `False`;
  - em experimentos que comparam solvers, usar `allow_fallback=False`.

### 3. Limite de memória por `RLIMIT_AS` (Alta)

- **Onde:** `src/solver/mip/block_assignment.py:28` (`_memory_limit`), com
  `DEFAULT_MEMORY_LIMIT_GB = 8` (linha 22).
- **Problema:**
  - `RLIMIT_AS` limita o espaço de endereços virtual do processo inteiro, não a memória
    usada; NumPy/OpenBLAS e o Gurobi reservam muito espaço virtual por thread sem usá-lo;
  - em instâncias grandes isso pode virar um `MemoryError` falso muito antes de a RAM
    acabar;
  - numa máquina com menos de 8 GB (a de testes tem 7,1 GB), o limite não protege nada.
- **Contexto:** pelo caminho da calibração, que não passa por esse limite, o Gurobi resolveu
  800 setores com 1,1 GB. Se alguém observar falta de memória pelo `src/main.py`, este é o
  primeiro suspeito.
- **Sugestão:**
  - para o Gurobi, usar só os limites nativos (`MemLimit`/`SoftMemLimit`), mais `Threads`,
    `NodefileStart` e `NodefileDir` em instâncias grandes;
  - para o HiGHS, um limite de memória residente medido, ou nenhum.

### 4. Previsão automática em `full.yaml` não reproduz as instâncias (Média)

- **Onde:** `configs/experiments/sa_calibration/full.yaml` (`level_strategy: null`).
- **Problema:**
  - a seleção automática depende dos pacotes instalados: sem `statsmodels`, ela escolhia
    `naive_last` + `plain_average`; com ele, base40 passa a escolher ARIMA;
  - a mesma config gera instâncias diferentes conforme o ambiente, e não reproduz mais a
    calibração de setembro.
- **Sugestão:** fixar `level_strategy: naive_last` e `shape_strategy: plain_average`, como nas
  configs novas. Conferido: isso reproduz exatamente o ótimo do MIP e as trajetórias do SA de
  setembro.

### 5. ARIMA falha em silêncio (Média)

- **Onde:** `src/forecasting/scoring.py:140`.
- **Problema:** a suíte mostra `UserWarning: Skipping level strategy 'arima': too many indices
  for array`. Em alguma entrada o ARIMA quebra e a estratégia é descartada sem que a seleção
  avise quem chamou.
- **Sugestão:** reproduzir com a entrada que dispara o aviso, corrigir o formato do array e
  decidir se a falha deve virar erro ou ser registrada no resultado da seleção.

### 6. Retomada de CSV com colunas diferentes (Média)

- **Onde:** `src/calibration/runner.py` (`run_record`) e a retomada em `src/calibration/io.py`.
- **Problema:** cada execução agora grava 4 colunas a mais (`energy_mode`, `w_peak`, `w_std`,
  `w_range`). Retomar uma calibração antiga com o código novo acrescenta linhas com mais
  colunas que o cabeçalho, e o CSV fica desalinhado sem erro.
- **Sugestão:** ao retomar, comparar o cabeçalho existente com as chaves do registro e falhar
  com mensagem clara, ou reescrever o CSV com as colunas novas preenchidas com o padrão.

### 7. Extras de instalação e testes que falham sem eles (Média)

- **Onde:** `pyproject.toml` / `uv.lock` (extra `forecast`), README e os testes que importam
  `lightgbm` e `statsmodels`.
- **Problema:** sem `uv sync --extra forecast`, cerca de 30 testes falham por
  `ModuleNotFoundError`, o que parece regressão. O `optuna` não está em nenhum extra; as
  fases de TPE rodam com `uv run --with optuna`.
- **Sugestão:**
  - documentar `uv sync --all-extras` no README;
  - marcar esses testes com `pytest.importorskip`;
  - avaliar um extra `calibration` com `optuna` e `matplotlib`.

### 8. Nomes únicos para arquivos de teste (Baixa)

- **Problema:** os testes não têm `__init__.py`, então o pytest exige nomes de arquivo únicos
  no repositório inteiro. Um nome repetido só quebra a coleta na suíte completa, não ao rodar
  uma pasta. Aconteceu nesta rodada e foi corrigido renomeando o arquivo novo.
- **Sugestão:** registrar a regra em `docs/conventions/testing.md` ou adicionar `__init__.py`
  às pastas de teste.

## Próximos experimentos

### 9. Instância apertada em escala (Alta)

- **Por quê:**
  - a instância de 800 setores gerada usa no máximo 3,7% da capacidade dos CDs, e o Gurobi
    fecha o ótimo em 20 s;
  - ela não responde se o MIP continua fechando quando a capacidade restringe, que é a
    pergunta que justificaria o SA em escala.
- **Como:**
  - usar 633 setores, o número real confirmado pelo cliente (ou 800, para manter a
    comparação);
  - achar o menor `capacity_multiplier` viável com a bissecção existente
    (`min_feasible_multiplier`, até 9 sondagens de 60 s);
  - usar 1,15 × esse mínimo, como em tight40;
  - rodar o Gurobi com 1 h de limite e os parâmetros de memória, e o SA com as duas
    configurações calibradas, 8 sementes, 200 mil e 1 milhão de iterações.
- **Custo:** de 15 min a 1,5 h. O Gurobi usa 2 threads; o SA pode usar de 2 a 4 processos se
  a máquina estiver em uso.
- **Decisão que depende disso:**
  - se o MIP fechar também no caso apertado, o SA não se justifica nessa escala e a
    recomendação é usar o MIP;
  - se não fechar, o gap passa a ser medido contra o limite inferior, e a calibração precisa
    ser revalidada nesse regime.

### 10. Orçamento do SA proporcional ao número de setores (Média)

- **Por quê:** com 200 mil iterações, o gap vai de 1–3% (40 e 80 setores) para 8–11% (800).
  Com 1 milhão, cai para 4–5%. Há cerca de 17 iterações por par (setor, combinação) em 800
  setores, contra 330 em 40.
- **Como:** fixar iterações por setor (ex.: de 2.500 a 5.000 × |S|) e medir o gap em 40, 80,
  400 e 800 setores. Com a avaliação incremental, o custo por iteração quase não cresce com
  |S|.

### 11. ρ e T0 normalizados pela escala da energia (Média)

- **Por quê:** ρ e T0 têm a unidade da energia (itens), e o valor absoluto não se transfere:
  - o ρ calibrado para a energia de amplitude (146) deixa a energia de pico e desvio inviável
    em metade das execuções;
  - na energia de amplitude, ρ ≥ 223 já era viável em 100% das execuções da triagem; na nova,
    nenhuma execução com ρ < 223 foi viável e a faixa de 223 a 3.440 ficou em 90%.
- **Como:** medir a mediana de |ΔE| num passeio aleatório curto no início de cada execução e
  expressar ρ e T0 como múltiplos dela. Na calibração de amplitude, ρ ≈ 2 × e T0 ≈ 3,5 × essa
  mediana. Recalibrar com os múltiplos e checar se eles valem nas duas energias e em 400/800
  setores.

### 12. Avaliar a energia de pico e desvio pelo critério certo (Média)

- **Por quê:** a energia nova foi calibrada e ranqueada pelo gap da amplitude, que ela não
  minimiza. O resultado (desvio 1–3% menor, pico igual ou pior, amplitude 2–3 pontos pior)
  não mostra se ela seria vantajosa sob um critério de pico.
- **Como:**
  - definir com o negócio a métrica que importa (pico, desvio ou amplitude);
  - criar uma referência exata para ela, como um MIP que minimize o pico médio por ciclo
    (mesmas restrições, outro objetivo);
  - ranquear o fine-tuning por essa métrica.
- **Até lá:** manter a energia de amplitude como padrão.

### 13. Mais de uma instância por tamanho e aperto (Baixa)

- **Por quê:** há uma única instância por combinação de tamanho e aperto, então a variação
  entre instâncias não foi medida.
- **Como:** 3 a 5 sementes do gerador por tamanho, revalidando só a configuração calibrada.

### 14. Outros esquemas de resfriamento (Baixa)

- **Por quê:** só o resfriamento geométrico com α derivado do orçamento foi testado.
- **Como:** resfriamento em patamares e adaptativo por taxa de aceitação. Exige um gancho de
  agenda de temperatura no SA.

## Dados do cliente

### 15. Taxa de amostragem da base de demanda × capacidade (Média)

- **Por quê:**
  - a base de demanda é uma amostra dos pedidos, e os CDs aparecem usando 10% a 30% da
    capacidade informada (o CD 2800 aparece em 145%) (`docs/context/natura-client-answers.md`);
  - sem a taxa de amostragem, não dá para saber se, em escala real, a capacidade restringe;
  - essa resposta decide se o regime relevante é o folgado (o MIP fecha em segundos) ou o
    apertado (item 9).
- **Como:** perguntar ao cliente a fração dos pedidos presente na base, ou o volume total
  diário real por CD.

## Documentação

### 16. Pendências de documentação (Baixa)

- **Nova energia no README do solver:** documentar `energy_mode` e os pesos em
  `src/solver/README.md` (ou no docstring do módulo do SA), com a recomendação de manter a
  energia de amplitude como padrão.
