using JuMP, Gurobi, CSV, DataFrames

# ==========================================
# CONFIGURAÇÕES INICIAIS E LEITURA DE DADOS
# ==========================================
pasta_instancias = "C:/Users/Cliente/Desktop/Projeto IPO/Novas_intancias/Instancias_Semanais_Gargalo"
nome_instancia = "N50_T12sem_Sat95" # Testando direto com 500 SKUs
buckets_por_semana = 42 # Granularidade extrema (4 horas) que estourou a RAM antes

println("Iniciando Matheurística R&F para: $nome_instancia com $buckets_por_semana buckets/semana")

df_prod  = CSV.read(joinpath(pasta_instancias, nome_instancia, "prod.csv"), DataFrame)
df_dem   = CSV.read(joinpath(pasta_instancias, nome_instancia, "dem.csv"), DataFrame)
df_setup = CSV.read(joinpath(pasta_instancias, nome_instancia, "setup.csv"), DataFrame)

produtos = Int.(df_prod.id)
produtos_com_zero = [0; produtos] 
periodos = Int.(sort(unique(df_dem.periodo))) 
N = length(produtos)
T_total = length(periodos)

d = Dict((row.item, row.periodo) => row.demanda for row in eachrow(df_dem))
Cap = Dict(row.periodo => row.capacidade for row in eachrow(df_dem))
h = Dict(df_prod.id[i] => df_prod.custo_estq[i] for i in 1:N)
c = Dict(df_prod.id[i] => df_prod.custo_prod[i] for i in 1:N)
a = Dict(df_prod.id[i] => df_prod.tempo_prod[i] for i in 1:N)
tempo_setup = Dict((row.de, row.para) => row.tempo for row in eachrow(df_setup))    

# --- MAPEAMENTO DA ESPARSIDADE ---
ativos_na_semana = Dict{Int, Vector{Int}}()
for t in periodos
    itens_com_demanda = unique(df_dem[df_dem.periodo .== t, :item])
    ativos_na_semana[t] = [0; itens_com_demanda]
end
Φ_0 = Dict(p => (p == 0 ? 1 : 0) for p in produtos_com_zero) 

# --- ESTRUTURAS TEMPORAIS ---
duracao_bucket_min = 10080.0 / buckets_por_semana
total_buckets = T_total * buckets_por_semana
S = 1:total_buckets 
S_t = Dict(periodos[idx] => ((idx-1)*buckets_por_semana + 1):(idx*buckets_por_semana) for idx in 1:T_total)
t_de_s = Dict(s => periodos[idx] for idx in 1:T_total for s in S_t[periodos[idx]])

# ==========================================
# CONSTRUÇÃO DO MODELO GLOBAL (S_relax)
# ==========================================
model = Model(Gurobi.Optimizer)
set_attribute(model, "OutputFlag", 1) # Desliga o log detalhado para não poluir o terminal no R&F

# Variáveis Contínuas (Densas)
@variable(model, I[produtos, 0:total_buckets] >= 0)    
@variable(model, y[produtos, S] >= 0)           
@variable(model, x[i=produtos, s=S; i in ativos_na_semana[t_de_s[s]]] >= 0)                  

# [CRÍTICO PARA O R&F] Variáveis de Setup nascem RELAXADAS (Contínuas entre 0 e 1)
# Todo o horizonte S começa implicitamente como S_relax
@variable(model, 0 <= Φ[i=produtos_com_zero, s=S; i in ativos_na_semana[t_de_s[s]]] <= 1)                  
@variable(model, 0 <= η[i=produtos_com_zero, j=produtos_com_zero, s=S; 
                  i in ativos_na_semana[t_de_s[s]] && j in ativos_na_semana[t_de_s[s]]] <= 1)        

# Funções Auxiliares (Esparsidade)
_x(i, s) = haskey(x, (i, s)) ? x[i, s] : 0.0
_Φ(i, s) = haskey(Φ, (i, s)) ? Φ[i, s] : 0.0
_η(i, j, s) = haskey(η, (i, j, s)) ? η[i, j, s] : 0.0

for p in produtos
    fix(I[p, 0], 0.0; force=true)
end

@objective(model, Min, sum(c[i] * _x(i, s) for i in produtos for s in S) + 
                       sum(h[i] * I[i, s] for i in produtos for s in S))

# Restrições Estruturais (Mesmas do modelo exato)
@constraint(model, cap_semanal[t in periodos], 
    sum(_x(i, s) * a[i] for i in produtos for s in S_t[t]) + 
    sum(_η(i, j, s) * tempo_setup[i, j] for i in ativos_na_semana[t] for j in ativos_na_semana[t] for s in S_t[t]) <= Cap[t])

@constraint(model, cap_fisica_bucket[s in S], 
    sum(_x(i, s) * a[i] for i in produtos) + 
    sum(_η(i, j, s) * tempo_setup[i, j] for i in ativos_na_semana[t_de_s[s]] for j in ativos_na_semana[t_de_s[s]]) <= duracao_bucket_min * sum(_Φ(i, s) for i in produtos_com_zero))

@constraint(model, balanco[i in produtos, s in S], I[i, s] == I[i, s-1] + _x(i, s) - y[i, s])

for s in S
    t_atual = t_de_s[s]
    for i in ativos_na_semana[t_atual]
        for j in ativos_na_semana[t_atual]
            estado_anterior = (s == 1) ? Φ_0[i] : _Φ(i, s-1)
            @constraint(model, _η(i, j, s) >= estado_anterior + _Φ(j, s) - 1)
        end
    end
end

@constraint(model, config_unica[s in S], sum(_Φ(i, s) for i in ativos_na_semana[t_de_s[s]]) == 1)
@constraint(model, atendimento_total[i in produtos, t in periodos], sum(y[i, s] for s in S_t[t]) == get(d, (i, t), 0.0))
@constraint(model, limite_saida[i in produtos, s in S], y[i, s] <= (s == 1 ? 0.0 : I[i, s-1]))

# ==========================================
# EXECUÇÃO DO RELAX-AND-FIX (Fase 1)
# ==========================================

# W = buckets_por_semana # Janela de otimização de 1 semana
# println("\n>>> Iniciando Iterações R&F (Janela W = $W)")

W = 6
Passo = 3

tempo_inicio_rf = time()
inviavel_rf = false

for s_start in 1:Passo:total_buckets
    # A janela que o Gurobi vai otimizar (S_int)
    s_end = min(s_start + W - 1, total_buckets)
    S_int = s_start:s_end
    
    # A sub-janela que nós vamos REALMENTE CONGELAR (S_fix_step)
    s_fix_end = min(s_start + Passo - 1, total_buckets)
    S_fix_step = s_start:s_fix_end
    
    println(" -> Otimizando: [$s_start a $s_end] | Fixando apenas: [$s_start a $s_fix_end]")


    # 1. Impor Integridade para as variáveis do S_int atual
    for s in S_int
        t_atual = t_de_s[s]
        for i in ativos_na_semana[t_atual]
            set_binary(Φ[i, s])
            for j in ativos_na_semana[t_atual]
                set_binary(η[i, j, s])
            end
        end
    end
    
    if s_start == 1
        # A 1ª janela carrega o LP inteiro de 12 semanas. Precisa de muito tempo!
        println("   [Aviso] Primeira iteração: Concedendo 300 segundos para o Nó Raiz...")
        set_attribute(model, "TimeLimit", 300) 
    else
        # As janelas seguintes são leves pois o passado já está fixado
        set_attribute(model, "TimeLimit", 60) 
    end
    
    set_attribute(model, "MIPGap", 0.005)   # Aceita 0.5% de erro
    set_attribute(model, "MIPFocus", 1)     # Foca em viabilidade~;
    set_attribute(model, "NodeLimit", 100)  # Evita degenerescência   
    
    # 2. Resolvemos o subproblema
    optimize!(model)
    
    status_rf = termination_status(model)

    if !has_values(model)
        println(" [!] FALHA NA JANELA: O solver parou com status $status_rf e NÃO encontrou nenhuma solução factível.")
        println(" -> Motivo: O tempo limite esgotou antes de fechar os setups, ou o cenário é fisicamente inviável.")
        global inviavel_rf = true
        break
    end

    # if termination_status(model) == MOI.INFEASIBLE
    #     println(" [!] ERRO: Modelo inviável na janela [$s_start a $s_end].")
    #     global inviavel_rf = true
    #     break
    # end
    
    # 3. Congelar/Fixar variáveis (O S_int atual se torna o S_fix do futuro)
    valores_phi = Dict()
    valores_eta = Dict()
    
    for s in S_fix_step
        t_atual = t_de_s[s]
        for i in ativos_na_semana[t_atual]
            valores_phi[(i, s)] = round(value(Φ[i, s]))
            for j in ativos_na_semana[t_atual]
                valores_eta[(i, j, s)] = round(value(η[i, j, s]))
            end
        end
    end
    
    # 4. Fixar as variáveis no modelo usando os valores que salvamos
    for s in S_fix_step
        t_atual = t_de_s[s]
        for i in ativos_na_semana[t_atual]
            fix(Φ[i, s], valores_phi[(i, s)]; force=true)
            for j in ativos_na_semana[t_atual]
                fix(η[i, j, s], valores_eta[(i, j, s)]; force=true)
            end
        end
    end
end

tempo_total_rf = time() - tempo_inicio_rf

# ==========================================
# CÁLCULO FINAL DA FUNÇÃO OBJETIVO
# ==========================================
set_attribute(model, "TimeLimit", 10) 
optimize!(model)


# ==========================================
# RESULTADOS DA FASE 1
# ==========================================
# ==========================================
# FASE 2: FIX-AND-OPTIMIZE (Melhoria Local)
# ==========================================
# Só executa o F&O se o R&F tiver encontrado uma solução factível inicial

global custo_total = 0.0

if !inviavel_rf
    println("\n>>> Iniciando Fase 2: Fix-and-Optimize (Refinamento da Solução)")

    W_fo = 6       # Descongela 1 dia inteiro para reavaliar
    Passo_fo = 3   # Desliza 3 buckets por vez para manter a sobreposição
    
    tempo_inicio_fo = time()

    for s_start in 1:Passo_fo:total_buckets
        # 1. Definir a Janela que será Descongelada (S_int_fo)
        s_end = min(s_start + W_fo - 1, total_buckets)
        S_int_fo = s_start:s_end
        
        println(" -> F&O Reotimizando a janela: [$s_start a $s_end]")
        
        # 2. Descongelar (unfix) as variáveis desta janela para o solver poder alterá-las
        for s in S_int_fo
            t_atual = t_de_s[s]
            for i in ativos_na_semana[t_atual]
                if is_fixed(Φ[i, s])
                    unfix(Φ[i, s])
                    set_binary(Φ[i, s]) # Garante que volta para a árvore de busca como binária
                end
                for j in ativos_na_semana[t_atual]
                    if is_fixed(η[i, j, s])
                        unfix(η[i, j, s])
                        set_binary(η[i, j, s])
                    end
                end
            end
        end
        
        # 3. Reotimizar (Dar tempo ao solver para procurar uma rota mais barata)
        set_attribute(model, "TimeLimit", 60) # 1 minuto para polir cada dia
        set_attribute(model, "MIPGap", 0.001) # Busca fina
        
        optimize!(model)
        
        # 4. Extrair os valores otimizados (podem ser os mesmos ou melhores)
        valores_phi_fo = Dict()
        valores_eta_fo = Dict()
        
        for s in S_int_fo
            t_atual = t_de_s[s]
            for i in ativos_na_semana[t_atual]
                valores_phi_fo[(i, s)] = round(value(Φ[i, s]))
                for j in ativos_na_semana[t_atual]
                    valores_eta_fo[(i, j, s)] = round(value(η[i, j, s]))
                end
            end
        end
        
        # 5. Congelar novamente as variáveis com a solução polida
        for s in S_int_fo
            t_atual = t_de_s[s]
            for i in ativos_na_semana[t_atual]
                fix(Φ[i, s], valores_phi_fo[(i, s)]; force=true)
                for j in ativos_na_semana[t_atual]
                    fix(η[i, j, s], valores_eta_fo[(i, j, s)]; force=true)
                end
            end
        end
    end
    tempo_total_fo = time() - tempo_inicio_fo

    # ==========================================
    # CÁLCULO FINAL GLOBAL
    # ==========================================
    set_attribute(model, "TimeLimit", 10) 
    optimize!(model)
    
    
    global custo_final_fo = round(objective_value(model), digits=2)
    println("\n==========================================")
    println(">>> MATHEURÍSTICA COMPLETA FINALIZADA <<<")
    println("Tempo R&F (Fase 1): $(round(tempo_total_rf, digits=2)) segundos")
    println("Custo R&F (Fase 1): $custo_total")
    println("Tempo F&O (Fase 2): $(round(tempo_total_fo, digits=2)) segundos")
    println("Custo F&O (Fase 2): $custo_final_fo")
    
    if custo_final_fo < custo_total
        println("=> SUCESSO: O Fix-and-Optimize gerou uma economia de $(round(custo_total - custo_final_fo, digits=2)) !")
    else
        println("=> O R&F já tinha encontrado uma solução localmente ótima para esta partição de janelas.")
    end
    println("==========================================")
end


df_producao = DataFrame(
    Semana = Int[], 
    Bucket = Int[], 
    SKU = Int[], 
    Setup_Ativo = Int[], 
    Volume_Produzido = Float64[], 
    Estoque_Final = Float64[]
)

for s in S
    t_atual = t_de_s[s]
    for i in ativos_na_semana[t_atual]
        # Usamos as funções auxiliares _Φ e _x para evitar erros de variáveis que não existem (esparsidade)
        val_phi = round(_Φ(i, s))
        val_x   = round(_x(i, s), digits=2)
        val_I   = (i != 0) ? round(value(I[i, s]), digits=2) : 0.0 # O item 0 (dummy) não tem estoque
        
        # Filtro: Guardar apenas se a máquina está configurada para o item, ou se há produção/estoque
        if val_phi > 0 || val_x > 0 || val_I > 0
            push!(df_producao, (t_atual, s, i, Int(val_phi), val_x, val_I))
        end
    end
end

CSV.write("Plano_Producao_Final.csv", df_producao)
println(" -> Arquivo 'Plano_Producao_Final.csv' gerado com sucesso!")

# 2. Tabela de Transições de Setup (A variável Eta)
df_setups = DataFrame(
    Semana = Int[], 
    Bucket = Int[], 
    De_SKU = Int[], 
    Para_SKU = Int[]
)

for s in S
    t_atual = t_de_s[s]
    for i in ativos_na_semana[t_atual]
        for j in ativos_na_semana[t_atual]
            val_eta = round(_η(i, j, s))
            if val_eta == 1.0
                push!(df_setups, (t_atual, s, i, j))
            end
        end
    end
end

CSV.write("Transicoes_Setup_Final.csv", df_setups)
println(" -> Arquivo 'Transicoes_Setup_Final.csv' gerado com sucesso!")
println("==========================================")