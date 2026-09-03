# Plano de Implementação — Projeto de Machine Learning com o Indian Liver Patient Dataset (ILPD)

## Visão Geral do Projeto

**Objetivo:** Desenvolver um modelo de classificação binária capaz de prever se um paciente possui doença hepática, utilizando dados clínicos e demográficos do Indian Liver Patient Dataset (ILPD).

**Dataset:** Indian Liver Patient Dataset (ILPD)

| Característica | Valor |
|---|---|
| Total de registros | 583 |
| Features preditoras | 10 |
| Variável-alvo | `Dataset` (1 = doença hepática, 2 = sem doença hepática) |
| Variável categórica | `Gender` |
| Valores ausentes | `Albumin_and_Globulin_Ratio` |
| Classe 1 (doença) | 416 pacientes (71,4%) |
| Classe 2 (sem doença) | 167 pacientes (28,6%) |

**Classificação binária final (após pré-processamento):**
- `1` = paciente com doença hepática
- `0` = paciente sem doença hepática

**Nota sobre desbalanceamento:** A proporção entre as classes é de aproximadamente 2,5:1. Esse desbalanceamento deve ser considerado em todas as etapas do projeto, desde a escolha das métricas até as estratégias de treinamento.

## Estrutura do Projeto (Notebooks Separados)

Para manter a organização e modularidade, **cada fase do projeto deve ser implementada em um Jupyter Notebook separado**. A transição de dados entre as fases será feita salvando o estado (ex: DataFrames transformados, modelos, pipelines) em arquivos (como `.csv`, `.pkl` ou `.joblib`) que serão carregados no notebook da fase seguinte.

**Estrutura de arquivos:**
- `01_Fase1_Preparacao.ipynb`
- `02_Fase2_EDA.ipynb`
- `03_Fase3_Pre_processamento.ipynb`
- `04_Fase4_Divisao.ipynb`
- `05_Fase5_Pipeline.ipynb`
- `06_Fase6_Baseline.ipynb`
- `data/` (para salvar os datasets intermediários)
- `models/` (para salvar pipelines e modelos treinados)

---

## Fase 1 — Preparação e Entendimento do Dataset

### Objetivo
Carregar o dataset, compreender sua estrutura e documentar todas as características relevantes antes de iniciar qualquer análise ou modelagem.

### Tarefas

1. **Obtenção e carregamento do dataset**
   - Carregar o dataset utilizando `pandas.read_csv()`.
   - Verificar se o carregamento foi realizado corretamente (primeiras e últimas linhas).

2. **Identificação das colunas**
   - Listar todas as colunas do dataset e descrever o significado de cada uma:
     - `Age` — Idade do paciente.
     - `Gender` — Gênero do paciente (Male/Female).
     - `Total_Bilirubin` — Bilirrubina total.
     - `Direct_Bilirubin` — Bilirrubina direta.
     - `Alkaline_Phosphotase` — Fosfatase alcalina.
     - `Alamine_Aminotransferase` — SGPT/ALT (alanina aminotransferase).
     - `Aspartate_Aminotransferase` — SGOT/AST (aspartato aminotransferase).
     - `Total_Protiens` — Proteínas totais.
     - `Albumin` — Albumina.
     - `Albumin_and_Globulin_Ratio` — Razão albumina/globulina (A/G).
     - `Dataset` — Variável-alvo (1 = doença hepática, 2 = sem doença hepática).

3. **Tipos de dados**
   - Utilizar `df.dtypes` e `df.info()` para identificar os tipos de cada coluna.
   - Verificar se os tipos são compatíveis com o esperado (numéricos para features clínicas, categórico para `Gender`, inteiro para `Dataset`).

4. **Tamanho do dataset**
   - Confirmar o número de registros (583) e o número de colunas (11).
   - Registrar `df.shape`.

5. **Análise inicial da variável-alvo**
   - Utilizar `df['Dataset'].value_counts()` para confirmar a distribuição:
     - 416 registros com valor 1 (doença hepática).
     - 167 registros com valor 2 (sem doença hepática).
   - Calcular as proporções percentuais de cada classe.

6. **Identificação de valores ausentes**
   - Utilizar `df.isnull().sum()` para identificar colunas com valores ausentes.
   - Verificar que a coluna `Albumin_and_Globulin_Ratio` possui valores faltantes.
   - Registrar a quantidade exata de valores ausentes por coluna.

7. **Identificação de duplicatas**
   - Utilizar `df.duplicated().sum()` para verificar a existência de registros duplicados.
   - Se houver duplicatas, analisar se são duplicatas genuínas (pacientes diferentes com dados iguais) ou possíveis erros de entrada.
   - Documentar a decisão de manter ou remover duplicatas, com justificativa.

8. **Verificação de inconsistências**
   - Verificar se há valores negativos em colunas que não deveriam ter (ex.: idade, bilirrubina).
   - Verificar se `Gender` contém apenas valores esperados (Male/Female).
   - Verificar se `Dataset` contém apenas os valores 1 e 2.
   - Utilizar `df.describe()` para inspecionar estatísticas descritivas e identificar valores atípicos óbvios.

9. **Documentação das características do dataset**
   - Compilar um resumo com todas as informações coletadas nesta fase.
   - Registrar observações relevantes para orientar as fases seguintes.

### Critérios de Conclusão

- [ ] Dataset carregado com sucesso sem erros.
- [ ] Todas as colunas identificadas e descritas.
- [ ] Tipos de dados verificados e documentados.
- [ ] Dimensão do dataset confirmada (583 × 11).
- [ ] Distribuição da variável-alvo confirmada e registrada.
- [ ] Valores ausentes identificados e quantificados.
- [ ] Duplicatas verificadas e decisão documentada.
- [ ] Inconsistências verificadas e documentadas.
- [ ] Resumo das características do dataset registrado.

---

## Fase 2 — Análise Exploratória dos Dados (EDA)

### Objetivo
Compreender o comportamento de cada feature, identificar padrões, outliers e relações entre as variáveis e a variável-alvo. Cada análise deve contribuir para decisões de pré-processamento e modelagem.

### Análises e Visualizações

#### 2.1 Distribuição das classes
- **Visualização:** Gráfico de barras com a contagem de cada classe (`Dataset`).
- **Objetivo:** Confirmar visualmente o desbalanceamento entre pacientes com e sem doença hepática.
- **Pergunta:** Qual é a proporção exata entre as classes? O desbalanceamento é suficiente para justificar técnicas de balanceamento?

#### 2.2 Distribuição das idades
- **Visualização:** Histograma e/ou boxplot de `Age`, separado por classe.
- **Objetivo:** Verificar se a idade é um fator diferenciador entre pacientes com e sem doença hepática.
- **Perguntas:** Qual a faixa etária predominante no dataset? A distribuição de idade difere significativamente entre as duas classes?

#### 2.3 Distribuição por gênero
- **Visualização:** Gráfico de barras empilhadas ou agrupadas de `Gender` por classe.
- **Objetivo:** Verificar a composição de gênero do dataset e se há diferenças na proporção de doença hepática entre gêneros.
- **Perguntas:** O dataset é equilibrado em termos de gênero? A prevalência da doença hepática difere entre homens e mulheres?

#### 2.4 Bilirrubina total e bilirrubina direta
- **Visualização:** Histogramas, boxplots por classe e scatterplot entre `Total_Bilirubin` e `Direct_Bilirubin`.
- **Objetivo:** Analisar a relação entre os dois tipos de bilirrubina e identificar possíveis outliers. Verificar se valores elevados estão associados à presença de doença.
- **Perguntas:** Existe correlação forte entre as duas medidas? Pacientes com doença hepática tendem a apresentar valores mais elevados? Há outliers extremos?

#### 2.5 Fosfatase alcalina
- **Visualização:** Histograma e boxplot de `Alkaline_Phosphotase` por classe.
- **Objetivo:** Avaliar a distribuição da fosfatase alcalina e sua associação com doença hepática.
- **Perguntas:** A fosfatase alcalina apresenta valores significativamente diferentes entre as classes? Há outliers?

#### 2.6 SGPT/ALT (Alanina Aminotransferase)
- **Visualização:** Histograma e boxplot de `Alamine_Aminotransferase` por classe.
- **Objetivo:** Avaliar se essa enzima hepática é um indicador relevante para a classificação.
- **Perguntas:** Os valores de ALT são mais elevados nos pacientes com doença hepática? A distribuição é muito assimétrica?

#### 2.7 SGOT/AST (Aspartato Aminotransferase)
- **Visualização:** Histograma e boxplot de `Aspartate_Aminotransferase` por classe.
- **Objetivo:** Análise análoga à do SGPT/ALT, para verificar se o SGOT/AST também é discriminativo.
- **Perguntas:** Os valores de AST diferem entre as classes? A correlação entre ALT e AST pode indicar redundância?

#### 2.8 Proteínas totais
- **Visualização:** Histograma e boxplot de `Total_Protiens` por classe.
- **Objetivo:** Verificar se o nível de proteínas totais está associado ao diagnóstico de doença hepática.
- **Perguntas:** A distribuição difere entre pacientes com e sem doença? Há sobreposição significativa entre as classes?

#### 2.9 Albumina
- **Visualização:** Histograma e boxplot de `Albumin` por classe.
- **Objetivo:** Avaliar se níveis baixos de albumina estão associados à doença hepática.
- **Perguntas:** Pacientes com doença hepática apresentam níveis mais baixos de albumina? A albumina tem poder discriminativo?

#### 2.10 Razão albumina/globulina (A/G Ratio)
- **Visualização:** Histograma e boxplot de `Albumin_and_Globulin_Ratio` por classe.
- **Objetivo:** Analisar a razão A/G como indicador de doença, considerando que essa coluna possui valores ausentes.
- **Perguntas:** A razão A/G é menor em pacientes com doença hepática? A presença de valores ausentes está associada a alguma classe específica?

#### 2.11 Relações entre features e variável-alvo
- **Visualização:** Pairplot colorido pela variável-alvo (amostragem se necessário por questões de desempenho).
- **Objetivo:** Obter uma visão global das relações bivariadas entre as features, e identificar possíveis padrões de separabilidade.
- **Perguntas:** Existem pares de features que separam bem as classes? Alguma feature isoladamente é suficiente para a classificação?

#### 2.12 Possíveis outliers
- **Visualização:** Boxplots combinados de todas as features numéricas.
- **Objetivo:** Identificar valores extremos que possam distorcer a modelagem.
- **Perguntas:** Quais features possuem outliers? Os outliers são valores plausíveis ou erros de medição? Eles estão associados a uma classe específica?
- **Decisão:** Documentar se os outliers serão tratados (remoção, capping, transformação logarítmica) e justificar a escolha.

#### 2.13 Correlações entre variáveis
- **Visualização:** Heatmap da matriz de correlação de Pearson para as features numéricas.
- **Objetivo:** Identificar multicolinearidade entre features e possíveis redundâncias.
- **Perguntas:** Quais pares de features possuem correlação alta? A alta correlação entre `Total_Bilirubin` e `Direct_Bilirubin` sugere a remoção de uma delas? Há features com baixa correlação com a variável-alvo?

### Critérios de Conclusão

- [ ] Todas as análises listadas realizadas e documentadas.
- [ ] Visualizações geradas e salvas.
- [ ] Observações sobre outliers documentadas, com decisão justificada.
- [ ] Correlações entre variáveis analisadas.
- [ ] Resumo das descobertas registrado para orientar as decisões de pré-processamento e modelagem.

---

## Fase 3 — Pré-processamento

### Objetivo
Preparar os dados para a modelagem, tratando valores ausentes, codificando variáveis categóricas e transformando a variável-alvo.

### Tarefas

#### 3.1 Tratamento dos valores ausentes
- A coluna `Albumin_and_Globulin_Ratio` possui valores faltantes.
- **Estratégia planejada:** Imputação pela mediana da coluna, calculada exclusivamente a partir dos dados de treinamento.
- **Justificativa:** A mediana é robusta a outliers, que foram identificados como prováveis na EDA.
- **Atenção a data leakage:** O valor da mediana utilizado para imputação deve ser calculado apenas com os dados de treinamento. O mesmo valor calculado no treino será aplicado aos dados de teste.

#### 3.2 Codificação de `Gender`
- A variável `Gender` é categórica com dois valores: Male e Female.
- **Estratégia planejada:** Codificação binária (Label Encoding ou mapeamento direto: Male = 1, Female = 0, ou vice-versa).
- **Alternativa:** One-Hot Encoding, embora para variável binária o resultado seja equivalente.
- **Justificativa:** Por se tratar de uma variável binária, a codificação simples é suficiente e não introduz ordinaidade artificial.

#### 3.3 Transformação da variável-alvo
- A variável `Dataset` originalmente utiliza os valores 1 (doença) e 2 (sem doença).
- **Transformação:**
  - `1` → `1` (doença hepática)
  - `2` → `0` (sem doença hepática)
- **Implementação:** `df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})` ou equivalente.
- Essa transformação deve ser realizada antes da separação entre features e alvo.

#### 3.4 Separação entre features (X) e alvo (y)
- Após as transformações, separar:
  - `X = df.drop('Dataset', axis=1)` — features preditoras.
  - `y = df['Dataset']` — variável-alvo.

#### 3.5 Identificação de possíveis problemas de escala
- As features numéricas possuem escalas muito diferentes (ex.: `Age` varia de 4 a 90; `Alkaline_Phosphotase` pode ultrapassar 2000).
- **Impacto:** Modelos baseados em distância (KNN, SVM) e modelos com regularização (Regressão Logística) são sensíveis à escala das features. Modelos baseados em árvore (Random Forest, XGBoost) são invariantes à escala.
- **Decisão:** A padronização (StandardScaler) será aplicada via pipeline, conforme a Fase 5.

#### 3.6 Tratamento de outliers
- O tratamento de outliers depende das descobertas da EDA (Fase 2).
- **Opções a considerar:**
  - **Manutenção:** Se os outliers forem valores clinicamente plausíveis.
  - **Capping (winsorization):** Limitar valores extremos ao percentil 1 e 99.
  - **Transformação logarítmica:** Aplicar `log1p` a features com distribuição muito assimétrica (ex.: bilirrubina, enzimas hepáticas).
  - **Remoção:** Somente em casos de valores claramente impossíveis.
- **Atenção a data leakage:** Os limites de capping ou os parâmetros de transformação devem ser calculados exclusivamente a partir dos dados de treinamento.

> ⚠️ **Aviso sobre Data Leakage:** Qualquer transformação que utilize informações estatísticas do dataset (média, mediana, percentis, desvio-padrão) deve ser ajustada (`.fit()`) exclusivamente nos dados de treinamento. Os dados de teste devem ser transformados (`.transform()`) utilizando os parâmetros aprendidos no treino. Isso será garantido pela utilização de pipelines na Fase 5.

### Critérios de Conclusão

- [ ] Valores ausentes tratados com estratégia documentada.
- [ ] Variável `Gender` codificada numericamente.
- [ ] Variável-alvo transformada para formato binário (1/0).
- [ ] Features e alvo separados em `X` e `y`.
- [ ] Problemas de escala identificados e estratégia definida.
- [ ] Decisão sobre outliers documentada e justificada.
- [ ] Nenhuma transformação baseada em estatísticas do dataset completo (sem data leakage).

---

## Fase 4 — Divisão dos Dados

### Objetivo
Separar o dataset em conjuntos de treinamento e teste de forma reprodutível e estratificada.

### Configuração

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Parâmetros

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `test_size` | 0.2 | 80% para treinamento e 20% para teste. É uma divisão padrão que preserva um volume razoável de dados para treino e teste. |
| `random_state` | 42 | Garante que a divisão seja reprodutível em todas as execuções. |
| `stratify` | `y` | Preserva a proporção das classes nos conjuntos de treinamento e teste. |

### Por que a estratificação é importante neste dataset?

O dataset ILPD é desbalanceado: aproximadamente 71,4% dos registros pertencem à classe 1 (doença hepática) e 28,6% à classe 0 (sem doença hepática). Sem estratificação, uma divisão aleatória pode resultar em proporções de classes significativamente diferentes entre treino e teste, especialmente considerando o tamanho relativamente pequeno do dataset (583 registros).

A estratificação garante que:
- O conjunto de treinamento mantém a mesma proporção de classes do dataset original, permitindo que o modelo aprenda a partir de uma distribuição representativa.
- O conjunto de teste reflete a distribuição real, permitindo que a avaliação de desempenho seja mais realista e representativa.
- Os resultados são mais estáveis e comparáveis entre diferentes execuções e experimentos.

### Tamanhos esperados (aproximados)

| Conjunto | Total | Classe 1 (doença) | Classe 0 (sem doença) |
|---|---|---|---|
| Treinamento (80%) | ~466 | ~333 | ~133 |
| Teste (20%) | ~117 | ~83 | ~34 |

### Critérios de Conclusão

- [ ] Dados divididos com os parâmetros especificados.
- [ ] Proporção das classes verificada em ambos os conjuntos.
- [ ] Nenhuma informação do conjunto de teste utilizada em etapas anteriores.

---

## Fase 5 — Pipeline de Pré-processamento

### Objetivo
Construir uma pipeline reprodutível utilizando ferramentas do scikit-learn que encapsule todas as transformações de pré-processamento, garantindo que não haja data leakage e que o mesmo fluxo possa ser aplicado de forma consistente ao treino e ao teste.

### Componentes da Pipeline

#### 5.1 Identificação das colunas

```python
colunas_numericas = ['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
                     'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                     'Aspartate_Aminotransferase', 'Total_Protiens',
                     'Albumin', 'Albumin_and_Globulin_Ratio']

colunas_categoricas = ['Gender']
```

#### 5.2 Pipeline numérica

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pipeline_numerica = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
```

- **Imputação:** `SimpleImputer` com estratégia `median` para preencher os valores ausentes em `Albumin_and_Globulin_Ratio`.
- **Padronização:** `StandardScaler` para normalizar todas as features numéricas para média 0 e desvio-padrão 1.

#### 5.3 Pipeline categórica

```python
from sklearn.preprocessing import OneHotEncoder

pipeline_categorica = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])
```

- **Codificação:** `OneHotEncoder` com `drop='first'` para evitar a armadilha da variável dummy (multicolinearidade perfeita). Como `Gender` é binária, isso resulta em uma única coluna.

#### 5.4 ColumnTransformer

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers=[
    ('num', pipeline_numerica, colunas_numericas),
    ('cat', pipeline_categorica, colunas_categoricas)
])
```

#### 5.5 Pipeline completa (preprocessor + modelo)

```python
from sklearn.pipeline import Pipeline

pipeline_completa = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', modelo)  # substituir por cada modelo a ser treinado
])
```

### Quais modelos precisam de padronização?

| Modelo | Precisa de padronização? | Justificativa |
|---|---|---|
| Regressão Logística | **Sim** | A regularização (L1/L2) penaliza os coeficientes, que são sensíveis à escala das features. Sem padronização, features com escalas maiores dominam a regularização. |
| KNN | **Sim** | O cálculo de distância (euclidiana, por exemplo) é diretamente afetado pela escala. Features com valores maiores dominam a distância. |
| SVM | **Sim** | O kernel RBF e outros kernels dependem de distância. A padronização é essencial para o bom funcionamento. |
| Random Forest | **Não** | Modelos baseados em árvore utilizam partições em valores individuais de cada feature. A escala não afeta a decisão de partição. |
| XGBoost | **Não** | Mesma justificativa do Random Forest. Entretanto, a padronização não prejudica o modelo. |
| MLP (MLPClassifier) | **Sim** | Redes neurais utilizam gradiente descendente para otimização dos pesos. Features com escalas diferentes fazem o treinamento convergir lentamente ou ficar instável. A padronização é essencial. |

**Nota:** Mesmo que modelos baseados em árvore não necessitem de padronização, utilizar uma pipeline única simplifica o código. Alternativamente, podem ser criadas pipelines separadas: uma com padronização (para modelos sensíveis à escala) e outra sem (para modelos baseados em árvore).

### Critérios de Conclusão

- [ ] Pipeline numérica implementada (imputação + padronização).
- [ ] Pipeline categórica implementada (codificação).
- [ ] `ColumnTransformer` configurado corretamente.
- [ ] Pipeline completa testada com `.fit_transform()` nos dados de treinamento.
- [ ] Verificação de que `.fit()` é chamado apenas nos dados de treinamento.
- [ ] Dados de teste transformados apenas com `.transform()`.

---

## Fase 6 — Modelagem: Baseline

### Objetivo
Estabelecer uma referência mínima de desempenho (baseline) para que todos os modelos subsequentes possam ser comparados de forma objetiva. Um modelo que não supere o baseline é considerado ineficaz.

### Modelos Baseline

#### 6.1 Dummy Classifier

```python
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='most_frequent')
```

- **Estratégia:** `most_frequent` — o classificador sempre prediz a classe majoritária (doença hepática, classe 1).
- **Finalidade:** Representa o pior caso aceitável. Um modelo útil precisa necessariamente superar o Dummy Classifier.
- **Resultado esperado:** Accuracy de aproximadamente 71,4% (proporção da classe majoritária), mas Recall da classe 0 = 0% e F1-score da classe 0 = 0%.

#### 6.2 Regressão Logística (baseline simples)

```python
from sklearn.linear_model import LogisticRegression

lr_baseline = LogisticRegression(random_state=42, max_iter=1000)
```

- **Finalidade:** Modelo linear simples, com configuração padrão, para servir como baseline mais realista.
- **Justificativa:** A Regressão Logística é interpretável, rápida e funciona bem como primeiro modelo. O desempenho dela indica se um modelo linear já consegue resolver o problema razoavelmente.

### Métricas de Avaliação do Baseline

| Métrica | Descrição |
|---|---|
| Accuracy | Proporção total de acertos. |
| Precision (classe 1) | Dentre os preditos como doença, quantos realmente têm doença. |
| Recall (classe 1) | Dentre os que realmente têm doença, quantos foram identificados. |
| F1-score | Média harmônica entre Precision e Recall. |
| ROC-AUC | Capacidade do modelo de discriminar entre as classes. |
| Matriz de Confusão | Visualização detalhada dos acertos e erros por classe. |

### Critérios de Conclusão

- [ ] Dummy Classifier treinado e avaliado.
- [ ] Regressão Logística treinada e avaliada com configuração padrão.
- [ ] Métricas do baseline registradas para referência futura.
- [ ] Resultados documentados sem interpretações prematuras.

---

## Fase 7 — Comparação de Modelos

### Objetivo
Implementar e comparar múltiplos algoritmos de classificação para identificar os candidatos mais promissores, antes de realizar ajuste de hiperparâmetros.

### Modelos Planejados

#### 7.1 Regressão Logística

| Aspecto | Detalhe |
|---|---|
| **Justificativa** | Modelo linear, interpretável, rápido e eficiente. Serve como ponto de partida sólido. |
| **Padronização** | Necessária (regularização L2 é sensível à escala). |
| **Hiperparâmetros principais** | `C` (inverso da força de regularização), `penalty` (L1, L2, ElasticNet), `solver`, `max_iter`. |
| **Estratégia de treinamento** | Treinar via pipeline com preprocessor. |
| **Métricas** | Accuracy, Precision, Recall, F1-score, ROC-AUC, matriz de confusão. |

#### 7.2 KNN (K-Nearest Neighbors)

| Aspecto | Detalhe |
|---|---|
| **Justificativa** | Modelo não-paramétrico baseado em instâncias. Captura relações locais nos dados. |
| **Padronização** | Necessária (distância euclidiana é sensível à escala). |
| **Hiperparâmetros principais** | `n_neighbors`, `weights` (uniform, distance), `metric` (euclidean, manhattan). |
| **Estratégia de treinamento** | Treinar via pipeline com preprocessor. |
| **Métricas** | Accuracy, Precision, Recall, F1-score, ROC-AUC, matriz de confusão. |

#### 7.3 SVM (Support Vector Machine)

| Aspecto | Detalhe |
|---|---|
| **Justificativa** | Eficaz em espaços de alta dimensionalidade e com margens de separação claras. Pode capturar fronteiras de decisão não-lineares com kernel RBF. |
| **Padronização** | Necessária (kernel RBF depende de distância). |
| **Hiperparâmetros principais** | `C` (regularização), `kernel` (linear, rbf, poly), `gamma` (escala do kernel). |
| **Estratégia de treinamento** | Treinar via pipeline com preprocessor. Utilizar `SVC(probability=True)` para calcular ROC-AUC. |
| **Métricas** | Accuracy, Precision, Recall, F1-score, ROC-AUC, matriz de confusão. |

#### 7.4 Random Forest

| Aspecto | Detalhe |
|---|---|
| **Justificativa** | Ensemble de árvores de decisão. Robusto a outliers e não requer padronização. Fornece importância das features nativamente. |
| **Padronização** | Não necessária (invariante à escala). |
| **Hiperparâmetros principais** | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`. |
| **Estratégia de treinamento** | Pode usar pipeline com ou sem scaler. |
| **Métricas** | Accuracy, Precision, Recall, F1-score, ROC-AUC, matriz de confusão. |

#### 7.5 XGBoost (se compatível com o ambiente)

| Aspecto | Detalhe |
|---|---|
| **Justificativa** | Gradient boosting otimizado. Frequentemente alcança os melhores resultados em competições e problemas tabulares. |
| **Padronização** | Não necessária (baseado em árvore). |
| **Hiperparâmetros principais** | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `scale_pos_weight`. |
| **Estratégia de treinamento** | Verificar se `xgboost` está instalado no ambiente. Treinar via pipeline. |
| **Métricas** | Accuracy, Precision, Recall, F1-score, ROC-AUC, matriz de confusão. |

#### 7.6 MLP (Multi-Layer Perceptron)

| Aspecto | Detalhe |
|---|---|
| **Justificativa** | Rede neural capaz de capturar relações não-lineares complexas entre as features. O `MLPClassifier` do scikit-learn permite utilizar uma rede neural com a mesma interface dos demais modelos, facilitando a integração ao pipeline. |
| **Padronização** | Necessária (o gradiente descendente é sensível à escala das features). |
| **Hiperparâmetros principais** | `hidden_layer_sizes` (arquitetura da rede), `activation` (relu, tanh), `solver` (adam, sgd), `alpha` (regularização L2), `learning_rate`, `max_iter`, `early_stopping`. |
| **Estratégia de treinamento** | Treinar via pipeline com preprocessor (com scaler). Utilizar `early_stopping=True` para evitar overfitting. Fixar `random_state=42` para reprodutibilidade. |
| **Métricas** | Accuracy, Precision, Recall, F1-score, ROC-AUC, matriz de confusão. |
| **Cuidados** | O dataset é pequeno (583 registros), o que aumenta o risco de overfitting. Arquiteturas simples (poucas camadas e neurônios) são preferíveis. A regularização (`alpha`) e o `early_stopping` são essenciais. |

### Estratégia Geral de Comparação

1. Todos os modelos serão treinados utilizando os mesmos dados de treinamento.
2. A avaliação preliminar será feita com os dados de treinamento via validação cruzada (se necessário nesta fase).
3. Não serão utilizados dados de teste para comparação nesta fase.
4. Os resultados serão registrados em uma tabela para facilitar a comparação.

### Critérios de Conclusão

- [ ] Todos os modelos implementados e treinados.
- [ ] Métricas registradas para cada modelo.
- [ ] Tabela comparativa criada.
- [ ] Observações iniciais documentadas (sem conclusões precipitadas).

---

## Fase 8 — Tratamento do Desbalanceamento

### Objetivo
Investigar se o desbalanceamento de classes prejudica o desempenho dos modelos e comparar diferentes estratégias de tratamento.

### Experimentos Planejados

#### Experimento 1 — Sem tratamento do desbalanceamento
- Treinar os modelos sem nenhuma técnica de balanceamento.
- Serve como linha de referência para comparação.

#### Experimento 2 — `class_weight="balanced"`
- Utilizar o parâmetro `class_weight="balanced"` nos modelos que suportam essa configuração.
- **Modelos compatíveis:**
  - Regressão Logística: `LogisticRegression(class_weight='balanced')`
  - SVM: `SVC(class_weight='balanced')`
  - Random Forest: `RandomForestClassifier(class_weight='balanced')`
  - XGBoost: Utilizar `scale_pos_weight` com a razão entre as classes.
- **KNN:** Não suporta `class_weight`. Pode-se utilizar `weights='distance'` como alternativa parcial.
- **MLP:** Não suporta `class_weight` nativamente. O balanceamento pode ser feito via SMOTE ou utilizando `sample_weight` manualmente durante o treinamento.
- **Funcionamento:** O `class_weight="balanced"` ajusta internamente os pesos das amostras de forma inversamente proporcional à frequência da classe, sem modificar os dados.

#### Experimento 3 — SMOTE (Synthetic Minority Over-sampling Technique)
- Aplicar SMOTE para gerar amostras sintéticas da classe minoritária.
- **Biblioteca:** `imblearn` (`from imblearn.over_sampling import SMOTE`).

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)
```

> ⚠️ **ATENÇÃO — Data Leakage com SMOTE:**
>
> O SMOTE deve ser aplicado **exclusivamente ao conjunto de treinamento**, **após** a divisão entre treino e teste. Aplicar SMOTE antes da divisão faz com que amostras sintéticas baseadas nos dados de teste vazem para o treinamento, inflando artificialmente as métricas e produzindo uma avaliação irrealista do modelo.
>
> **Regra:** Nunca aplicar SMOTE ao dataset completo. O fluxo correto é:
> 1. Dividir os dados (`train_test_split`).
> 2. Aplicar pré-processamento (pipeline) aos dados de treinamento.
> 3. Aplicar SMOTE aos dados de treinamento já preprocessados.
> 4. Treinar o modelo com os dados balanceados.
> 5. Avaliar no conjunto de teste original (sem SMOTE).
>
> Alternativamente, utilizar `imblearn.pipeline.Pipeline` que integra SMOTE ao pipeline do scikit-learn.

### Comparação dos Resultados

Após executar os três experimentos, comparar:
- Métricas globais (Accuracy, F1-score, ROC-AUC).
- Recall da classe positiva (doença hepática) — métrica crítica neste contexto.
- Precision — para verificar se o tratamento não gera excesso de falsos positivos.
- Determinar se o tratamento do desbalanceamento realmente melhora o desempenho ou se introduz trade-offs inaceitáveis.

### Critérios de Conclusão

- [ ] Três experimentos realizados para cada modelo.
- [ ] Resultados comparados em tabela.
- [ ] Melhor estratégia de balanceamento identificada para cada modelo.
- [ ] Data leakage evitado em todos os experimentos.
- [ ] Conclusão documentada sobre a necessidade (ou não) do tratamento de desbalanceamento.

---

## Fase 9 — Avaliação dos Modelos

### Objetivo
Definir critérios objetivos e métricas quantitativas para avaliar e comparar todos os modelos treinados, com atenção especial ao contexto médico do problema.

### Métricas de Avaliação

| Métrica | Fórmula/Descrição | Relevância neste projeto |
|---|---|---|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Visão geral, mas pode ser enganosa em datasets desbalanceados. |
| **Precision** | TP / (TP + FP) | Proporção de predições positivas corretas. Importante para avaliar a taxa de alarmes falsos. |
| **Recall** | TP / (TP + FN) | Proporção de positivos reais identificados. **Métrica crítica neste projeto.** |
| **F1-score** | 2 × (Precision × Recall) / (Precision + Recall) | Equilíbrio entre Precision e Recall. Útil quando há desbalanceamento. |
| **ROC-AUC** | Área sob a curva ROC | Capacidade geral de discriminação, independente do threshold. |
| **Matriz de Confusão** | Tabela TP/TN/FP/FN | Visualização detalhada dos tipos de erro. |

### Por que o Recall é especialmente importante?

No contexto de classificação médica, um **falso negativo** (paciente com doença hepática classificado como saudável) é potencialmente mais perigoso do que um **falso positivo** (paciente saudável classificado como doente):

- **Falso negativo:** O paciente doente não recebe o encaminhamento necessário, podendo agravar sua condição por falta de tratamento.
- **Falso positivo:** O paciente saudável é encaminhado para exames adicionais, gerando custo e ansiedade, mas sem risco direto à saúde.

Portanto, o **Recall da classe positiva (doença hepática)** deve receber atenção especial na avaliação. Um modelo com Accuracy alta mas Recall baixo pode ser clinicamente inadequado.

> **Nota:** Essa análise não implica que o modelo possa ser utilizado como ferramenta clínica. Trata-se de uma consideração acadêmica para orientar a escolha de métricas.

### Tabela Comparativa Final (Planejamento)

A tabela abaixo será preenchida com os resultados obtidos:

| Modelo | Balanceamento | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|---|
| Dummy Classifier | — | | | | | |
| Regressão Logística | Nenhum | | | | | |
| Regressão Logística | class_weight | | | | | |
| Regressão Logística | SMOTE | | | | | |
| KNN | Nenhum | | | | | |
| KNN | SMOTE | | | | | |
| SVM | Nenhum | | | | | |
| SVM | class_weight | | | | | |
| SVM | SMOTE | | | | | |
| Random Forest | Nenhum | | | | | |
| Random Forest | class_weight | | | | | |
| Random Forest | SMOTE | | | | | |
| XGBoost | Nenhum | | | | | |
| XGBoost | scale_pos_weight | | | | | |
| XGBoost | SMOTE | | | | | |
| MLP | Nenhum | | | | | |
| MLP | SMOTE | | | | | |

### Critérios de Conclusão

- [ ] Todas as métricas calculadas para todos os modelos e estratégias de balanceamento.
- [ ] Tabela comparativa preenchida.
- [ ] Análise do Recall da classe positiva documentada.
- [ ] Matrizes de confusão geradas para todos os modelos.

---

## Fase 10 — Validação Cruzada e Ajuste de Hiperparâmetros

### Objetivo
Selecionar os melhores hiperparâmetros para os modelos mais promissores, utilizando validação cruzada estratificada e busca sistemática.

### Validação Cruzada Estratificada

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

- **Número de folds:** 5 (compromisso entre variância e viés da estimativa).
- **Estratificação:** Garante que cada fold mantenha a proporção das classes.
- **Shuffle:** Embaralha os dados antes de criar os folds.
- **Escopo:** Utilizar **somente os dados de treinamento** para validação cruzada. O conjunto de teste permanece reservado.

### Busca de Hiperparâmetros

#### Opção 1 — Grid Search

```python
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=pipeline_completa,
    param_grid=param_grid,
    cv=cv,
    scoring='recall',  # ou 'f1', dependendo da prioridade
    n_jobs=-1,
    verbose=1
)
```

- **Vantagem:** Exploração exaustiva do espaço de hiperparâmetros.
- **Desvantagem:** Custo computacional elevado para muitas combinações.

#### Opção 2 — Randomized Search

```python
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator=pipeline_completa,
    param_distributions=param_distributions,
    n_iter=50,
    cv=cv,
    scoring='recall',
    random_state=42,
    n_jobs=-1,
    verbose=1
)
```

- **Vantagem:** Mais eficiente computacionalmente, especialmente com muitos hiperparâmetros.
- **Desvantagem:** Não garante encontrar o melhor ponto do espaço.

### Hiperparâmetros a Explorar (por modelo)

#### Regressão Logística
```python
param_grid_lr = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear', 'saga']
}
```

#### KNN
```python
param_grid_knn = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
}
```

#### SVM
```python
param_grid_svm = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__kernel': ['rbf', 'linear'],
    'classifier__gamma': ['scale', 'auto', 0.01, 0.1]
}
```

#### Random Forest
```python
param_grid_rf = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 5, 10, 15, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2']
}
```

#### XGBoost
```python
param_grid_xgb = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 5, 7, 10],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__subsample': [0.7, 0.8, 1.0],
    'classifier__colsample_bytree': [0.7, 0.8, 1.0]
}
```

#### MLP (Multi-Layer Perceptron)
```python
param_grid_mlp = {
    'classifier__hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)],
    'classifier__activation': ['relu', 'tanh'],
    'classifier__solver': ['adam'],
    'classifier__alpha': [0.0001, 0.001, 0.01, 0.1],
    'classifier__learning_rate': ['constant', 'adaptive'],
    'classifier__max_iter': [500],
    'classifier__early_stopping': [True]
}
```

> **Nota sobre a MLP:** Dado o tamanho reduzido do dataset (583 registros), recomenda-se utilizar `RandomizedSearchCV` em vez de `GridSearchCV` para a MLP, pois o espaço de hiperparâmetros é grande. Arquiteturas com poucas camadas e neurônios devem ser priorizadas para evitar overfitting.

### Métrica de Otimização

- **Primária:** `recall` — priorizar a identificação de pacientes doentes.
- **Alternativa:** `f1` — quando o equilíbrio entre Precision e Recall for preferido.
- A escolha da métrica de otimização deve ser documentada e justificada.

### Comparação Justa

- Todos os modelos devem ser avaliados utilizando a mesma validação cruzada (mesmo `cv`).
- As métricas reportadas devem incluir média e desvio-padrão dos folds.
- A comparação deve ser feita entre os melhores hiperparâmetros de cada modelo.

### Critérios de Conclusão

- [ ] Validação cruzada estratificada implementada.
- [ ] Busca de hiperparâmetros realizada para os modelos mais promissores.
- [ ] Melhores hiperparâmetros registrados para cada modelo.
- [ ] Métricas de validação cruzada (média ± desvio-padrão) documentadas.
- [ ] Modelos comparados de forma justa.
- [ ] Conjunto de teste **não utilizado** nesta fase.

---

## Fase 11 — Avaliação Final

### Objetivo
Avaliar o melhor modelo selecionado no conjunto de teste reservado, obtendo uma estimativa realista do desempenho em dados não vistos.

### Procedimento

1. **Selecionar o melhor modelo** com base nos resultados da Fase 10 (validação cruzada).
2. **Re-treinar o modelo** com os melhores hiperparâmetros utilizando **todo o conjunto de treinamento**.
3. **Avaliar no conjunto de teste:**
   - Calcular todas as métricas definidas na Fase 9.
   - Gerar a matriz de confusão.
   - Plotar a curva ROC.
4. **Análise de erros e acertos:**
   - Examinar os casos de falsos negativos e falsos positivos.
   - Verificar se há padrões nos erros (ex.: faixa etária, gênero, valores de exames).
5. **Comparação com o baseline:**
   - Comparar as métricas finais com as do Dummy Classifier e da Regressão Logística baseline.
   - Quantificar a melhoria relativa.

> ⚠️ **Regra fundamental:** O conjunto de teste NÃO deve ser utilizado para tomar decisões durante o desenvolvimento (escolha de features, hiperparâmetros, estratégia de balanceamento, etc.). Ele é reservado exclusivamente para a avaliação final. Se o modelo for ajustado com base nos resultados do teste, a avaliação perde sua validade.

### Resultados a Registrar

- Métricas finais do melhor modelo.
- Matriz de confusão do melhor modelo.
- Curva ROC do melhor modelo.
- Comparação com o baseline.
- Análise qualitativa dos erros.

### Critérios de Conclusão

- [ ] Melhor modelo selecionado e justificado.
- [ ] Modelo re-treinado com todo o conjunto de treinamento.
- [ ] Avaliação no conjunto de teste realizada.
- [ ] Todas as métricas calculadas e registradas.
- [ ] Matriz de confusão e curva ROC geradas.
- [ ] Análise de erros documentada.
- [ ] Comparação com baseline documentada.
- [ ] Conjunto de teste utilizado apenas nesta fase.

---

## Fase 12 — Interpretabilidade

### Objetivo
Compreender quais features influenciam as previsões do modelo, tornando os resultados mais transparentes e interpretáveis.

### Métodos Planejados

#### 12.1 Coeficientes da Regressão Logística
- Se a Regressão Logística for o modelo final (ou um dos modelos avaliados), extrair e analisar os coeficientes.
- Coeficientes positivos indicam associação com a classe positiva (doença); negativos indicam associação com a classe negativa (sem doença).
- A magnitude indica a importância relativa da feature (após padronização).
- **Visualização:** Gráfico de barras horizontal com os coeficientes ordenados.

#### 12.2 Importância das features em modelos de árvore
- Random Forest e XGBoost fornecem nativamente a importância de cada feature.
- **Tipos de importância:**
  - `feature_importances_` — baseada em impureza (Gini ou entropia). Pode ser enviesada em favor de features com mais categorias ou maior cardinalidade.
  - Importância por permutação (`permutation_importance`) — mais robusta, mede a queda no desempenho ao embaralhar cada feature.
- **Visualização:** Gráfico de barras horizontal com a importância ordenada.

#### 12.3 SHAP (SHapley Additive exPlanations)
- Se apropriado e se a biblioteca `shap` estiver disponível.
- SHAP fornece uma explicação individual para cada predição, baseada na teoria dos jogos (valores de Shapley).
- **Visualizações:**
  - `shap.summary_plot()` — visão global da importância e direção do efeito de cada feature.
  - `shap.force_plot()` — explicação individual de predições específicas.
- **Vantagem:** SHAP é model-agnostic e fornece explicações consistentes e teoricamente fundamentadas.

### Importância Preditiva vs. Causalidade

> **Aviso importante:** A importância de uma feature para o modelo preditivo **não implica causalidade**. Por exemplo, se o modelo identifica que a bilirrubina total é a feature mais importante, isso significa que ela é útil para a previsão no contexto deste dataset. **Não significa** que a bilirrubina total causa doença hepática.
>
> Para estabelecer relações causais, seriam necessários estudos controlados e delineamentos experimentais apropriados, que estão fora do escopo deste projeto.
>
> As análises de interpretabilidade devem ser apresentadas como "associações preditivas" e não como "causas da doença".

### Critérios de Conclusão

- [ ] Pelo menos um método de interpretabilidade aplicado ao modelo final.
- [ ] Features mais importantes identificadas e documentadas.
- [ ] Visualizações geradas.
- [ ] Distinção entre importância preditiva e causalidade documentada.

---

## Fase 13 — Análise de Limitações e Possíveis Vieses

### Objetivo
Documentar de forma transparente as limitações do projeto e do modelo, evitando generalizações indevidas.

### Limitações Identificadas

#### 13.1 Tamanho reduzido do dataset
- O dataset contém apenas 583 registros, o que é relativamente pequeno para técnicas de Machine Learning.
- Datasets pequenos aumentam a variância das estimativas e dificultam a generalização.
- Modelos complexos (muitos parâmetros) são mais propensos a overfitting em datasets pequenos.

#### 13.2 Desbalanceamento das classes
- A proporção 71,4% / 28,6% entre as classes pode enviesar modelos em direção à classe majoritária.
- Mesmo com técnicas de balanceamento (class_weight, SMOTE), o desbalanceamento limita a quantidade de exemplos reais da classe minoritária disponíveis para aprendizado.

#### 13.3 Origem geográfica específica
- Os dados foram coletados de pacientes do nordeste da Índia (Andhra Pradesh).
- Os padrões de doença hepática, fatores de risco e perfil clínico podem variar significativamente entre populações de diferentes regiões geográficas, etnias e contextos socioeconômicos.
- **Implicação:** O modelo pode não generalizar para populações de outros países ou regiões.

#### 13.4 Possíveis limitações de representatividade
- Não é possível verificar se a amostra é representativa da população geral de pacientes com e sem doença hepática.
- Vieses de seleção (quem procurou atendimento médico, quem foi incluído no estudo) podem afetar a composição do dataset.

#### 13.5 Distribuição por gênero
- A composição de gênero do dataset pode não refletir a distribuição real na população.
- Se houver desproporção significativa, o modelo pode ter desempenho diferente para diferentes gêneros.
- Essa limitação deve ser analisada na EDA e documentada nos resultados.

#### 13.6 Risco de overfitting
- O tamanho reduzido do dataset e o número relativamente alto de features aumentam o risco de overfitting.
- Modelos complexos podem memorizar os dados de treinamento sem capturar padrões generalizáveis.
- **Mitigações:** Validação cruzada, regularização, early stopping (para XGBoost), análise da diferença entre métricas de treino e teste.

#### 13.7 Limitações para generalização
- Os resultados obtidos são válidos no contexto deste dataset específico.
- Não é possível afirmar que o modelo terá o mesmo desempenho em outros datasets ou populações.
- A performance em novos dados pode ser significativamente diferente.

#### 13.8 Impossibilidade de uso como ferramenta clínica
- O modelo desenvolvido neste projeto é um exercício acadêmico e de aprendizado.
- **O modelo NÃO é uma ferramenta clínica validada** e não deve ser utilizado para diagnóstico ou decisões médicas.
- Para ser utilizado em ambiente clínico, um modelo precisaria:
  - Ser validado em múltiplos datasets independentes.
  - Passar por estudos clínicos prospectivos.
  - Ser aprovado por órgãos regulatórios competentes.
  - Ser utilizado como ferramenta auxiliar, nunca como substituto do julgamento médico.

> **Nota ética:** Nenhuma afirmação médica ou causal deve ser feita com base nos resultados deste projeto. As conclusões devem ser limitadas ao contexto do dataset e da tarefa de classificação.

### Critérios de Conclusão

- [ ] Todas as limitações documentadas.
- [ ] Linguagem cuidadosa utilizada, evitando generalizações.
- [ ] Distinção entre resultados acadêmicos e aplicação clínica explicitada.

---

## Fase 14 — Organização e Reprodutibilidade

### Objetivo
Garantir que o projeto esteja organizado de forma clara, documentada e reprodutível, permitindo que qualquer pessoa possa compreender e reexecutar o trabalho.

### Estrutura de Diretórios Proposta

```
ILPD_ML/
│
├── data/
│   ├── raw/                          # Dataset original, sem modificações
│   │   └── indian_liver_patient.csv
│   └── processed/                    # Dados processados (se salvos)
│       └── ...
│
├── notebooks/
│   ├── 01_preparacao_dados.ipynb     # Fase 1: Carregamento e entendimento
│   ├── 02_eda.ipynb                  # Fase 2: Análise exploratória
│   ├── 03_preprocessamento.ipynb     # Fase 3: Pré-processamento
│   ├── 04_modelagem_baseline.ipynb   # Fases 4-6: Divisão, pipeline, baseline
│   ├── 05_comparacao_modelos.ipynb   # Fases 7-8: Modelos e balanceamento
│   ├── 06_ajuste_hiperparametros.ipynb # Fase 10: Validação cruzada e tuning
│   ├── 07_avaliacao_final.ipynb      # Fase 11: Avaliação final
│   └── 08_interpretabilidade.ipynb   # Fase 12: Interpretabilidade
│
├── src/                              # Código-fonte reutilizável (opcional)
│   ├── __init__.py
│   ├── preprocessing.py              # Funções de pré-processamento
│   ├── evaluation.py                 # Funções de avaliação
│   └── visualization.py              # Funções de visualização
│
├── reports/
│   └── figures/                      # Gráficos e visualizações exportadas
│       └── ...
│
├── models/                           # Modelos treinados salvos (opcional)
│   └── ...
│
├── plano de implementação.md         # Este documento
├── requirements.txt                  # Dependências do projeto
├── README.md                         # Descrição geral do projeto
└── .gitignore                        # Arquivos a serem ignorados pelo Git
```

### Boas Práticas de Reprodutibilidade

1. **Versionar o código:** Utilizar Git para controle de versão.
2. **Fixar dependências:** Registrar as versões exatas das bibliotecas em `requirements.txt`.
3. **Fixar seeds:** Utilizar `random_state=42` em todas as operações estocásticas.
4. **Documentar decisões:** Cada decisão técnica (tratamento de outliers, escolha de hiperparâmetros, etc.) deve ser justificada e documentada nos notebooks ou neste plano.
5. **Separar dados brutos:** Nunca modificar o arquivo original do dataset. Trabalhar sempre com cópias.
6. **Notebooks executáveis:** Cada notebook deve ser executável do início ao fim sem erros, na ordem numérica.
7. **Nomes descritivos:** Utilizar nomes claros e descritivos para variáveis, funções e arquivos.

### Dependências Esperadas

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
shap
jupyter
```

### Critérios de Conclusão

- [ ] Estrutura de diretórios criada.
- [ ] `requirements.txt` gerado.
- [ ] `README.md` escrito com descrição do projeto.
- [ ] `.gitignore` configurado.
- [ ] Notebooks organizados e executáveis na ordem.
- [ ] Código documentado e comentado.
- [ ] Todas as seeds fixadas.
- [ ] Projeto publicável e reprodutível.
