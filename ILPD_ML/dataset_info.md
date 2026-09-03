Indian Liver Patient Records (ILPD)
Visão geral

O Indian Liver Patient Records (também conhecido como ILPD — Indian Liver Patient Dataset) é um conjunto de dados disponível no Kaggle sob o identificador uciml/indian-liver-patient-records, originalmente doado ao UCI Machine Learning Repository em 2012.

O dataset reúne registros de pacientes coletados na região Nordeste de Andhra Pradesh, Índia, com o objetivo de apoiar a criação de modelos de classificação para prever se um paciente tem ou não doença hepática (fígado), com base em marcadores bioquímicos do sangue.

Origem: UCI Machine Learning Repository (doado em 20/05/2012)
Área: Saúde e Medicina
Tarefa associada: Classificação (binária)
Nº de instâncias: 583 pacientes
Nº de atributos (features): 10 + 1 variável-alvo
Tipos de dados: Inteiro, Real e Categórico (Gender)
Distribuição das classes
Classe	Descrição	Nº de registros
1	Paciente com doença hepática	416
2	Paciente sem doença hepática	167

Do total de 583 registros, 441 são do sexo masculino e 142 do sexo feminino — um desbalanceamento relevante a se considerar em análises e modelagem.

Colunas do dataset
Coluna	Tipo	Descrição
Age	Inteiro	Idade do paciente. Pacientes com mais de 89 anos foram registrados como idade "90"
Gender	Categórico	Sexo do paciente (Male / Female)
Total_Bilirubin	Real	Bilirrubina total no sangue
Direct_Bilirubin	Real	Bilirrubina direta (conjugada)
Alkaline_Phosphotase	Inteiro	Nível de fosfatase alcalina
Alamine_Aminotransferase (SGPT)	Inteiro	Enzima alanina aminotransferase
Aspartate_Aminotransferase (SGOT)	Inteiro	Enzima aspartato aminotransferase
Total_Protiens	Real	Proteínas totais no sangue
Albumin	Real	Nível de albumina
Albumin_and_Globulin_Ratio	Real	Razão albumina/globulina (pode conter valores ausentes)
Dataset (ou Selector)	Inteiro (alvo)	1 = tem doença hepática, 2 = não tem doença hepática

⚠️ Observação: a coluna Albumin_and_Globulin_Ratio costuma apresentar alguns valores faltantes (NaN), sendo comum aplicar imputação (ex.: média/mediana) antes de treinar modelos.

Considerações sobre sensibilidade dos dados

O próprio UCI destaca que o dataset contém informações potencialmente sensíveis (idade e gênero dos pacientes) e que, historicamente, pacientes do sexo feminino parecem ser menos diagnosticados precocemente com doenças hepáticas — um ponto interessante para análise exploratória.

Usos comuns

Este dataset é amplamente utilizado em pesquisas e tutoriais de machine learning para:

Classificação binária (doença hepática vs. saudável) com algoritmos como Regressão Logística, SVM, Random Forest, KNN, Naive Bayes, XGBoost, entre outros
Estudos de balanceamento de classes (ex.: técnicas como SMOTE, SMOTE-ENN)
Análise exploratória de dados (EDA) relacionando enzimas hepáticas, bilirrubina e proteínas com o diagnóstico
Comparações entre populações (ex.: pacientes indianos vs. americanos em estudos correlatos)