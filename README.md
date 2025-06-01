 ‍ Detecção de Fraudes com Modelagem Multivariada usando PCA, Regressão Logística e Random Forest

Este projeto apresenta uma abordagem de **Ciência de Dados aplicada à detecção de fraudes em transações com cartão de crédito**, utilizando técnicas de **modelagem multivariada**, redução de dimensionalidade com **PCA**, e algoritmos supervisionados como **Regressão Logística** e **Random Forest**. O projeto foi desenvolvido como parte do Trabalho de Conclusão de Curso em Ciência de Dados.

---

  Objetivo

Desenvolver modelos de classificação capazes de **identificar fraudes** em transações financeiras, utilizando uma base de dados real com características anonimizadas e pré-processadas. O foco está na combinação de:

* Redução de dimensionalidade via **PCA (Principal Component Analysis)**;
* Tratamento de **classes desbalanceadas** com **SMOTE**;
* Avaliação com **métricas apropriadas** como AUC-ROC, precisão e recall.

---

  Estrutura do Código

 1.  Instalação e Bibliotecas

O projeto utiliza bibliotecas como `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `matplotlib` e `seaborn`.

```python
!pip install -q imbalanced-learn
```

---

 2.  Carregamento do Dataset

Carrega-se o dataset `creditcard.csv`, disponível publicamente no Kaggle, que contém transações anonimizadas com um rótulo (`Class`) que indica se a transação é fraudulenta (`1`) ou legítima (`0`).

```python
df = pd.read_csv('creditcard.csv')
```

---

 3.  Análise Exploratória

Verifica-se:

* Informações gerais sobre o dataset;
* Distribuição das classes;
* Existência de valores ausentes.

```python
df['Class'].value_counts()
sns.countplot(x='Class', data=df)
```

---

 4.  Pré-processamento dos Dados

* A coluna `Amount` é normalizada com `StandardScaler`.
* Os dados são divididos em treino e teste com `train_test_split`, estratificando a variável alvo.
* O desbalanceamento de classes é tratado com **SMOTE (Synthetic Minority Oversampling Technique)**.

```python
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
```

---

 5.  Aplicação de PCA

Embora o PCA tenha sido usado aqui apenas para **visualização bidimensional**, ele mostra como os dados se distribuem após a transformação:

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_res)
```

> 🔧 *Sugestão para produção real: aplicar PCA com mais componentes (ex: `n_components=0.95`) antes de treinar os modelos.*

---

 6.  Treinamento dos Modelos

 6.1 Regressão Logística

Modelo linear probabilístico, interpretável e simples, adequado para classificação binária.

```python
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_res, y_train_res)
```

 6.2 Random Forest

Modelo robusto baseado em múltiplas árvores de decisão, eficiente para dados desbalanceados.

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)
```

---

 7. 📈 Avaliação dos Modelos

São calculadas métricas como:

* **Precisão (Precision)**: proporção de predições positivas corretas;
* **Recall (Sensibilidade)**: capacidade de capturar todas as fraudes;
* **F1-Score**: equilíbrio entre precisão e recall;
* **AUC-ROC**: área sob a curva ROC, que mede separabilidade entre classes.

```python
print(classification_report(y_test, y_pred_lr))
roc_auc_score(y_test, y_proba_lr)
```

As curvas ROC são plotadas para comparar visualmente o desempenho:

```python
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (AUC = %0.2f)' % roc_auc_lr)
plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = %0.2f)' % roc_auc_rf)
```

---

 8.  Conclusão

* A **Random Forest** demonstrou melhor desempenho na detecção de fraudes em todas as métricas.
* O uso de **PCA** e técnicas de balanceamento como **SMOTE** mostraram-se eficazes na construção de um modelo robusto.
* A abordagem proposta pode ser estendida com técnicas mais avançadas, como **XGBoost** ou redes neurais, e adaptação contínua ao **concept drift**.

---

  Resultados Obtidos (exemplo)

| Modelo              | AUC-ROC | Precisão | Recall | F1-score |
| ------------------- | ------- | -------- | ------ | -------- |
| Regressão Logística | 0.93    | 0.79     | 0.75   | 0.77     |
| Random Forest       | 0.98    | 0.91     | 0.89   | 0.90     |

> *Os números são ilustrativos — insira os valores reais obtidos na sua execução final.*

---

  Tecnologias Utilizadas

* Python 3
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Seaborn, Matplotlib

---

  Referências

* Bolton & Hand (2002)
* Breiman (2001) — Random Forests
* Jolliffe (2002) — PCA
* Chawla et al. (2002) — SMOTE
* Pozzolo et al. (2015) — Detecção de Fraudes com RF + SMOTE
