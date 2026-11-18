# Andamento do Projeto: Assistente Acadêmico Inteligente

**Atualizado em:** 18/11/2026 · **Prazo de entrega:** _a definir_ · **Repositório:** <https://github.com/helberkaue/assistente-academico-ia>

## Resumo

O assistente lê fotos de material didático impresso e responde perguntas sobre o conteúdo, mostrando o trecho de onde tirou a resposta. Ele é formado por quatro módulos: Visão Computacional, Machine Learning, PLN e Agente de decisão.

**Hoje:** o módulo de Visão Computacional está pronto e avaliado. Os outros três módulos, a interface e o artigo ainda não começaram.

## Situação por critério de avaliação

| Critério | Pontos | Situação |
|---|:---:|---|
| Visão Computacional | 15 | ✅ Pronto. Falta avaliar com fotos reais |
| Definição do problema e arquitetura | 10 | 🟡 Problema e diagrama no README. Falta formalizar objetivos e público-alvo |
| GitHub e reprodutibilidade | 5 | 🟡 Estrutura, testes e instruções prontos. Cresce com os próximos módulos |
| Experimentos e avaliação | 10 | 🟡 1 de 4 experimentos feito |
| Machine Learning | 15 | ❌ Não começado |
| Processamento de Linguagem Natural | 15 | ❌ Não começado |
| Agente / decisão | 10 | ❌ Não começado |
| Integração e funcionamento | 10 | ❌ Não começado |
| Artigo no Overleaf | 10 | ❌ Não começado |

## O que já está pronto

- **Pré-processamento de imagem (OpenCV):** remoção de ruído, correção de sombra, correção de inclinação e binarização.
- **OCR:** EasyOCR como padrão e Tesseract como opção, os dois com a mesma interface.
- **Métricas:** CER e WER, que medem a taxa de erro por caractere e por palavra.
- **Gerador de imagens de teste:** cria imagens de texto com defeitos controlados e o gabarito pronto.
- **Experimento 1:** em 60 imagens, o erro de caracteres caiu de **31,1%** sem pré-processamento para **4,5%** com ele. A binarização piorou o resultado (8,9%) e por isso ficou desligada. Os gráficos estão em `resultados/graficos/`.
- **15 testes automatizados**, todos passando.

## O que falta desenvolver

| Módulo | O que fazer | Como avaliar |
|---|---|---|
| **Machine Learning** | Classificar a matéria do texto com TF-IDF e comparar Regressão Logística, Naive Bayes e SVM, treinando com textos da Wikipédia | Acurácia, F1-macro, matriz de confusão; desempenho com texto vindo do OCR |
| **PLN** | Reconhecer a intenção da pergunta, buscar o trecho relevante (TF-IDF contra embeddings) e extrair a resposta | Recall@k, MRR, Exact Match e F1 nas perguntas de avaliação |
| **Agente de decisão** | Regras que escolhem a ação: pedir outra foto, responder, resumir ou dizer "não encontrei" | Taxa de acerto e de abstenção conforme o limiar |
| **Interface** | App em Streamlit: upload da foto, texto extraído, matéria detectada, chat e resposta com o trecho de origem | Teste de ponta a ponta com fotos reais |

**Ordem:** ML e PLN em paralelo, depois o Agente de decisão, depois a Interface, os experimentos finais e o artigo.

## O que depende da equipe

### Agora (travam o desenvolvimento)

- [ ] **Confirmar as matérias.** Hoje são Biologia, História, Matemática, Química, Geografia e Programação, e as fotos precisam ser dessas matérias.
- [ ] **Nomes completos e responsabilidade de cada integrante.**
- [ ] **Informar o prazo de entrega.**
- [ ] **Entrar como colaborador no GitHub** (*Settings → Collaborators*).

### Em paralelo

- [ ] **Tirar de 20 a 30 fotos reais** de material impresso (veja o guia abaixo).
- [ ] **Transcrever cada foto** em um `.txt` com o mesmo nome, exatamente como está impresso.
- [ ] **Escrever de 40 a 60 perguntas de avaliação**, incluindo umas 10 cuja resposta **não** está no material.

### No final

- [ ] **Testar a instalação** em um computador que não foi usado no desenvolvimento, seguindo só o README.
- [ ] **Avaliar as respostas do app à mão:** certa, parcial ou errada.
- [ ] **Cada integrante conseguir explicar o próprio módulo** e os números dos resultados.
- [ ] **Escrever o artigo no Overleaf** (template IEEE Conference, até 6 páginas) e compartilhar o link.
- [ ] **Entregar:** link do GitHub, link do Overleaf e o PDF.

## Guia rápido das fotos

**O que fotografar:** texto impresso das matérias combinadas, de um parágrafo a meia página por foto, com definições, listas e datas que dê para perguntar. O repositório é público, então usem trechos impressos da Wikipédia, material próprio ou slides com permissão. Evitem páginas de livro comercial.

**Condições (para ~25 fotos):**

| Condição | Qtd. | Como tirar |
|---|:---:|---|
| Boa | ~10 | Luz uniforme, celular paralelo à folha |
| Com sombra | ~5 | Sombra da mão ou luz vindo de um lado |
| Inclinada | ~5 | Folha girada de 3° a 10° na mesa |
| Pouca luz | ~2 | Ambiente escuro, sem flash |
| Difícil | ~3 | Perspectiva, página curva, reflexo ou tela do projetor |

**Regras:**
- Câmera normal do celular, sem app de scanner, sem filtro e sem recorte.
- Sem linhas cortadas na borda da foto.
- Sem rostos, nomes ou matrículas.

**Onde e como salvar:** em `dados/imagens_reais/`.

```
foto01.jpg
foto01.txt                       <- texto exato da foto
materias.csv                     <- arquivo,materia,condicao
                                    foto01.jpg,Biologia,boa
```

**Perguntas de avaliação**, em `dados/perguntas.csv`:

```
arquivo,pergunta,resposta_esperada,intencao
foto01.jpg,O que é a fotossíntese?,processo pelo qual plantas...,definicao
```

A `resposta_esperada` deve ser um trecho copiado do texto da foto. Os tipos de `intencao` são: `definicao`, `lista`, `factual` e `resumo`.

---

Instalação e execução: veja o [README](README.md).
