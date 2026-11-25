# Experimentos com o juiz Qwen3-8B

No Colab, use [avaliacao_colab.ipynb](../avaliacao_colab.ipynb) com GPU T4. O notebook clona a branch `feature/llm-judge` em `/content`, acessa os módulos e `respostas.jsonl` do repositório e grava `checkpoint.sqlite`, tabelas e gráficos no Google Drive. Na linha de comando, execute a partir da raiz do projeto, em Python 3.11+ com PyTorch CUDA e GPU compatível com quantização NF4. O carregamento do modelo ocorre na primeira chamada que ainda precisa de julgamento. A primeira execução real precisa de acesso ao Hugging Face.

```sh
python -m pip install -r avaliacao/requirements.txt
python -m unittest discover -s tests -v
```

## GSM8K

```sh
python -m avaliacao gsm8k --entrada respostas.jsonl --saida saidas_juiz/gsm8k
```

Cada candidato recebe um veredito binário sobre a correção da resposta. O juiz vê o texto e o problema. A referência independente é a comparação entre o campo `numero` do Antônio e o `gabarito`, com tolerância absoluta inferior a 1e-6. Uma extração ausente conta como erro. Essa referência mede a resposta numérica final e pode divergir da avaliação do texto.

As estratégias usam as primeiras 4, 8 ou 16 respostas e os limiares adaptativos 0,90 e 0,95 do notebook original. Na variante com juiz, cada resposta consumida conta no custo, e apenas números aprovados votam. Se nenhum número votar, o problema conta como erro. Em empate vence o primeiro número observado entre os empatados. A parada usa a probabilidade beta dos dois números mais votados. A comparação é retrospectiva sobre os mesmos candidatos, na mesma ordem. A economia reportada é a fração de amostras que a estratégia dispensaria em relação ao orçamento disponível de até 16. O experimento também avalia todos os candidatos para medir a qualidade do juiz. O custo observado dessa avaliação inclui esses julgamentos.

## MT-Bench

```sh
python -m avaliacao mtbench --saida saidas_juiz/mtbench
```

Usa o split `human` de `lmsys/mt_bench_human_judgments`, com revisão fixa. Para cada anotação, o juiz compara A e B até o turno indicado. O prompt omite os nomes dos modelos, o anotador e o vencedor humano. As classes são A, B e empate. A concordância exata considera os empates como terceira classe. Cada anotação humana tem o mesmo peso, inclusive quando várias pessoas julgaram o mesmo par. Os relatórios mostram o total e cada turno. Esse protocolo avalia preferências pareadas. [Ficha do conjunto](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)

## Retomar e interpretar os arquivos

Repita o mesmo comando com a mesma entrada, limite e pasta de saída. Cada julgamento concluído e cada item agregado têm transações separadas em `checkpoint.sqlite`. Uma interrupção no meio de um item preserva os julgamentos já gravados. A retomada pode repetir a inferência em curso se a interrupção anteceder sua gravação. Prompts idênticos compartilham um julgamento. O checkpoint verifica a identidade da fonte, revisão do modelo e versão do protocolo. Mantenha a pasta em armazenamento persistente e use uma sessão por diretório no Colab.

Para um recorte posterior, acrescente, por exemplo, `--limite 3` e escolha uma pasta própria. No GSM8K o limite conta problemas. No MT-Bench conta anotações humanas. Para usar uma cópia local do split MT-Bench, passe `--entrada caminho.jsonl` com os mesmos campos do dataset.

Ao terminar, o experimento exporta `metricas.csv`, `classes.csv`, `confusao.csv` e `metricas.png`. A acurácia usa todos os julgamentos, cobertura é a fração com saída válida, e macro-F1 é a média simples do F1 das classes do protocolo. Saídas inválidas aparecem na matriz e contam como erros. Divisões sem suporte recebem zero nas métricas por classe. No GSM8K, `estrategias.csv` e `estrategias.png` comparam a acurácia por problema e o consumo de amostras. O CSV também traz paradas antecipadas incorretas, economia e problemas sem resposta. Cada estratégia fixa inclui apenas problemas com candidatos suficientes para o orçamento indicado.

Os testes usam entradas sintéticas, simulam o backend e escrevem os relatórios em diretórios temporários. O notebook vem sem saídas executadas.
