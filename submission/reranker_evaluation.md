# RFP Retrieval Evaluation

Evaluated 18 of 18 labeled questions.

| Mode | Recall@1 | Recall@5 | MRR@5 | Mean latency (s) |
|---|---:|---:|---:|---:|
| ms_marco | 0.611 | 0.889 | 0.727 | 1.91 |
| cohere | 0.778 | 0.889 | 0.833 | 2.32 |

**Method:** Each question has one annotated evidence passage identified by source document, physical page, and an evidence phrase. A hit occurs when any top-k chunk contains that evidence at the specified source. With one gold evidence unit per question, Recall@k is equivalent to Hit@k. MRR@k uses the first matching rank.

**Latency:** Includes retrieval, network and reranking inference; excludes one-time MS MARCO model loading and intentional Cohere rate-limit delays.
