# MMR and ONNX exploratory experiments

## Baseline
- Condition: C2, text + Marqo image features
- Split: s4
- Training seed: 42
- Catalogue: 53,232 products
- Evaluation customers: 49,061

## MMR
MMR reranked a fixed pool of 300 candidates into 50 recommendations.
Relevance used saved model-vector cosine scores.
Redundancy used Marqo image-embedding cosine similarity.

Visual diversity increased as diversification strengthened, while
Recall@50 and NDCG@50 decreased.

At lambda=0.999:
- Recall@50 difference: approximately -0.000070
- Pointwise 95% customer-bootstrap interval: [-0.000166, 0.000012]
- Visual diversity difference: approximately +0.001369

An interval containing zero does not establish unchanged accuracy.
The lambda grid was expanded after inspecting test-set pilot results;
all comparisons are exploratory.

## ONNX correctness
On 1,000 fixed sampled customers:
- Top-50 and Top-300 recommendation sets matched between implementations.
- Recall@50 and Recall@300 were identical.
- Some within-list ordering differed near numerical ties.
- Maximum observed score difference: approximately 5.36e-7.

## ONNX CPU latency
Single-request measurements, one computational thread:
20 warm-up calls per scope/backend, followed by
3 rounds of 500 measurements per backend and scope.

| Scope | PyTorch median | ONNX median | PyTorch p95 | ONNX p95 |
|---|---:|---:|---:|---:|
| User tower | 0.300869 ms | 0.125833 ms | 1.411889 ms | 0.180367 ms |
| User tower + scoring + Top-300 | 3.039797 ms | 2.828390 ms | 10.439685 ms | 9.747339 ms |

User-tower median latency decreased by approximately 58.2%.
Retrieval-component median latency decreased by approximately 7.0%.

## File sizes
Per tower:
- PyTorch weights: 1,315,979 bytes
- ONNX: 1,314,365 bytes

The approximately 0.12% difference is not substantial compression.
No quantization was applied.

## Limitations
- One trained condition, split and seed.
- Bootstrap intervals are pointwise and conditional on the fixed model.
- Visual diversity uses the same representation targeted by MMR.
- Visual diversity is not a direct measure of user satisfaction.
- ONNX recommendation agreement was checked on a 1,000-customer sample.
- Latency was measured on the Colab CPU, not the deployment computer.
- Item vectors were cached for retrieval.
- Timing excludes history averaging, database access, HTTP handling,
  text/image encoding and MMR.
- Full API latency below 100 ms has not been established.

## Saved results
- MMR tables, recommendations, bootstrap intervals and figures: mmr/
- ONNX models, correctness checks and latency measurements: onnx/
