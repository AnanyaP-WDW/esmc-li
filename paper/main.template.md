# ESMC-LI: Scaling Late-Interaction Protein Retrieval: A Practical Index for ESM-C Residue Embeddings

## Abstract

Late-interaction retrieval represents a protein as a set of residue vectors and
scores a pair with MaxSim. ProtoCol [@protocol] showed that this idea is
sensitive for protein homolog search, but it deferred efficient search over
large databases. We ask whether the idea survives deployment: full-length
chains, database sizes where pooling and alignment baselines stop being the
only practical option, and an index that runs locally. We use ESM-C
{{n:model.params_m}}M [@esmc] with a learned projection and a partially
fine-tuned backbone. On SCOPe
ASTRAL40 [@scope] the fine-tuned scorer reaches cR@10
{{n:t2.m3.cR@10_pm}}, compared with {{n:t2.protocol_trained.cR@10}} for a
reproduced ProtoCol, {{n:t2.maxsim960.cR@10}} for frozen ESM-C MaxSim, and
{{n:t2.pooled.cR@10}} for mean-pooled ESM-C. On a deployment-shaped corpus of
{{n:data.fl.db_n}} full-length Swiss-Prot chains [@uniprot], it reaches
{{n:t3.trained_raw.cR@10}} cR@10 versus {{n:t3.mmseqs2.cR@10}} for MMseqs2,
with a larger gap in sensitivity at superfamily level
({{n:t3.trained_raw.sens_superfamily}} versus
{{n:t3.mmseqs2.sens_superfamily}}). Exact search costs
{{n:t6.sq.100000}} s/query at {{n:t6.n.100000}} proteins and
{{n:t6.sq.1000000}} s/query at {{n:t6.n.1000000}} on an Apple M4 Max laptop;
the index is {{n:t6.gb.100000}} GB and {{n:t6.gb.1000000}} GB respectively, and
build time is {{n:t6.build_h.100000}} h and {{n:t6.build_h.1000000}} h. Build
and storage, not query search, are the wall. Compression does not pay off, and
coarse reranking helps only in the middle of the measured range. We release an
offline package and the measured system.

## Introduction

Protein sequence databases are large, and the retrieval question is simple:
given a query sequence, return the proteins that share its homology or
function. Alignment methods such as BLAST [@blast] and MMseqs2 [@mmseqs2] are
strong and fast. Dense retrieval methods summarize a protein as one vector
[@dhr] [@knnprott5] [@tmvec], or compare residue representations with an
alignment-like aggregation [@plmsearch] [@plmblast] [@peba]. Late interaction
sits between these extremes: it keeps one vector per residue and scores a pair
with MaxSim, the sum over query residues of the best match in the candidate.
ColBERT [@colbert] made this practical for text; ProtoCol [@protocol] brought
it to protein homolog search and showed a sensitivity advantage over pooled
ESM-2 embeddings and MMseqs2 on small SCOPe and Pfam test sets.

ProtoCol left the systems question open. Its evaluation used short domains,
truncated sequences to a few hundred residues, and test sets of a few thousand
proteins. The model idea is theirs. This paper asks the next question: does
late-interaction protein retrieval survive a real database, and what does it
cost? We make four contributions.

- **A deployment-shaped benchmark.** We evaluate on
  {{n:data.fl.db_n}} full-length Swiss-Prot chains with {{n:data.fl.queries_n}}
  clan-disjoint queries, including {{n:data.fl.gt2046_n}} chains longer than
  the ESM-C context window. This is closer to what a user indexes than a set of
  short domains.
- **A backbone and training study.** Frozen ESM-C is weaker than frozen ESM-2
  35M under the same harness ({{n:t2b.t1.maxsim960.cr10}} versus
  {{n:t2b.t1.diagD.cr10}} cR@10). The ESM-C upgrade requires fine-tuning; a
  {{n:train.k_n}}-block suffix plus a {{n:model.d}}-dimensional head recovers
  and exceeds the reproduced ProtoCol on SCOPe.
- **An exact index and measured systems.** We store projected residue rows in a
  single memory-mapped `.esmci` file, search with tiled GEMM plus a fused Metal
  max kernel, and measure exact search from
  {{n:t6.n.10000}} to {{n:t6.n.1000000}} proteins. We also report where
  compression and coarse retrieval stop helping.
- **Software.** The **ESMC-LI** package builds, saves, loads, and searches
  an index offline through a small `ProteinIndex` API.

## Related work

**Late interaction.** ColBERT [@colbert] introduced contextualized late
interaction for text. PLAID [@plaid], MUVERA [@muvera], and DESSERT [@dessert]
made multi-vector retrieval efficient, and PyLate [@pylate] provides a modern
training and retrieval stack. ProtoCol [@protocol] is the direct prior work for
proteins: it trains a residue-level projection with a symmetric contrastive
objective and evaluates capped recall on SCOPe and Pfam. We reuse that model
idea and cite it as such; our contribution is the deployment-shaped benchmark,
the ESM-C backbone study, and the index/search measurements.

**Dense protein retrieval.** DHR [@dhr] trains a dual encoder for homolog
detection. TM-Vec and DeepBLAST [@tmvec] predict structural similarity and
alignment from protein language model embeddings. PLMSearch and PLMAlign
[@plmsearch] use residue embeddings with a learned similarity head and an
alignment stage. pLM-BLAST [@plmblast] and PEbA [@peba] compare residue
representations directly. knnProtT5 [@knnprott5] is the mean-pooled
nearest-neighbor baseline that motivates our pooled control. These methods are
not interchangeable with ours: they use different backbones, splits, and
metrics. We therefore cite them as related work rather than quoting their
numbers next to ours.

**Structure and scale.** Foldseek [@foldseek] searches structures and is the
natural comparison when structures exist; we evaluate sequences only. ERAST
[@erast] targets billion-scale vector search with a hosted database. Our
systems claim is deliberately smaller and exact: a local index whose search
cost we measure.

**Sequence baselines.** MMseqs2 [@mmseqs2] and BLAST [@blast] are the
alignment baselines. MinHash over k-mers [@minhash] is a cheap sketch baseline
in the spirit of ProtoCol's comparison. These are the only external retrieval
methods we run end to end; all other comparisons are left to future work.

## Method

### Representation and scorer

ESM-C {{n:model.params_m}}M [@esmc] maps a sequence to one vector per residue.
We take the final
layer, apply a linear projection from H={{n:model.h}} to D={{n:model.d}}, and
L2-normalize each projected residue row. For a query A and candidate B,
MaxSim is

    S(A, B) = sum_{i in A} max_{j in B} a_i . b_j.

We use the raw sum. Dividing by the query length is rank-invariant within a
query and affects only calibration; dividing by the candidate length or by the
geometric mean of the two lengths changes the ranking and is evaluated as an
ablation. ESM-C's context is {{n:model.ctx_n}} tokens; residues are packed into
windows of {{n:model.win_n}} with {{n:model.overlap_n}} overlap, and the
overlap rows are emitted once. The resulting row count equals the sequence
length exactly. A residue may match any residue in the candidate; chunk
identity is dropped at scoring time.

### Training

The projection is trained with symmetric InfoNCE [@infonce] over MaxSim, using
in-batch negatives. We use temperature {{n:train.tau_n}}, batch
{{n:train.batch_n}}, and {{n:train.epochs_n}} epochs; the temperature is swept
rather than inherited from the vision-language default [@clip]. We mask false
negatives:
an in-batch pair that shares the positive's superfamily is removed from the
denominator. This is a small correctness fix over ProtoCol's unfiltered
in-batch objective. The first phase freezes ESM-C and trains only the
projection. The second phase fine-tunes the last {{n:train.k_n}} transformer
blocks and the final layer norm with the projection, using a smaller learning
rate for the backbone ({{n:train.lr_blocks_n}}) than for the head
({{n:train.lr_head_n}}). Training pairs are same-superfamily SCOPe domains;
selection uses only the validation split. The test split is evaluated once
after selection. We use {{n:train.seeds_n}} seeds and report mean and
population standard deviation.

### Index and search

The index stores the projected residue rows as half-precision values in a
single file: a fixed header, one offset per protein, and a packed payload.
Metadata lives in a JSON sidecar. Inference runs through `esmc.cpp`
[@esmccpp]. Loading is a memory map; opening the
{{n:data.fl.db_n}}-protein index takes milliseconds. Search is exact: tile the
query against the database rows, keep a running maximum per candidate, and
return the top list. On Apple Silicon the max is fused into the Metal kernel so
the similarity matrix never leaves the GPU. A numpy path is the fallback. The
optional coarse stage pools each protein into one vector, retrieves C
candidates, and reranks them with exact MaxSim.

## Experimental setup

**SCOPe.** We use SCOPe ASTRAL40: {{n:data.scope.domains}} domains split by
superfamily into {{n:data.scope.splits}} train/val/test. The test set has
{{n:data.scope.test_n}} domains. The maximum train/test identity over
alignments covering at least eighty percent of both sequences is
{{n:data.scope.identity}}. Queries are the test rows themselves; the corpus is
the split, and the query is excluded from its own ranking. Relevance is the
shared superfamily, following ProtoCol's capped recall.

**Full-length Swiss-Prot.** We build a corpus of {{n:data.fl.db_n}}
single-clan reviewed chains, with {{n:data.fl.queries_n}} queries drawn
round-robin from {{n:data.fl.query_clans_n}} test clans. Queries are rows of the
corpus and are scored leave-one-out, so the frozen evaluation harness applies
unchanged. {{n:data.fl.gt2046_n}} chains exceed the residue window; the longest
is {{n:data.fl.maxlen_n}} amino acids. Train, validation, and test clans are
disjoint.

**Scaling.** For systems measurements we use UniRef50 [@uniref50] subsets of
{{n:t6.n.10000}}, {{n:t6.n.100000}}, and {{n:t6.n.1000000}} proteins. These
measure build, storage, and search, not retrieval quality: the larger subsets
do not have complete homology labels.

**Baselines.** MMseqs2 is run with the settings that reproduce ProtoCol's
published comparison. MinHash uses short k-mers and a fixed sketch budget.
Pooled ESM-C is
the mean of the residue rows with cosine similarity. Frozen MaxSim uses the
same ESM-C rows without training, at H and at a random D projection. ProtoCol
is reproduced from its released code on our split. FAISS flat on pooled vectors
[@faiss] is a latency floor, not a quality baseline. We report capped recall at
several cutoffs, MRR, and sensitivity to the first false positive at family,
superfamily, and fold level.

**Hardware.** All measurements are on an Apple M4 Max laptop with Metal and a
CPU fallback. We do not claim GPU-server numbers.

## Results

### SCOPe: training is necessary, and it beats the reproduced baseline

Table 2 is the main domain result. Frozen ESM-C MaxSim is far ahead of pooled
ESM-C ({{n:t2.maxsim960.cR@10}} versus {{n:t2.pooled.cR@10}} cR@10) but still
behind the reproduced ProtoCol. A trained projection closes most of the gap:
{{n:t2.m2.cR@10_pm}}. Fine-tuning the last {{n:train.k_n}} blocks closes the
rest: {{n:t2.m3.cR@10_pm}}, above the reproduced ProtoCol
({{n:t2.protocol_trained.cR@10}}) and far above MMseqs2
({{n:t2.mmseqs2.cR@10}}). The sensitivity numbers tell the same story: at
superfamily level the fine-tuned model reaches
{{n:t2.m3.sens_superfamily_pm}}, versus {{n:t2.mmseqs2.sens_superfamily}} for
MMseqs2. ProtoCol's released evaluator does not report sensitivity; its capped
recall is {{n:t2.protocol_trained.cR@10}}.

<!-- TABLE:t2_scope -->

The backbone control is Table 2b. Frozen ESM-C is not a drop-in replacement for
a smaller frozen ESM-2: on the same harness, frozen ESM-2 35M reaches
{{n:t2b.t1.diagD.cr10}} cR@10 versus {{n:t2b.t1.maxsim960.cr10}} for frozen
ESM-C. The ESM-C gain appears only after fine-tuning. This is a useful negative
result: a stronger backbone does not remove the need for task training.

<!-- TABLE:t2b_backbone -->

### Full-length chains: the deployment-shaped result

Table 3 is the result that matters for users. On {{n:data.fl.db_n}}
full-length chains, the fine-tuned scorer reaches
{{n:t3.trained_raw.cR@10}} cR@10 versus {{n:t3.mmseqs2.cR@10}} for MMseqs2 and
{{n:t3.maxsim960_raw.cR@10}} for frozen MaxSim. The gap widens for the stricter
sensitivity metric: {{n:t3.trained_raw.sens_superfamily}} versus
{{n:t3.mmseqs2.sens_superfamily}} at superfamily level and
{{n:t3.trained_raw.sens_family}} versus {{n:t3.mmseqs2.sens_family}} at family
level. Pooling loses most of the signal ({{n:t3.pooled_raw.cR@10}} cR@10), and
MinHash is weaker still ({{n:t3.minhash.cR@10}}). This is the main scientific
claim: residue-level late interaction is not only a domain benchmark artifact;
it survives full-length chains and a clan-disjoint corpus.

<!-- TABLE:t3_fulllength -->

### Length bias: real but mild; do not normalize the candidate length

The suspected confound is that MaxSim grows with candidate length because a
longer candidate offers more chances for a high-scoring match. We measure it
directly. On the full-length corpus the Spearman correlation between candidate
length and score over non-homolog pairs is
{{n:t4.trained.raw.rho}} for the fine-tuned scorer and
{{n:t4.maxsim960.raw.rho}} for frozen MaxSim. That is mild. Raw scoring wins:
dividing by the query length is rank-invariant, while dividing by the candidate
length collapses cR@10 from {{n:t4.trained.raw.cr10}} to
{{n:t4.trained.divB.cr10}}, and the geometric-mean variant also loses
({{n:t4.trained.divGeo.cr10}}). We therefore keep the raw sum. The measurement
matters more than a correction: the length prior is not strong enough to
explain the retrieval quality, and candidate-length normalization is worse than
doing nothing.

<!-- TABLE:t4_length -->

<!-- FIGURE:f1_length -->

*Figure 1. Length normalization on the full-length corpus. Raw and query-length
normalization are rank-equivalent; candidate-length normalization destroys the
ranking.*

### Ablations: dimension, temperature, masking, and depth

Table 5a shows the projection dimension. D={{n:model.d}} is the default:
{{n:t5a.d128.cr10_pm}} validation cR@10 at {{n:t5a.d128.bytes}} bytes per
residue. D={{n:t5a.d64.d}} is smaller but weaker, and D={{n:t5a.d256.d}} buys
{{n:t5a.d256.cr10_pm}} for twice the index. Table 5b shows that masking wins at
every temperature; temperature itself is a flat knob in this range. Table 5c
shows that three fine-tuned blocks beat two
({{n:t5c.k3.val_pm}} versus {{n:t5c.k2.val_pm}} validation cR@10), and the
selected model reaches {{n:t5c.k3.test_pm}} on test.

<!-- TABLE:t5a_dims -->

<!-- TABLE:t5b_tau -->

<!-- TABLE:t5c_k -->

### Systems: exact search is practical to a million, build is the wall

Table 6a is the systems result. At {{n:t6.n.100000}} proteins, exact search costs
{{n:t6.sq.100000}} s/query; at {{n:t6.n.1000000}} it costs
{{n:t6.sq.1000000}} s/query. The index is {{n:t6.gb.100000}} GB and
{{n:t6.gb.1000000}} GB. Build time is {{n:t6.build_h.100000}} h and
{{n:t6.build_h.1000000}} h. Opening an index is milliseconds, so the interactive
cost is query embedding plus search, not load. On one hundred thousand proteins
the fused Metal path reaches {{n:t6b.sq.metal_fused}} s/query
({{n:t6b.qps.metal_fused}} q/s), versus {{n:t6b.sq.numpy_cpu}} s/query on the
CPU and {{n:t6b.sq.mps}} on MPS. The CPU path is already near the GEMM roofline
({{n:t6b.eff.numpy_cpu}} percent); the GPU win comes from fusing the max and
avoiding the score-matrix write.

<!-- TABLE:t6a_scaling -->

<!-- TABLE:t6b_devices -->

<!-- FIGURE:f2_scaling -->

*Figure 2. Exact search and index size versus database size. The
million-protein point is measured; beyond it we do not extrapolate a promise.*

The coarse stage is Table 8. It recovers
{{n:t8.recall.1000000}} of the exhaustive top-10 candidate set at
{{n:t8.C.1000000}} candidates on one million proteins, but costs
{{n:t8.coarse.1000000}} s/query against {{n:t8.exhaustive.1000000}} for
exhaustive: only {{n:t8.speedup.1000000}}x. The rerank gathers candidates at
random from an {{n:t6.gb.1000000}} GB index, so the gather dominates. Coarse
retrieval is therefore not shipped as a default. It is the right direction for
a future layout, not a measured win here.

<!-- TABLE:t8_coarse -->

### Compression: the recall bar is reachable, the latency bar is not

Table 7 reports compression on the full-length index. Integer quantization is
only a factor of two smaller, below our size bar. Product quantization reaches
a factor of {{n:t7.pq_m32.ratio}} at a cR@10 drop of {{n:t7.pq_m32.drop}}
({{n:t7.pq_m32.cr10}}), which passes the recall bar, but decode plus MaxSim was
slower than reading half-precision rows. We keep the uncompressed index and
report the negative result. A fused decode-and-max kernel is the obvious next
step; it is not part of this paper's claim.

<!-- TABLE:t7_compression -->

<!-- FIGURE:f3_compression -->

*Figure 3. Compression trade-off. Passing the recall bar is not enough; the
decode path must also beat reading the raw rows.*

### Package

The package exposes `ProteinIndex.from_fasta`, `build`, `load`, `search`, and
`search_batch`, plus a CLI. A fresh install builds and searches a small SCOPe
index offline; the release smoke retrieves the same superfamily in
{{n:t9.smoke.superfamily}} of {{n:t9.smoke.queries}} top-1 results on a
{{n:t9.smoke.n}}-protein index. Table 9 profiles
the runtime. At one thousand proteins, Metal search is
{{n:t9.Metal.1000.search}} ms/query; at one hundred thousand it is
{{n:t9.Metal.100000.search_s}} s/query. The million-protein numpy path is
{{n:t9.numpy.1000000.search_s}} s/query, which is why the systems claim is
batch-mode at that size. The index is not self-contained: the matching head and
fine-tuned GGUF must travel with it, and the sidecar records their hashes.

<!-- TABLE:t9_profile -->

<!-- TABLE:t9b_smoke -->

## Discussion and limitations

The result is a practical exact index, not a new retrieval model. ProtoCol
introduced the model idea; we show that it survives full-length chains and that
its exact search is usable at a hundred thousand proteins on a laptop and
batch-usable at a million. The negative results are part of the contribution:
frozen ESM-C is weaker than frozen ESM-2, length normalization hurts, coarse
reranking does not pay off at a million with a random gather, and compression
that passes recall can still lose on latency.

The limits are equally clear. We did not run DHR, TM-Vec, PLMSearch, pLM-BLAST,
ERAST, or Foldseek end to end, so we do not claim a head-to-head win against
them. We did not evaluate Pfam retrieval. The identity control is one ASTRAL40
filter; we do not report ten, twenty, or thirty percent subsets. The full-length
experiment uses one corpus and one seed for the deployment number; the SCOPe
result is the three-seed result. The systems measurements are Apple Silicon
only. The million-protein point has systems measurements, not labeled retrieval
quality. Ten million proteins would need roughly a machine-week of build time
and an index close to the machine's memory limit; we did not run it, and we do
not claim it. Finally, the package's million-protein path falls back to numpy;
Metal is not claimed there.

## Conclusion

Late-interaction protein retrieval is practical when the representation is
fine-tuned, the index is exact, and the target is a local database. On SCOPe the
fine-tuned ESM-C scorer is slightly above a reproduced ProtoCol and far above
pooled and alignment baselines. On full-length Swiss-Prot it is clearly above
MMseqs2 at the sensitivity metrics that matter for remote relations. Exact
search costs about a second per query at a hundred thousand proteins and about
eleven seconds at a million, with build and storage as the real constraints. The
code, model, and index format are released so the next experiment can be a
better index rather than a reimplementation.

## Appendix A. Reproducibility

All paper numbers are generated from `paper/results.lock.json`, a hash-pinned
projection of `results/runs.jsonl` at a recorded commit. `make tables`
regenerates every table and data figure without the model, the embedding cache,
the network, or `esmc.cpp`. `make claims` regenerates this audit; `make audit`
checks that every numeric sentence maps to a table cell and that every citation
is verified. The fine-tuned GGUF and projection head are published separately
from the package and are hash-checked at load time. The SCOPe and Swiss-Prot
source files, split hashes, and model hashes are recorded in
`data/manifest.json` and `data/fulllength/manifest.json`. The model weights are
governed by the Cambrian Open License [@cambrian]; the runtime is MIT.

<!-- REFERENCES -->
