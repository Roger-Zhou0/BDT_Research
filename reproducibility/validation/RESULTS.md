# Supplementary record recomputation

Offline arithmetic; zero new model evaluations. Confidence intervals are paired across the stated training parents, not checkpoints or observations.

| Study | Comparison | Mean | 95% interval | Evidence |
|---|---|---:|---|---|
| CMS1 | bdt - adamw | -0.002696905285 | [-0.003930910772, -0.001462899798] | 18 original REPORT files |
| CMS1 | bdt - scrambled_matched | -0.006098807169 | [-0.007726635975, -0.004470978362] | 18 original REPORT files |
| GCT2 REPORT | bdt - adamw | -0.002203855974 | [-0.002272996780, -0.002134715168] | Unrounded audited projection; same six trained parents |
| GCT2 REPORT | bdt - scrambled_matched | -0.003147829634 | [-0.003213828911, -0.003081830358] | Unrounded audited projection; same six trained parents |
| GCT2 FEV1 | bdt - adamw | -0.002219119522 | [-0.002260623706, -0.002177615339] | Unrounded audited projection; same six trained parents |
| GCT2 FEV1 | bdt - scrambled_matched | -0.003214982647 | [-0.003312358846, -0.003117606447] | Unrounded audited projection; same six trained parents |
| BAC1 | reassigned - preserved | +0.003300288692 | [+0.002971574612, +0.003629002772] | Unrounded final analysis + exact 24-endpoint cohort lock |

## Four reused adjacent-copy parents

| Study | Comparison | Mean | Interval level | Interval |
|---|---|---:|---:|---|
| AGT1 | draw0_reassigned - draw0_preserved | +0.008230128791 | 97.5% | [+0.007825390231, +0.008634867352] |
| AGT1 | draw0_preserved - adamw | -0.006139748264 | 97.5% | [-0.006553540096, -0.005725956433] |
| FRT1 | fixed_true_double - adamw | -0.005608562380 | 95% | [-0.005768655113, -0.005448469647] |
| ART1 | assembled - fixed | -0.001198649406 | 97.5% | [-0.001448325494, -0.000948973319] |
| ART1 | assembled - combined | +0.000508984551 | 97.5% | [+0.000350096780, +0.000667872322] |
| ART1 | difference - fixed | -0.000169567298 | 95% | [-0.000220540796, -0.000118593800] |
| ART1 | attention - fixed | -0.001117981505 | 95% | [-0.001381260025, -0.000854702985] |
| ART1 | assembled - difference | -0.001029082108 | 95% | [-0.001241049309, -0.000817114907] |
| ART1 | assembled - attention | -0.000080667902 | 95% | [-0.000190556717, +0.000029220914] |
| ART1 | assembled - adamw | -0.006807211787 | 95% | [-0.006957848001, -0.006656575572] |
| BCT1 | bdt_budget_gs - structured_budget_gs | -0.000618126709 | 95% | [-0.000772291517, -0.000463961901] |
| BCT1 | bdt_budget_gs - combined | +0.000095988624 | 95% | [+0.000073702431, +0.000118274817] |
| BCT1 | structured_budget_gs - combined | +0.000714115333 | 95% | [+0.000548703190, +0.000879527476] |
| BCT1 | bdt_budget_gs - assembled | -0.000412995927 | 95% | [-0.000518551215, -0.000307440639] |
| BCT1 | structured_budget_gs - assembled | +0.000205130782 | 95% | [+0.000100315805, +0.000309945759] |

## Four fresh BCT2 parents (separate cohort)

Full-BDT budgets minus structured budgets: -0.000593933277 nats/token; paired 95% interval [-0.000633208717, -0.000554657837].
All eight original score receipts and the exact final report are checked. Anonymous seal/protocol derivatives retain explicit original/public hash mappings. No pooling with BCT1.


The JSON also preserves 97.5% CMS1/GCT2 intervals, all per-parent values, source precision, 240 repeated allocation observations, and hashes of the 42 table fragments. Display-level and raw-record evidence remain separate.
