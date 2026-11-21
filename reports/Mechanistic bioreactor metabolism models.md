# Build lean kinetics first, then learn the rates

The best model for this challenge is a **serial grey-box hybrid built on a deliberately small mechanistic core**. The core has five parts: per-batch mass balances with exact bolus jump maps and the measured volumes; Monod–Pirt growth and glucose uptake; viable → dead → lysed cell compartments; a smooth switch from lactate production to lactate consumption; and a latent "nutrient pool" that turns the unknown feeds B, C and D into three fitted, interpretable nutrient loadings. Once that core is validated, a small neural network or Gaussian process replaces only the specific rates the mechanistic terms get wrong (growth μ and lactate rate q_Lac). No public benchmark names a single best CHO model. Still, every head-to-head comparison found ranks hybrids at or above mechanistic models in accuracy, while mechanistic models are more robust to which batches go into training ([Bayer et al. 2023](https://doi.org/10.1002/biot.202200381); [Pinto et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10515724/)). The closest published analogue to this task is a 2026 Sartorius digital twin trained on 23 uniquely fed Ambr15 runs and validated leave-one-batch-out, and it uses almost exactly this recipe ([Richelle et al. 2026](https://doi.org/10.34133/csbj.0078)). Metabolic-network models (EFM macro-models, dFBA, genome-scale and kinetic-metabolic models) are the wrong dynamic engine here. Every validated CHO example used 7–60 measured metabolites and 45–253 parameters, while four measured species support roughly 10–15 shared parameters. At that size an EFM-derived model and an unstructured Monod model reduce to the same three reactions. The largest documented accuracy gains come from three ordinary choices rather than exotic kinetics: replaying each batch's true feed volumes (averaged feeds drove glucose R² negative in a 656-run study), a lactate switch (lactate remains the worst-predicted state, R² 0.36–0.77), and dead-cell lysis so that TCD can level off. Batch-to-batch variability that the feeds cannot explain is the least-solved problem. No CHO paper applies hierarchical or mixed-effects ODEs, so a population model with a few random effects is both a practical safeguard and a real novelty. In Python, a SciPy baseline takes a few hours to build and fit, and a JAX hybrid plus symbolic distillation fits into day two.

## Four measured species cap the model at about fifteen parameters

What decides this challenge is how little the data can pin down, not how sophisticated the kinetics are. Each batch reports viable and total cell density, one substrate, one byproduct and volume, all sampled offline, and three of the four feeds have unknown composition. That is enough to identify:

- a growth rate
- a death rate
- a lysis rate
- one uptake rate
- one secretion rate that changes sign
- one or two latent limitations

Not much more. The field has known this for a long time. Reviews call the unstructured, unsegregated Monod approach dominant "due to its traceability and simple formalism" and say large kinetic models are "severely hampered by the scarcity of kinetic data" ([Kyriakopoulos et al. 2018](https://doi.org/10.1002/biot.201700229)). A 2025 review repeats that for kinetic models "the number of parameters estimated can become very large, while the data available to describe them is limited", and that stoichiometric models stay underdetermined ([Ranpura et al. 2025](https://doi.org/10.1016/j.csbj.2025.06.035)).

Metabolic-network models show the problem most clearly. The elementary-flux-mode (EFM) line compresses an intracellular network into a few macro-reactions between measured extracellular species. Its central design rule is that **the number of retained macro-reactions must stay below the number of measured species** ([Maton et al. 2022](https://dipot.ulb.ac.be/dspace/bitstream/2013/358040/3/1-s2.0-S0959152422001627-main.pdf)). Earlier work built phase-wise EFM models joined by smooth switching ([Provost et al. 2006](https://doi.org/10.1007/s00449-006-0083-y)) and found ways around the combinatorial enumeration of detailed CHO networks ([Zamorano et al. 2013](https://doi.org/10.1016/j.jbiotec.2012.05.005)). With glucose, lactate and biomass measured, that rule leaves two or three macro-reactions.

Dynamic FBA sits at the other extreme. A pH- and temperature-dependent CHO dFBA validated on 20 Ambr250 runs reached **R² ≥ 0.97 for growth and titer and ≥ 0.85 for other metabolites**. It needed rich exo-metabolomics, and it still had **205 parameters after BIC pruning from 253** ([Ghodba et al. 2025](https://doi.org/10.1016/j.jbiotec.2025.08.010)). Other network models have the same limits:

- **Kinetic-priority hybrid dFBA** (35 reactions, **45 kinetic parameters**): reproduced the lactate and alanine shifts but mispredicted a 38 mM lactate-spike run ([Khare et al. 2025](https://doi.org/10.1101/2025.10.28.685134)).
- **Jolicoeur-lab kinetic-metabolic model**: **139 parameters, of which the 20 most sensitive were fitted**. It used intracellular metabolomics and had no separate validation set ([Robitaille et al. 2015](https://doi.org/10.1371/journal.pone.0136815)).
- **Hybrid cybernetic EFM model**: captured the lactate shift in batch CHO-S with a mean prediction error below 10%, but was never validated independently ([Martínez et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7174696/)).
- **COSMIC-dFBA**: splits uptake rates into growth-state and production-state contributions on a genome-scale model, and predicts final cell density and titer within 10% ([Gopalakrishnan et al. 2024](https://doi.org/10.1016/j.ymben.2024.02.012)).
- **HybridFBA**: constrains a reduced CHO-K1 genome-scale model with **210 degrees of freedom** using 27 measured exchange fluxes ([Ramos et al. 2022](https://doi.org/10.1007/s00449-022-02795-9)).

One often-cited 2017 poly-pathway EFM paper was **retracted** ([retraction notice](https://doi.org/10.1016/j.jbiotec.2017.11.013)). The column-generation follow-up should be cited instead ([Hagrot et al. 2019](https://doi.org/10.1016/j.mec.2018.e00083)).

Two lessons follow. First, no CHO dFBA or kinetic-metabolic model in this survey was trained and validated on glucose, lactate and cell counts alone. Here their value is as structural priors: stoichiometric yield bounds from a genome-scale model such as iCHO3K (11,004 reactions; [LewisLabUCSD/iCHO3K](https://github.com/LewisLabUCSD/iCHO3K)), and a justification for which macro-reactions to keep.

Second, at this data scale the network and unstructured families converge. An EFM-justified model for this dataset has three reactions, each with Monod or inhibition kinetics:

1. growth: glucose + a lumped nutrient → biomass + lactate
2. overflow: glucose → lactate
3. lactate consumption, switched on late in culture

That comes to roughly 10–15 parameters. Term for term, it is an unstructured Monod model with a lactate switch, and it matches the 13-parameter Multi Clone Kinetic Model (MCKM) fitted to 656 Ambr15 runs ([Sietaram et al. 2025](https://doi.org/10.1093/jimb/kuaf029)). The closest network-derived precedent for the challenge task is Ben Yahia et al. They made μmax a function of integral viable cell density and modelled unmeasured intracellular pools, and the model predicted **new feeding strategies "even outside the range" of the training data** ([Ben Yahia et al. 2021](https://doi.org/10.1016/j.ymben.2021.04.004)).

| Family | Representative work | Size | Data it needed | Out-of-sample evidence | Role in this challenge |
|---|---|---|---|---|---|
| Unstructured Monod–Pirt | Xing 2010; Okamura 2024; MCKM 2025 | 6–8 states, 13–18 parameters | VCD, Glc, Lac (± Gln, NH₄) | Okamura held-out R²: VCD 0.71, Glc 0.83, Lac 0.67, dead cells 0.96 | **Backbone** |
| EFM macro-model | Zamorano 2013; Maton 2022; Ben Yahia 2017/2021 | 2–3 macro-reactions at this data scale | extracellular species | Ben Yahia: new feeds and 2 L → 2000 L rate prediction | Justifies the macro-reactions; same as above here |
| Small kinetic–stoichiometric dFBA | Nolan & Lee 2011; Khare 2025 | 34–35 reactions; 45 parameters | amino acids, ¹³C | partial; fails a lactate spike | Not identifiable |
| Genome-scale dFBA | Ghodba 2025; COSMIC-dFBA 2024 | 205–253 parameters | ~60 metabolites | R² ≥ 0.85–0.97 on 20 runs | Yield bounds only |
| Kinetic-metabolic | Robitaille 2015 | 139 parameters (20 fitted) | intracellular metabolomics | none | Qualitative regulation only |
| Cybernetic EFM | Martínez 2020 | not reported | VCD, dead, Glc, Lac, Gln, Glu (batch) | none | Switch logic |
| Serial hybrid (NN/GP rates) | Pinto 2023; Richelle 2026; Doyle 2026 | 4–5 learned rates | ~20+ runs typical | batch-held-out splits | **Escalation target** |
| Stochastic multi-scale | Wang, Harcum & Xie 2026 | phase-transition SDEs | online qO₂, pH | 1–5-day look-ahead WAPE 5.7–12.1% | Uncertainty design |

## Monod–Pirt mass balances remain the backbone, and three terms decide accuracy

The standard CHO model is built from four pieces ([Ben Yahia et al. 2015](https://doi.org/10.1007/s00253-015-6743-6)):

- Monod saturation in the limiting substrates, multiplied by inhibition terms in the byproducts
- glucose uptake tied to growth through a Pirt law (yield plus maintenance)
- lactate coupled to glucose through a yield
- product formation by Luedeking–Piret

The most-cited instance, Xing et al. 2010, has six ODEs and 18 parameters. Twelve were estimated by MCMC from seven 5000 L runs, and the model was validated on two perturbed runs ([Xing et al. 2010](https://doi.org/10.1002/btpr.284)). A daily-bolus CHO application writes the equations out ([Kornecki & Strube 2018](https://doi.org/10.3390/bioengineering5010025)):

```
μ  = μmax · GLC/(Kglc+GLC) · GLN/(Kgln+GLN) · KIlac/(KIlac+LAC) · KIamm/(KIamm+AMM)
μd = kd · LAC/(KDlac+LAC) · AMM/(KDamm+AMM)
```

Lactate there is purely yield-coupled to glucose uptake.

The case against simpler curve models is direct. One study compared three model types over batch, bolus and continuous fed-batch at 3 L and 15 L. A Monod model predicted cells, glucose, glutamine, lactate and ammonia well. Logistic and regression models failed on nutrients and metabolites because of "the inability of both models to incorporate a feed" ([Craven et al. 2013](https://doi.org/10.1002/btpr.1664)).

Two modern refinements matter for this challenge:

- **Capacity term and unmeasured limiting substrate (Pörtner/Möller school).** A capacity term (Xmax − Xv)/Xmax and an **unmeasured limiting-substrate** state stop growth without the real limiting nutrient being known ([Kuchemüller et al. 2023](https://doi.org/10.1002/elsc.202200059)). Their model-assisted DoE found the same optimum with 4 experiments that classical DoE needed 16 (batch) or 29 (fed-batch) to find ([Möller et al. 2019](https://doi.org/10.1007/s00449-019-02089-7)).
- **Smooth switches (Sartorius 2026).** Monod fractions are replaced by smooth, differentiable switches: [tanh(2z/k)]² for limitation and 1/((z/2k)³ + 1) for inhibition ([Kuchemüller et al. 2026](https://doi.org/10.1007/s00449-026-03362-2); [Richelle et al. 2026](https://doi.org/10.34133/csbj.0078)).

These models usually fail on identifiability, not structure. A Sobol' sensitivity analysis of a dynamic mAb model found that "most parameters have low sensitivity indices and exhibit strong interactions with one another" ([Kontoravdi et al. 2005](https://doi.org/10.1021/bp050028k)). In MCKM ([Sietaram et al. 2025](https://doi.org/10.1093/jimb/kuaf029)):

- the μmax–Kglc collinearity index was 14.33
- the Y_Lac/Glc–Y_X/Lac pair reached 5.8×10⁶
- maintenance m_glc was "not reliably estimated"
- across 157 clones, **μmax was consistent (a "host property") while Kglc varied widely**

Compensation between parameters shows up across papers. One high-density CHO model pairs μmax = 0.27 h⁻¹ with Kglc = 36.7 mM, and its Kglc moved by −61.6% between reactor types ([Okamura et al. 2024](https://doi.org/10.1002/btpr.3486)). An older review warns that pushing K_S toward zero to mimic balanced growth breaks transfer to fed-batch ([Ben Yahia et al. 2015](https://doi.org/10.1007/s00253-015-6743-6)). Bolus-fed glucose usually sits far above any plausible Kglc. The practical rule is therefore to fix Kglc near 1 mM and let μmax carry the growth signal.

The table below gives log-scale bounds. The values are specific to clone and model structure. Do not compare them across papers without the equations they belong to.

| Parameter | Published CHO values | Suggested fit bounds | Sources |
|---|---|---|---|
| μmax | 0.025–0.045 h⁻¹ (0.6–1.1 d⁻¹); MCKM 0.0258 ± 0.0083 h⁻¹ | 0.01–0.06 h⁻¹ (0.24–1.44 d⁻¹) | [Sietaram 2025](https://doi.org/10.1093/jimb/kuaf029); [Kornecki 2018](https://doi.org/10.3390/bioengineering5010025) |
| Death rate k_d | 0.004–0.02 h⁻¹; MCKM campaign means 0.0116–0.0185 h⁻¹ | 10⁻⁴–0.03 h⁻¹ | [Kornecki 2018](https://doi.org/10.3390/bioengineering5010025); [Sietaram 2025](https://doi.org/10.1093/jimb/kuaf029) |
| Dead-cell lysis k_lys | 6.9–8.8×10⁻³ h⁻¹ (≈0.17–0.21 d⁻¹) | 10⁻³–2×10⁻² h⁻¹ | [Xing Z. et al. 2024](https://doi.org/10.1007/s10616-023-00598-8) |
| Kglc | 0.084 mM to ~5.8 mM (36.7 mM in a compensating structure) | fix ≈ 1 mM | [Ben Yahia 2015](https://doi.org/10.1007/s00253-015-6743-6); [Okamura 2024](https://doi.org/10.1002/btpr.3486) |
| Lactate growth inhibition K_I,Lac | 43 mM (Xing-type); 2.7–3.9 mM (Okamura structure) | 5–100 mM | [Kornecki 2018](https://doi.org/10.3390/bioengineering5010025) |
| Y_X/Glc | 0.36–0.49×10⁹ cells/mmol, i.e. 0.36–0.49 (10⁶ cells/mL)/mM (my conversion) | 0.1–1 (10⁶ cells/mL)/mM | [Kornecki 2018](https://doi.org/10.3390/bioengineering5010025); [Okamura 2024](https://doi.org/10.1002/btpr.3486) |
| Y_Lac/Glc | ≈0.57–0.70 mol/mol | 0.3–2 mol/mol | [Sietaram 2025](https://doi.org/10.1093/jimb/kuaf029); [Kornecki 2018](https://doi.org/10.3390/bioengineering5010025) |
| m_Glc | poorly identifiable | fix small or drop | [Sietaram 2025](https://doi.org/10.1093/jimb/kuaf029) |

### Lactate is the worst-predicted state in every recent model

The lactate shift has no single trigger. In NS0 fed-batch it "occurs upon the cessation of rapid growth and under conditions of low glycolysis flux and high extracellular lactate" ([Mulukutla et al. 2012](https://doi.org/10.1016/j.ymben.2011.12.006)). Other studies name different triggers:

- redox imbalance ([Hartley et al. 2018](https://doi.org/10.1002/bit.26603))
- falling glycolytic flux ([Zalai et al. 2015](https://doi.org/10.1002/btpr.2179))
- glutamine depletion at high dissolved oxygen ([Okamura et al. 2024](https://doi.org/10.1002/btpr.3486))
- glucose depletion across 656 Ambr15 runs ([Sietaram et al. 2025](https://doi.org/10.1093/jimb/kuaf029))
- a temperature downshift, in a way that differed by cell line under identical feeding ([Yu et al. 2026](https://doi.org/10.1002/bit.70269))

The mechanistic explanation that matters most for forecasting is **glycolytic bistability**. Lactate inhibition combined with AKT signalling creates a steady-state structure in which the switch "can only occur in certain regions". As a result, the same cell line and medium sometimes shift and sometimes do not ([Mulukutla et al. 2015](https://doi.org/10.1371/journal.pone.0121561)). A deterministic model given only t = 0 and the feeds will therefore carry some error on lactate that cannot be removed.

| Formulation | Trigger encoded | Published performance | Usable with this data? |
|---|---|---|---|
| Pure yield coupling, q_Lac = Y·q_Glc ([Kornecki & Strube 2018](https://doi.org/10.3390/bioengineering5010025)) | none | R² ≥ 0.90 where no strong shift occurred; "cannot follow the shifts" ([Okamura 2024](https://doi.org/10.1002/btpr.3486)) | baseline only |
| Glucose threshold a_glc ([Sietaram et al. 2025](https://doi.org/10.1093/jimb/kuaf029)) | glucose depletion | NRMSE 33.5–44.4% (in-sample fits) | yes, smoothed |
| Time switch after day 5 plus Monod consumption ([Xing Z. et al. 2024](https://doi.org/10.1007/s10616-023-00598-8)) | culture age | descriptive fits | weak for new feed schedules |
| Separate parameter sets before and after day 7.6 ([Kim et al. 2026](https://doi.org/10.1002/bit.70300)) | phase | each batch calibrated separately | weak |
| Sigmoid in glutamine with regime-dependent growth ([Okamura et al. 2024](https://doi.org/10.1002/btpr.3486)) | Gln, DO | **held-out R² 0.674** | no (Gln, DO not measured) |
| Overflow: production = max(0, v_Glc − v_Ox,max) ([Richelle et al. 2026](https://doi.org/10.34133/csbj.0078)) | glycolytic flux vs oxidative capacity | R² 0.36; misses late re-accumulation | yes, one parameter |
| Activation constants Lac_max1/2 ([Yu et al. 2026](https://doi.org/10.1002/bit.70269)) | lactate level | constants "cannot be fully resolved" separately | risky |
| Reversible LDH kinetics in a kinetic–MFA hybrid ([Khare et al. 2026](https://doi.org/10.1038/s41540-026-00703-5)) | mass action | trend right, levels underpredicted | needs network |

No study compares these forms on one dataset, and the MCKM authors state that "no kinetic model in the literature of CHO incorporates an accurate mathematical description of the lactate switch" ([Sietaram et al. 2025](https://doi.org/10.1093/jimb/kuaf029)). Three practical consequences follow for this challenge:

- **Glutamine and DO triggers cannot be used**, because neither is measured.
- **Time-based switches break on new feed schedules.** They encode a calendar rather than a state, so they do not carry over to batches fed differently.
- **A glucose trigger may never fire.** In both close analogues, the substrate feed held glucose above a floor: above 5 g/L in [Richelle et al. 2026](https://doi.org/10.34133/csbj.0078) and at least 3 g/L in [Kim et al. 2026](https://doi.org/10.1002/bit.70300). If the challenge's substrate feed does the same, the switch has to be driven by falling growth (μ/μmax) or by the latent nutrient pool, as Mulukutla's "cessation of rapid growth" suggests.

A smooth Hill switch, H(z) = zⁿ/(K_swⁿ + zⁿ), with z chosen by leave-batch-out validation, keeps gradients well behaved. Hard thresholds do not.

### Lysis is what lets the TCD curve bend

Published death-rate forms fall into five groups:

- **Byproduct-driven**, using Monod factors in lactate and ammonia. Typical values are kd = 0.004 h⁻¹, KDlac = 45.8 mM and KDamm = 6.51 mM ([Kornecki & Strube 2018](https://doi.org/10.3390/bioengineering5010025)).
- **Byproduct plus glucose starvation**, with a starvation constant KDI_glc of 0.58–0.81 mM ([Okamura et al. 2024](https://doi.org/10.1002/btpr.3486)).
- **Ammonia threshold**, with accelerated death above 5 mM ([Xing Z. et al. 2024](https://doi.org/10.1007/s10616-023-00598-8)).
- **Ammonium only**, with mean kd of 0.0116–0.0185 h⁻¹ ([Sietaram et al. 2025](https://doi.org/10.1093/jimb/kuaf029)).
- **Toxicity from accumulated "biomaterial"**, the newest form. Sartorius drives death through μd = kd + k_Td·γ_X, where γ_X integrates viable cells and is diluted by feeds. Lysis is tracked as an internal, unmeasured state, because "trypan blue exclusion does not distinguish between dead and lysed cells" ([Richelle et al. 2026](https://doi.org/10.34133/csbj.0078)). The HEK293 counterpart adds toxicity from lysed cells ([Kuchemüller et al. 2026](https://doi.org/10.1007/s00449-026-03362-2)).

Ammonia is not measured in this challenge, so ammonia-driven forms cannot be identified. The integral-of-viable-cells term is an attractive stand-in because it needs no feed composition.

Lysis is what reproduces TCD. Without a lysis term, dead cells only accumulate and TCD can only rise, which typically fails late in culture. Three formulations are published:

- **First-order lysis on Xt − Xv** ([Kuchemüller et al. 2023](https://doi.org/10.1002/elsc.202200059)).
- **Michaelis–Menten lysis of dead cells.** This gave a **dead-cell prediction R² of 0.955 on an unseen run** ([Okamura et al. 2024](https://doi.org/10.1002/btpr.3486)).
- **Two lysis routes, from dead cells and directly from viable cells.** Fitted rates were kDL = 6.9–8.8×10⁻³ h⁻¹ and kL = 1.3–1.4×10⁻³ h⁻¹, and both viable and dead densities were reproduced with **R² > 0.95** ([Xing Z. et al. 2024](https://doi.org/10.1007/s10616-023-00598-8)).

One design choice matters. **Additive death terms** (k_d0 + term₁ + term₂) are more robust than products of Monod factors, because a product forces death to zero whenever one driver is absent.

### Feed bookkeeping beats kinetic sophistication

The single most damaging error in the recent literature was feeding the model the wrong feeds. MCKM applied one run's feeding schedule to all runs, and glucose fits collapsed to **R² = −0.24 ± 0.48 with NRMSE ≈ 50%** for one product line. The authors list "averaged feed amounts" as a limitation ([Sietaram et al. 2025](https://doi.org/10.1093/jimb/kuaf029)). Unknown feed composition causes the same failure more quietly. A glucose misalignment at 32 °C was attributed to the uncharacterised proprietary "Feed X" that the filter did not represent ([Yu et al. 2026](https://doi.org/10.1002/bit.70269)). Bolus-feed transients remain "a challenge requiring model-structure work" even in a 2026 GP hybrid ([Doyle et al. 2026](https://doi.org/10.1002/bit.70339)). Kornecki & Strube likewise found feed-phase glucose "significantly more challenging" because physiological effects of the feed medium were not modelled ([Kornecki & Strube 2018](https://doi.org/10.3390/bioengineering5010025)).

Published balances take three forms:

- balances on total amounts, d(V·C)/dt ([Okamura et al. 2024](https://doi.org/10.1002/btpr.3486))
- concentration form with explicit dilution ([Kuchemüller et al. 2026](https://doi.org/10.1007/s00449-026-03362-2))
- impulse terms V_F,i·c_F/V at each bolus ([Xing Z. et al. 2024](https://doi.org/10.1007/s10616-023-00598-8))

All three amount to the same exact jump map: integrate between events, then at each bolus update the volume, mix the incoming feed into every dissolved species, and dilute cells by V⁻/V⁺. Sampling removes volume but leaves concentrations unchanged. Sampling convention matters, because VCD shows sawtooth drops at every feed. Wang et al. sampled "prior to feeding" ([Wang, Harcum & Xie 2026](https://doi.org/10.1002/bit.70205)), and the model must use the same ordering.

For feeds of unknown composition, three published devices exist:

- **A lumped unmeasured limiting substrate**, consumed by cells and entering μ through a Monod term ([Kuchemüller et al. 2023](https://doi.org/10.1002/elsc.202200059)). This goes back to serum-Monod models of hybridoma growth ([Ben Yahia et al. 2015](https://doi.org/10.1007/s00253-015-6743-6)).
- **A catch-all accumulated biomaterial** that inhibits growth or drives death ([Richelle et al. 2026](https://doi.org/10.34133/csbj.0078)).
- **Unmeasured intracellular pools**, with μmax made a function of integral viable cell density ([Ben Yahia et al. 2021](https://doi.org/10.1016/j.ymben.2021.04.004)).

Extending the first device gives the construction recommended below. It is an inference, not a published, validated CHO model. A dimensionless nutrient pool N starts at 1 (basal medium) and is consumed in proportion to growth. Each bolus of feed j adds γ_j·ΔV_j/V⁺, where γ_j is that feed's "richness" relative to basal medium. The three fitted γ values then become process insight: they estimate **how much growth capacity each unknown feed delivers per millilitre**. That is exactly the kind of interpretable output the challenge rewards.

## Serial hybrids beat mechanistic models wherever the two have been compared

The serial hybrid keeps the mass balances and replaces specific rates with a data-driven function of the current state. The idea dates to Psichogios & Ungar ([1992](https://doi.org/10.1002/aic.690381003)) and was systematised by von Stosch et al. ([2014](https://doi.org/10.1016/j.compchemeng.2013.08.008)). For this challenge the key early result is Narayanan et al. Their hybrids predicted the time evolution of CHO fed-batch variables **"using only the initial and process conditions"**, the exact hackathon setting, more accurately and robustly than statistical models ([Narayanan et al. 2019](https://doi.org/10.1002/bit.27097)).

| Study | Data | Hybrid structure | Held-out result |
|---|---|---|---|
| [Bayer et al. 2021](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2021.740215/full) | 18 shake-flask conditions plus 11 × 15 L CHO runs | one hidden layer, **4 tanh neurons** → μ, q_P | NRMSE: VCC 10.9–13.8%, titer 17.8–21.1% |
| [Bayer et al. 2023](https://doi.org/10.1002/biot.202200381) | 33 CHO fed-batch runs, 4-factor design | serial hybrid vs mechanistic | hybrid more accurate on all 4 partitions; mechanistic more robust to the partition |
| [Pinto et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10515724/) | 24 CHO-K1 Ambr250 runs, 30 species | 3 × 25 ReLU (2,855 weights) → 30 rates | test WMSE 1.88 vs 2.46 for the shallow hybrid (**−23.6%**) |
| [Ramos et al. 2024](https://doi.org/10.1002/bit.28668) | 20 HEK293 runs, 27 species | LSTM (7 states) in series with balances | test R² 0.93 |
| [Richelle et al. 2026](https://doi.org/10.34133/csbj.0078) | 23 CHO-S Ambr15 runs, 8 feed formulations | NN μ (sigmoid baseline + learned deviation) + kinetic ODE + PCA-constrained dFBA | leave-batch-out ≈ direct validation; lactate R² 0.36 in the ODE layer |
| [Doyle et al. 2026](https://doi.org/10.1002/bit.70339) | 2 intensified CHO processes | GP rates inside mass balances | beats a Monod model; bolus transients unresolved |
| [Wang, Harcum & Xie 2026](https://doi.org/10.1002/bit.70205) | 9 CHO-K1 ambr250 runs | stochastic multi-scale, probabilistic phase transitions | WAPE 5.7–12.1% (1–5-day look-ahead); 95% interval coverage 91.5–94.3% |

The pattern is consistent. Mechanistic models carry prior knowledge "relatively independently of the utilized data partition". Hybrids are "more data dependent" but "yielded a higher accuracy on all data partitions" ([Bayer et al. 2023](https://doi.org/10.1002/biot.202200381)).

The strongest CHO results share a training recipe:

- integrate whole trajectories from the initial state
- use a weighted least-squares loss scaled by measurement variance
- regularize with few hidden units, dropout or early stopping
- split train and test by whole batch, never by time point

Pinto et al. trained the deep model with Adam, minibatching, 0.2 weight dropout and sensitivity-equation gradients ([Pinto et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10515724/)). An earlier general deep-hybrid framework reported **18.4% better accuracy and 43.4% less CPU time** than shallow hybrids on a *Pichia* process ([Pinto et al. 2022](https://novaresearch.unl.pt/en/publications/a-general-deep-hybrid-model-for-bioreactor-systems-combining-firs/)).

Three cautions keep expectations realistic:

- **The train–test gap is real.** In Pinto et al. the test WMSE was about 40% above training (1.88 vs 1.35).
- **Depth may pay off less here.** The deep-vs-shallow gain came with 30 measured species. With four or five species the benefit of depth will be smaller and the overfitting risk larger; no study has tested this, so it is an inference.
- **Hybrids interpolate; they do not extrapolate.** A constrained UDE-style hybrid trained on only three microbial batches reached test R² of 0.9–0.99. It produced **negative R² for glucose and acetate when extrapolated to a lower initial glucose** ([Martínez et al. 2025](https://www.biorxiv.org/content/10.1101/2025.11.14.688550.full)).

No study benchmarks a pure LSTM or transformer against a hybrid on held-out CHO runs with t = 0-only inputs. Claims that black-box sequence models cannot compete rest on the related evidence above, not on a direct test.

### Neural ODEs are serial hybrids under another name; vanilla PINNs fit this task poorly

Universal differential equations (UDEs) put neural networks inside an ODE right-hand side and train them through the solver ([Rackauckas et al. 2020](https://arxiv.org/abs/2001.04385)). This is the same model class as the hybrid semi-parametric models above; a Waterloo thesis notes that "hybrid neural differential equation" and UDE name one thing ([Menard thesis](https://uwaterloo.ca/computational-mathematics/sites/default/files/uploads/documents/menard_jyler_0.pdf)). A 2025 review identifies stiff dynamics and sparse, noisy data as the main training obstacles ([npj Syst Biol Appl 2025](https://www.nature.com/articles/s41540-025-00550-w)). Recent identifiability work warns that learned NN components may not be functionally identifiable, so plots of learned rates need care in interpretation ([arXiv 2510.14140](https://arxiv.org/pdf/2510.14140)).

The bioprocess UDE evidence is mostly microbial. A batch-fermentation UDE performed **3.6-fold and 1.7-fold better** than its parametric counterpart ([Herrera-Ruiz et al. 2025](https://onlinelibrary.wiley.com/doi/full/10.1002/eng2.70502)). For CHO, the main example is a grey-box model that embeds the metabolic flux-balance step as a differentiable convex-optimization layer ([Cui et al. 2023](https://arxiv.org/abs/2305.03257)). No accessible CHO UDE paper reports held-out-batch numbers.

Physics-informed neural networks (PINNs) for CHO exist, for example to fuse small-scale mechanistic knowledge with pilot-scale data ([Yang et al. 2024](https://par.nsf.gov/servlets/purl/10537019)) and for physics-encoded FiLM transfer across scales ([Li et al. 2026](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.70542)). Their held-out accuracy was behind paywalls. A vanilla collocation PINN learns one time-indexed trajectory, so it needs extra conditioning on the initial state and the feeds to generalise across batches. An ODE-integrated hybrid handles both naturally, and handles boluses by restarting the integration at each feed.

### Symbolic regression distils hybrids; it does not replace them

Plain polynomial SINDy suits bioprocess kinetics poorly. A coculture study reports that standard libraries "fail to capture saturation kinetics and metabolic switching" ([PLOS Comput Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1014759)). Implicit SINDy extends the method to rational terms and recovers Michaelis–Menten kinetics ([Mangan et al.](https://arxiv.org/pdf/1605.08368)). Derivative-based SINDy is also noise- and sampling-sensitive, which hurts with daily samples broken up by bolus jumps ([Menard thesis](https://uwaterloo.ca/computational-mathematics/sites/default/files/uploads/documents/menard_jyler_0.pdf)). The robust workflow on noisy biological data is two-step: fit a neural or hybrid model first, then run sparse regression on its smooth outputs ([PMC11753677](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11753677/)).

The bioprocess-specific line puts symbolic regression inside a hybrid backbone:

- **Functional-Hybrid models** draw on a library of Monod and inhibition "functional beliefs". They matched ANN hybrids in interpolation and gave **superior extrapolation** ([Narayanan et al. 2022](https://www.sciencedirect.com/science/article/abs/pii/S1385894721046088)).
- **Two-step regression** that avoids repeated integration "slightly outperformed" NN benchmarks ([Forster et al. 2024](https://dau.url.edu/handle/20.500.14342/4620)).
- **Time-varying parameter profiles followed by symbolic regression** proved data-efficient in silico ([Riezzo et al. 2026](https://doi.org/10.1002/bit.70328)).
- **LLM-driven agents** recovered Haldane and autocatalytic terms with under 6% parameter error, on a simulated reactor only ([Aydin & Arkun 2026](https://arxiv.org/abs/2610.35391)).

All of these were validated in silico or on microbial systems. None has CHO held-out benchmarks. For this challenge, symbolic regression belongs in the final interpretability step: distil the learned μ and q_Lac into Monod/Haldane expressions, then check that the distilled model loses little held-out accuracy.

## Batch-to-batch variability is the least-solved part of the problem

Most published "validation" is per-run refitting, which cannot forecast a new batch:

- MCKM fits all 13 parameters to each run separately ([Sietaram et al. 2025](https://doi.org/10.1093/jimb/kuaf029)).
- An LLM-guided optimisation study calibrated each of 8 Ambr15 batches independently ([Kim et al. 2026](https://doi.org/10.1002/bit.70300)).
- An ensemble Kalman filter transferred a 24-parameter CHO model across cell lines, scales and feeds "without reparameterization". It did so by assimilating each new run's own measurements, with open-loop forecasts starting at days 7, 9 and 12 ([Yu et al. 2026](https://doi.org/10.1002/bit.70269)). That does not apply when only t = 0 is known.

True out-of-sample results are rarer:

- Okamura's model fitted on one experiment and applied to another gave **VCD R² 0.71, glucose 0.83, lactate 0.67 and dead cells 0.96** ([Okamura et al. 2024](https://doi.org/10.1002/btpr.3486)).
- Sartorius' leave-one-batch-out and leave-one-media-group-out generalisation "remained comparable" to direct validation ([Richelle et al. 2026](https://doi.org/10.34133/csbj.0078)).

Treat these numbers as realistic targets, and treat in-sample figures such as MCKM's 6–7% biomass NRMSE as optimistic.

**No CHO paper was found that applies formal hierarchical Bayesian or nonlinear mixed-effects (NLME) ODE kinetics.** Neighbouring fields have the tools:

- hierarchical Monod ODEs across replicate microbial batches, with the Python packages calibr8 and murefi ([Helleckes et al. 2022](https://doi.org/10.1371/journal.pcbi.1009223))
- mixed-effects growth curves showing that ignoring batch effects biases estimates ([Tonner et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7644099/))
- NLME separating population from batch effects in transfection kinetics ([Fröhlich et al. 2018](https://www.nature.com/articles/s41540-018-0079-7))

Two CHO-adjacent ideas point the same way. Joint "multi-cell-line learning" of shared mechanistic parameters improved accuracy with limited data ([Lu et al. 2024](https://doi.org/10.1002/bit.28757)). Learned entity-embedding vectors fed to a GP hybrid act as the machine-learning analogue of random effects ([Hutter et al. 2021](https://arxiv.org/abs/2011.13863)).

Be precise about what hierarchy buys when a holdout batch reveals only its t = 0 state. It cannot estimate that batch's random effect. It does three other things:

1. **Better population parameters**, by shrinking noisy per-batch estimates.
2. **Calibrated predictive intervals**, by integrating over the random-effect distribution. Wang et al. show that calibrated coverage of 91.5–94.3% is achievable in CHO ([Wang, Harcum & Xie 2026](https://doi.org/10.1002/bit.70205)).
3. **A covariate model.** If t = 0 viability, initial glucose or total feed volumes predict a batch's growth or death parameters, they move the forecast for a new batch.

Together with the lactate bistability above, this argues for reporting lactate with uncertainty bands rather than as a sharp curve. Prediction semantics also matter. For a skewed output such as VCD, the trajectory at the mean parameters is not the mean trajectory. Report the predictive median under absolute-error metrics and the predictive mean under squared-error metrics.

## A two-day plan from a ten-parameter baseline to a hybrid

The recommendation below combines the most transferable published elements into one structure. It is a synthesis and must be validated on the challenge data.

### The recommended model in equations

States per batch:

- Xv: viable cells, 10⁶ cells/mL
- Xd: dead but unlysed cells, Xd = TCD − VCD
- S: substrate, mM
- L: lactate, mM
- N: latent nutrient pool, dimensionless, N(0) = 1
- I: integral of viable cells ("biomaterial/age")
- V: volume

Between events, V is held at the last measured value:

```
dXv/dt = (μ − μd)·Xv
dXd/dt =  μd·Xv − k_lys·Xd                 TCD = Xv + Xd   (lysed cells leave the count)
dS/dt  = −q_S·Xv
dL/dt  =  q_L·Xv
dN/dt  = −(μ / Y_XN)·Xv
dI/dt  =  Xv

μ    = μmax · S/(K_S+S) · N/(K_N+N) · K_IL/(K_IL+L)
μd   = k_d0 + k_dL·L/(K_dL+L) + k_dN·K_dN/(K_dN+N) + k_T·I        (additive; prune by CV)
q_S  = μ/Y_XS + m_S·S/(K_S+S)                                   (Pirt; maintenance vanishes at S=0)
H    = zⁿ/(K_swⁿ + zⁿ),  z ∈ {S, μ/μmax, N}                      (smooth lactate switch)
q_L  = Y_LS·q_S·H − q_Lmax·L/(K_L+L)·(1 − H)                     (production → consumption)

Bolus at t_k with volumes ΔV_S, ΔV_B, ΔV_C, ΔV_D:
V⁺ = V⁻ + ΣΔV_j
S⁺ = (S⁻V⁻ + c_S·ΔV_S [+ Σ c_S,j·ΔV_j]) / V⁺
N⁺ = (N⁻V⁻ + γ_B·ΔV_B + γ_C·ΔV_C + γ_D·ΔV_D) / V⁺
Xv⁺, Xd⁺, L⁺, I⁺ = (value)⁻ · V⁻/V⁺
Sample or volume stamp: V ← V_measured; concentrations unchanged.
```

Some parameters are fixed to keep the model identifiable:

- K_S ≈ 1 mM
- Hill exponent n ≈ 2–4
- K_L and K_dL at plausible literature values
- N(0) = 1, because Y_XN and the γ_j otherwise trade off against the initial pool
- at most one of the two lactate yields left free

That leaves about **8 free parameters** in the core:

- μmax, K_IL
- k_d0, k_dL, k_lys
- Y_XS, m_S, Y_LS

The switch adds 2 (K_sw, q_Lmax) and the nutrient pool adds 5–6 (Y_XN, K_N, γ_B, γ_C, γ_D, k_dN). The total of 15–16 is at the practical ceiling, so prune terms that do not improve leave-batch-out error.

One identifiability trap is specific to this dataset, and the check below is my inference. **If feeds B, C and D are always added in fixed volume ratios across batches, their γ values cannot be separated.** Check the rank of the per-batch feed-volume matrix and merge γ values where the columns are collinear.

### Replay every bolus exactly and learn what feeds B, C and D are worth

Settle the event semantics before any fitting. Order events that share a timestamp as measure → remove sample → feed, unless the data say otherwise. Then:

- **Volume.** Snap V to every measured volume stamp. This absorbs sampling, evaporation and unlogged additions.
- **Substrate feed.** Use the known concentration c_S if it is given. Otherwise estimate it from pre- and post-feed substrate jumps as c_S ≈ (S⁺V⁺ − S⁻V⁻)/ΔV_S and fix it.
- **Glucose in B, C and D.** If glucose jumps at times when only B, C or D is fed, give that feed its own glucose concentration c_S,j. Otherwise set it to zero.
- **Inhibitory feeds.** If a feed looks inhibitory (for example through osmolality), add a separate inhibitor state with its own unknown per-volume loading instead of overloading N.

Never let an adaptive solver "find" a bolus by itself. Solvers step over dose events unless stop times are declared ([AMICI docs](https://amici.readthedocs.io/en/latest/implementation_discontinuities.html); [Julia Discourse](https://discourse.julialang.org/t/implementing-dosing-regimens-in-diffeq/43173)). In diffrax, `jump_ts` handles jumps in the vector field, not jumps in the state, so state jumps require restarting the solve ([diffrax docs](https://docs.kidger.site/diffrax/api/stepsize_controller/)). Undeclared jumps have produced wrong gradients ([diffrax #363](https://www.github.com/patrick-kidger/diffrax/issues/363)). If the hybrid shows systematic post-feed transients of the kind Doyle et al. flagged, add "time since last feed" as an input to the network.

### Fit population parameters in log space, warm-start, then refine open-loop

Fitting runs in five steps:

1. **Fit one shared parameter vector to all training batches.** Use parameters in log space within bounds two to three decades wide.
2. **Weight the residuals.** For VCD and TCD use log-scale residuals with σ ≈ 0.1–0.2. For substrate and lactate use absolute residuals scaled by a per-species σ. Normalise per batch so densely sampled runs do not dominate, and use a soft-L1 loss for outliers.
3. **Multi-start the optimiser.** Run 20–50 Latin-hypercube starts of `scipy.optimize.least_squares` (TRF method, `x_scale="jac"`), optionally seeded by CMA-ES. A large benchmark found multi-start gradient-based local optimisation often succeeds on these problems ([Villaverde et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6394396)), and calibration is typically multimodal ([Villaverde et al. protocol](https://arxiv.org/abs/2105.12008v2)).
4. **Warm-start if fits stall.** Do a "teacher-forced" fit that resets states to measured values at each sample, or a multiple-shooting fit (CasADi with the jump map in the continuity constraints).
5. **Refine with full single shooting from t = 0.** The holdout forecast is open-loop from t = 0, so the final fit must be too.

Check identifiability at the optimum. Look at the singular values of the residual Jacobian and at pairwise collinearity, using MCKM's index threshold of 20 ([Sietaram et al. 2025](https://doi.org/10.1093/jimb/kuaf029)), and run a quick profile likelihood on three to five key parameters.

Only after this, add random effects on two or three parameters: μmax, k_d0 or k_T, and K_sw or Y_XS. Fit them as a MAP penalty Σ ηᵀΩ⁻¹η, or with NumPyro SVI. Then regress the per-batch estimates η̂ on one or two t = 0 or feed-total covariates, inside the cross-validation folds.

### Validate by leaving whole batches out and beating two baselines

Use leave-one-batch-out or `GroupKFold` grouped by batch, with the entire pipeline (including any covariate regression) inside each fold. Score per species on the hackathon's own metric and timestamps.

Every candidate must beat two baselines:

- **Mechanistic baseline M0**, to show that each added term earns its keep.
- **Non-mechanistic regressor**, for example gradient boosting on time, the t = 0 state, cumulative feed amounts and current volume. This shows whether the mechanistic structure adds value at all. The same regressor can later serve as a residual corrector.

Before trusting any hybrid, check whether the holdout feed profiles and initial states lie inside the training envelope, given the extrapolation failure documented by [Martínez et al. 2025](https://www.biorxiv.org/content/10.1101/2025.11.14.688550.full).

Published held-out levels give a calibration point: roughly **10–14% NRMSE on viable cells** ([Bayer et al. 2021](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2021.740215/full)) and **R² ≈ 0.7 / 0.83 / 0.67 / 0.96 for VCD, glucose, lactate and dead cells** ([Okamura et al. 2024](https://doi.org/10.1002/btpr.3486)).

### Escalation ladder for a 1–2 day sprint

| Step | Slot | What it adds | Free params (approx.) | Evidence for it | Keep if (leave-batch-out) |
|---|---|---|---|---|---|
| B0 | Day 1, hour 1–2 | Gradient-boosting baseline on t, t=0 state, cumulative feeds, V | n/a | logistic/regression models cannot structurally use feeds ([Craven 2013](https://doi.org/10.1002/btpr.1664)) | always reported |
| M0 | Day 1 morning | Event-exact balances; Monod growth with lactate inhibition; Pirt uptake; yield-coupled lactate; constant + lactate death; first-order lysis | ~8 | [Kornecki 2018](https://doi.org/10.3390/bioengineering5010025); [Sietaram 2025](https://doi.org/10.1093/jimb/kuaf029) | must beat B0 on VCD/TCD/S |
| M1 | Day 1 afternoon | Smooth lactate switch; compare triggers S, μ/μmax, overflow | +2 | lactate is the weakest state everywhere | lactate error falls |
| M2 | Day 1 afternoon | Latent nutrient N with γ_B, γ_C, γ_D (+ starvation death) | +5–6 | [Kuchemüller 2023](https://doi.org/10.1002/elsc.202200059); [Ben Yahia 2021](https://doi.org/10.1016/j.ymben.2021.04.004) | VCD peak/decline improves |
| M3 | Day 1 evening | Biomaterial/age toxicity I in death (or growth) | +1–2 | [Richelle 2026](https://doi.org/10.34133/csbj.0078); [Kuchemüller 2026](https://doi.org/10.1007/s00449-026-03362-2) | late-phase VCD/TCD improves |
| M4 | Day 1 evening | Random effects on 2–3 parameters + 1–2 covariates; predictive intervals | +Ω | [Helleckes 2022](https://doi.org/10.1371/journal.pcbi.1009223); [Lu 2024](https://doi.org/10.1002/bit.28757) | mean error does not worsen; intervals calibrated |
| H1 | Day 2 morning | Small MLP (1–2 layers × 8–16 units) as μ = μ_mech·exp(NN_μ) and q_L = q_L,mech + NN_L; inputs S, L, Xv, N (or cumulative B/C/D per volume), I | +100–300 weights | [Bayer 2023](https://doi.org/10.1002/biot.202200381); [Pinto 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10515724/); [Richelle 2026](https://doi.org/10.34133/csbj.0078) | beats best M-model by more than the fold-to-fold spread |
| H1′ | Day 2 (alternative) | GP instead of MLP for the rate corrections | — | [Doyle 2026](https://doi.org/10.1002/bit.70339) | same |
| H2 | Day 2 afternoon | Batch embeddings set to the population mean for holdouts | +2–3 per batch | [Hutter 2021](https://arxiv.org/abs/2011.13863) | only with clear gain |
| H3 | Day 2 afternoon | Symbolic distillation of learned μ and q_L into Monod/Haldane terms | back to ~M-size | [Narayanan 2022](https://www.sciencedirect.com/science/article/abs/pii/S1385894721046088) | distilled model stays close to H1 held-out |

For H1, three design choices keep the hybrid tied to the mechanistic model:

- Initialise the network's output layer at zero, so the hybrid starts exactly at the mechanistic solution.
- Penalise the size of the correction (L2), so it shrinks toward mechanism wherever the data are silent.
- Train an ensemble of 5–10 seeds with early stopping on an inner batch split.

This "mechanistic baseline plus learned deviation" design is how the Sartorius twin parameterises μ ([Richelle et al. 2026](https://doi.org/10.34133/csbj.0078)). It also preserves interpretability: the correction is a visible, plottable function of the state. Keep q_L free to change sign so that the lactate switch can be learned. Constrain μ and μd to be non-negative with softplus or exponential maps.

### Python stack and a simulator skeleton

| Need | Recommended tool | Status (PyPI, Nov 2026) | Caveat |
|---|---|---|---|
| Baseline simulation and fit | SciPy `solve_ivp` (LSODA/BDF) + `least_squares` | [scipy 1.18.1](https://pypi.org/project/scipy/) | no impulsive jumps; restart at each event ([docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)) |
| Global starting points | pycma (CMA-ES), `differential_evolution` | [cma 4.6.0](https://pypi.org/project/cma/) | for ≤ ~20 parameters |
| Differentiable simulation, `vmap`, hybrids | JAX + diffrax + equinox + optimistix/optax | [diffrax 0.7.2](https://pypi.org/project/diffrax/), [equinox 0.13.8](https://pypi.org/project/equinox/), [optimistix 0.1.0](https://pypi.org/project/optimistix/) | state jumps need segment restarts |
| Hierarchical Bayes | NumPyro on the JAX simulator; or PyMC + sunode | [numpyro 0.22.0](https://pypi.org/project/numpyro/); [sunode](https://github.com/pymc-devs/sunode) | sunode is conda-forge only |
| PyTorch alternative | torchode | [torchode 1.2.0](https://pypi.org/project/torchode/) | torchdiffeq last released Nov 2024 |
| Multiple shooting | CasADi + IPOPT | [casadi 3.8.2](https://pypi.org/project/casadi/) | jump map goes in the continuity constraints |
| Profile likelihood, multistart tooling | pyPESTO + AMICI + PEtab | [pypesto 0.8.0](https://pypi.org/project/pypesto/) | AMICI's PEtab v2 importer supports ≤ 2 periods ([source](https://amici.readthedocs.io/en/latest/_modules/amici/importers/petab/_petab_importer.html)) |
| Symbolic distillation | PySINDy | [pysindy 2.1.0](https://pypi.org/project/pysindy/) | use rational/Monod libraries |
| Stoichiometric yield bounds | COBRApy + iCHO3K | [cobrapy](https://github.com/opencobra/cobrapy); [iCHO3K](https://github.com/LewisLabUCSD/iCHO3K) | prior only |
| Avoid as primary | pyFOOMB, KIPET, SBML2HYB | Python ≤ 3.9 / last release 2022 / 2023 ([pyFOOMB](https://github.com/MicroPhen/pyFOOMB); [kipet](https://pypi.org/project/kipet/)) | stale dependencies |

Useful reference code includes the Wang/Harcum/Xie multi-scale CHO hybrid ([GitHub](https://github.com/kw48792/Multi-Scale-Hybrid-Model-with-Metabolic-Phase-Transitions)) and the Martínez UDE-style hybrid ([GitHub](https://github.com/carlosmvond/hybrid-modelling-of-batch-bioreactors)). The public AstraZeneca collection of 106 CHO cultures (5–500 L, at least 25 offline variables over up to 17 days) is a sanity-check dataset if units and feed records can be harmonised ([arXiv 2512.03460](https://arxiv.org/html/2512.03460)). Julia's SciML stack has the most mature combination of event handling and adjoints, but compile latency makes it worthwhile only if someone on the team already uses it.

Compute is not the bottleneck. A rough estimate (not measured) is 0.2–1 s per SciPy objective evaluation over 40 batches, and 10–40 minutes for 20 multistarts on 8 cores. JAX with `vmap` and autodiff Jacobians should be one to two orders of magnitude faster per iteration. The 2,855-weight CHO deep hybrid took about 3 CPU-hours in MATLAB ([Pinto et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10515724/)).

The core of the simulator is an event loop:

```python
import numpy as np
from scipy.integrate import solve_ivp

def rhs(t, y, p):
    Xv, Xd, S, L, N, I = np.maximum(y, 0.0)                 # 1e6 cells/mL, mM, dimensionless
    mu = p["mu_max"] * S/(p["K_S"]+S) * N/(p["K_N"]+N) * p["K_IL"]/(p["K_IL"]+L)
    H  = S**4 / (p["K_sw"]**4 + S**4)                        # swap trigger to mu/mu_max or N if S stays high
    kd = p["kd0"] + p["kdL"]*L/(p["K_dL"]+L) + p["kdN"]*p["K_dN"]/(p["K_dN"]+N) + p["kT"]*I
    qS = mu/p["Y_XS"] + p["m_S"]*S/(p["K_S"]+S)
    qL = p["Y_LS"]*qS*H - p["qL_max"]*L/(p["K_L"]+L)*(1.0 - H)
    return [(mu-kd)*Xv, kd*Xv - p["k_lys"]*Xd, -qS*Xv, qL*Xv, -mu/p["Y_XN"]*Xv, Xv]

def simulate(p, y0, V0, events, t_obs):
    """events: time-sorted (t, kind, payload); kind 'vol' -> measured V, 'feed' -> {feed_name: added volume}.
    Observations at an event time are recorded BEFORE the event (pre-feed sampling)."""
    y, V, t, out, t_obs = np.asarray(y0, float), V0, 0.0, {}, set(t_obs)
    for tb in sorted({e[0] for e in events} | t_obs):
        if tb > t:
            sol = solve_ivp(rhs, (t, tb), y, args=(p,), method="LSODA", rtol=1e-6, atol=1e-8)
            if not sol.success:
                return None                                   # caller converts to large finite residuals
            y, t = sol.y[:, -1], tb
        if tb in t_obs:
            out[tb] = y.copy()
        for _, kind, val in (e for e in events if e[0] == tb):
            if kind == "vol":
                V = val                                       # check whether stamps are pre- or post-feed
            else:
                Vn = V + sum(val.values())
                add = np.zeros_like(y)
                add[2] = p["c_S"] * val.get("S", 0.0)         # glucose from the substrate feed
                add[4] = sum(p["gamma_" + j] * val.get(j, 0.0) for j in "BCD")
                y, V = (y * V + add) / Vn, Vn                 # dilutes Xv, Xd, S, L, N, I; adds feed content
    return out
```

Rescale states to order-one units, clip negatives inside the right-hand side, and return large finite residuals when a solve fails, so that optimisers keep moving.

For process insight, show the judges:

- a parameter table with profile-likelihood intervals
- the three γ values as the relative nutrient value of feeds B, C and D
- the fitted lactate-switch trigger and threshold
- a decomposition of the death rate over time into its constant, lactate, starvation and age terms
- partial-dependence plots of any learned rate correction, alongside its distilled closed form

## Conclusion

The research changes the question. "Which kinetic model is best?" matters less than "how do we spend 15 identifiable parameters?" Spent on network detail, they buy nothing that four measured species can confirm. Spent on exact feed accounting, a lysis term, a state-triggered lactate switch and a per-feed nutrient loading, they buy the documented accuracy gains and a model whose parameters are themselves process findings. The hybrid step is worth taking, because every comparison favours it. It should be taken as a constrained correction to a validated mechanistic core, not as a replacement, because the mechanistic core is what stays robust when the training set is small or the holdout batches sit at the edge of the design space.

Two open problems are also opportunities for the team. First, no published CHO model estimates the nutrient value of several unknown-composition feeds as latent parameters. Second, none treats batch-to-batch variability with a hierarchical ODE model and calibrated intervals. A hackathon entry that does both, and shows under leave-batch-out validation where a learned rate correction actually helps, would be close to the 2026 state of the art rather than behind it. Lactate is the exception that calls for humility. Glycolytic bistability means part of its batch-to-batch variation cannot be predicted from t = 0 and the feeds alone. Lactate predictions are best presented with uncertainty bands, and judged with that ceiling in mind.
