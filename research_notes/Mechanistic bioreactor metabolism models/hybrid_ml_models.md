# Hybrid Mechanistic / Machine-Learning (Grey-Box) Models of CHO Fed-Batch Bioprocesses (survey, focus 2022–2026)

Scope note for the report writer: the application is a small dataset (dozens of ~10–14 day fed-batch runs) with offline VCD, TCD, one substrate (glucose), one byproduct (lactate) and volume, plus 4 bolus feeds (glucose + 3 feeds of unknown composition). The task is to predict whole trajectories of unseen batches from t=0 measurements plus the feed schedule. Where a paper's numbers could not be checked against the full text (paywalls: Wiley/ACS/AIChE returned HTTP 403), this is flagged. Items marked "[background, not re-fetched]" are classic or foundational references whose bibliographic details come from prior knowledge rather than a page retrieved in this session.

---

## 1. Hybrid semi-parametric models (mass-balance ODE backbone + data-driven specific rates): structures, NN inputs/outputs, training, CHO results

### Takeaway
The dominant and best-validated grey-box design for CHO fed-batch is the **serial hybrid**. Material balances for VCD, metabolites, product and volume, with dilution and feed terms, are integrated in time. A small data-driven model (FFNN, GP, PLS, or LSTM) maps the current state, and optionally process parameters, to the **cell-specific rates** (μ, death, q_Glc, q_Lac, q_P). The 2022–2026 frontier has moved from shallow single-hidden-layer tanh networks trained with Levenberg–Marquardt to **"deep hybrid"** models: 2–3 ReLU layers or LSTMs trained with ADAM, minibatching, dropout and sensitivity-equation gradients. In a 24-run CHO-K1 study (Pinto et al. 2023) these cut test error by about 24% relative to shallow hybrids.

### Cited Findings
**Foundational (pre-2022)**
- Psichogios & Ungar (1992) introduced the serial "hybrid neural network–first principles" model: an NN estimates unmeasured kinetic parameters or rates inside a mass-balance model. Citation: AIChE J 38:1499–1511, DOI 10.1002/aic.690381003 [background, not re-fetched] — [DOI](https://doi.org/10.1002/aic.690381003)
- von Stosch, Oliveira, Peres, Feyo de Azevedo (2014), "Hybrid semi-parametric modeling in process systems engineering: past, present and future", Comput Chem Eng 60:86–101, DOI 10.1016/j.compchemeng.2013.08.008. This is the canonical review of serial vs parallel structures [background, not re-fetched] — [DOI](https://doi.org/10.1016/j.compchemeng.2013.08.008)
- Narayanan, Sokolov, Morbidelli, Butté (2019), "A new generation of predictive models: The added value of hybrid models for manufacturing processes of therapeutic proteins", Biotechnol Bioeng 116(10):2540–2549, DOI 10.1002/bit.27097. It compared hybrid models quantitatively with state-of-the-art statistical models on 3.5 L fed-batch data. The hybrid models predicted time evolution of process variables **using only initial and process conditions** (exactly the hackathon setting), and were more accurate and robust and extrapolated better — [Europe PMC record](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22new%20generation%20of%20predictive%20models%22%20AND%20AUTH%3A%22Narayanan%22&resultType=core&format=json)
- Cruz-Bournazou, Narayanan, Fagnani, Butté (DataHow), "Hybrid Gaussian Process Models for continuous time series in bolus fed-batch cultures", bioRxiv 2021, DOI 10.1101/2021.12.27.474269. It is directly relevant to **bolus feeds**:
  - It uses a "continuous GP state-space model" (cGPSSM): a GP learns state derivatives from states, set points and piecewise-constant feeds. One finite element per day; bolus discontinuities are handled by equality constraints between adjacent elements.
  - In silico mammalian fed-batch, 6 states (VCD, Glc, Gln, NH4, Lac, titer), 30 LHS experiments (20 train, 10 validation), 15% noise. About 15 iterations sufficed. No numeric RMSEP is given in the text — [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.12.27.474269.full)

**Bayer et al. (Austria / TUHH)**
- Bayer, Duerkop, Striedner, Sissolak (2021), "Model Transferability and Reduced Experimental Burden in Cell Culture Process Development Facilitated by Hybrid Modeling and Intensified Design of Experiments", Front Bioeng Biotechnol 9:740215, DOI 10.3389/fbioe.2021.740215 — [Frontiers](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2021.740215/full)
  - **Structure:** serial hybrid. A single-hidden-layer ANN (tanh, **4 neurons** best, linear output, Levenberg–Marquardt with regularization) outputs the specific growth rate μ and the specific productivity q_P, which feed mass balances for VCC, titer and dilution.
  - **NN inputs:** 2 CPPs (temperature, feed glucose), Asp/Glu ratio, Gln, Asn and Ala, z-scored. Training used 40 random 60/40 train/validation splits with model averaging and early stopping.
  - **Data:** 18 shake-flask conditions in duplicate (bolus feed), plus 11 × 15 L runs (5 static, 6 iDoE with intra-run CPP shifts).
  - **Test NRMSE:** shaker-trained model transferred to 15 L gave VCC 10.92% and titer 17.79% (training 5.53% / 8.04%). The iDoE-trained model applied to static runs gave VCC 13.75% and titer 21.13%.
  - The roughly 67% experiment saving often quoted for iDoE comes from prior *E. coli* work cited in this paper and is not quantified for CHO here. Note that one search-engine summary attributed ">66% reduction" to this CHO paper, which the full text does not support.
- Bayer, Duerkop, Pörtner, Möller (2023), "Comparison of mechanistic and hybrid modeling approaches for characterization of a CHO cultivation process: Requirements, pitfalls and solution paths", Biotechnol J 18(1):e2200381, DOI 10.1002/biot.202200381 — [Europe PMC abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Comparison%20of%20mechanistic%20and%20hybrid%20modeling%20approaches%22&resultType=core&format=json)
  - 33 experiments in a 4-factor fractional factorial design, split into 4 data partitions.
  - The mechanistic model gave informative results fairly independently of the partition. The **hybrid model was more data-dependent but more accurate on all partitions.**
  - **Four carefully chosen initial experiments could represent the full design space.** Per-variable NRMSE values were not retrievable (paywall).

**Pinto / Ramos / Oliveira (NOVA Lisbon) "deep hybrid" line (2022–2024)**
- Pinto, Mestre, Ramos, Costa, Striedner, Oliveira (2022), "A general deep hybrid model for bioreactor systems: combining first principles with deep neural networks", Comput Chem Eng 165:107952, DOI 10.1016/j.compchemeng.2022.107952 (preprint bioRxiv 10.1101/2022.06.07.495118).
  - Methods: deep FFNN inside a general bioreactor mass-balance framework, ADAM, stochastic regularization, depth-dependent initialization, and modified sensitivity equations for faster gradients.
  - On a 50 L *P. pastoris* process: **+18.4% prediction accuracy and −43.4% CPU time** vs shallow hybrid — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Deep%20hybrid%20modeling%20of%20a%20HEK293%22%20OR%20TITLE%3A%22general%20deep%20hybrid%20model%20for%20bioreactor%22&resultType=core&format=json); [NOVA record](https://novaresearch.unl.pt/en/publications/a-general-deep-hybrid-model-for-bioreactor-systems-combining-firs/)
- Pinto, Ramos, Costa, Rossell, Dumas, Oliveira (2023), "Hybrid deep modeling of a CHO-K1 fed-batch process: combining first-principles with deep neural networks", Front Bioeng Biotechnol 11:1237963, DOI 10.3389/fbioe.2023.1237963 — [PMC10515724](https://pmc.ncbi.nlm.nih.gov/articles/PMC10515724/)
  - **Data:** 24 CHO-K1 fed-batch runs (Ambr 250, ~12 days, daily sampling, GSK), 30 measured species (Xv, product, Glc, Lac, Gln, Glu, NH4 and amino acids). Assumed measurement SD: 5% product, 10% Xv, 20% metabolites.
  - **Structure:** perfectly mixed fed-batch balances with 5 feed streams. NN inputs are all 30 concentrations normalized by their maxima. NN outputs are the 30 specific rates. Integration by RK4.
  - **Training:** ADAM (lr 1e-3), semidirect sensitivity equations, minibatch fraction 0.8, weight dropout 0.2, weighted least squares (inverse variance). Shallow baseline: one tanh layer trained with Levenberg–Marquardt, 10 restarts.
  - **Validation:** 10 random permutations of 20 train / 4 test batches.
  - **Results:** best deep model 25×25×25 ReLU (2,855 weights), test WMSE 1.88±0.44 vs best shallow 2.46±0.42 (**−23.6%**). Deep training costs 31.6% more CPU.
  - On a synthetic extrapolation set (star points of a CCD), the noise-free test WMSE fell by 73.8% (0.54 vs 2.06).
  - Qualitatively, Xv, product, Glc and Gln on test batches stay within error bars, and the **lactate/ammonium production→consumption switches are captured**. Per-state numeric test errors are not reported. Data are proprietary.
- Ramos, Pinto, Poiares-Oliveira, Peeters, Dumas, Oliveira (2024), "Deep hybrid modeling of a HEK293 process: Combining LSTM networks with first principles equations", Biotechnol Bioeng 121(5):1554–1568, DOI 10.1002/bit.28668 — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Deep%20hybrid%20modeling%20of%20a%20HEK293%22%20OR%20TITLE%3A%22general%20deep%20hybrid%20model%20for%20bioreactor%22&resultType=core&format=json)
  - 20 fed-batch cultures, 27 extracellular states, 784 architectures compared.
  - LSTM hybrids beat FFNN hybrids on training and test error and were less sensitive to resampling, at the cost of more parameters and CPU.
  - Best model: LSTM with 7 internal states in series with material balances, **test R² = 0.93** over ~10 days for all 27 variables.
- Pinto et al. (2023), "Hybrid Deep Modeling of a GS115 (Mut+) *Pichia pastoris* Culture with State–Space Reduction", Fermentation 9(7):643, DOI 10.3390/fermentation9070643 — [DOI](https://doi.org/10.3390/fermentation9070643)
- Review: "From Shallow to Deep Bioprocess Hybrid Modeling: Advances and Future Perspectives", Fermentation 2023, 9(10):922, DOI 10.3390/fermentation9100922. It reports deep models generalizing better than shallow ones. The author list was inconsistent in search metadata; verify — [DOI](https://doi.org/10.3390/fermentation9100922)

**Other CHO hybrid structures (2022–2025)**
- Okamura, Badr, Murakami, Sugiyama (U Tokyo, 2022), "Hybrid Modeling of CHO Cell Cultivation in Monoclonal Antibody Production with an Impurity Generation Module", Ind Eng Chem Res. It has three modules:
  1. Monod-type kinetic metabolism until cell death.
  2. A **data-driven module that updates kinetic parameters** from system dynamics and changing conditions (PCR-based according to the Narayanan 2023 review).
  3. HCP/DNA impurity kinetics.

  It was validated on new high-performance CHO lines at different operating conditions — [U Tokyo press release](https://www.t.u-tokyo.ac.jp/en/press/pr2022-09-30-001); [Narayanan 2023 review](https://www.frontiersin.org/journals/chemical-engineering/articles/10.3389/fceng.2023.1157889/full)
- Richelle, Andersson, Antonakoudis, Jakobsson, Pijeaud, Vernersson, Trygg (Sartorius, 2025), "A Hybrid Modeling Framework for Predictive Digital Twins of CHO Cell Culture", bioRxiv DOI 10.1101/2025.11.24.690194 (also listed in CSBJ) — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.24.690194.full)
  - **VCD model:** a deep NN (64-64-64-16-8-1) predicts μ as a sigmoid baseline plus a learned deviation, which feeds a viable/dead/lysed ODE.
  - **Metabolite model:** "FLEX" kinetic ODEs for Glc, Lac, Gln, Glu and NH4.
  - **Constraint-based layer:** PCA-constrained dFBA for amino acids.
  - **Data:** 23 Ambr15 CHO-S runs, 8 feed formulations. Validation: leave-one-batch-out and leave-one-media-group-out.
  - **Results:** FLEX fit R² Glc 0.91, Gln 0.98, Glu 0.91, NH4 0.92, but **lactate only 0.36**. VCD/titer test metrics are not given numerically in the text.
- Khare, Ndahiro, Klaubert, Ma, Bertalan, Kevrekidis, Harcum, Betenbaugh (2025), "Comparing Kinetic versus Stoichiometric Priorities in Hybrid Models of CHO Metabolism", bioRxiv DOI 10.1101/2025.10.28.685134 — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.10.28.685134v1.full)
  - Compares kinetic-priority and stoichiometric-priority dFBA hybrids (35 reactions, 24 metabolites, 14 kinetic reactions) on Ambr250 CHO with bolus glucose feeds.
  - The kinetic-oriented model generalized better to modified initial conditions and captured the lactate/alanine switch via reversible Michaelis–Menten terms. The lactate-fed case still needed parameter refits. Results are qualitative only.
- Sartorius/Waterloo hybrid dFBA-PLS model over 18 CHO fed-batch runs, reported normalized MSE ≈ 0.15 for most metabolites. Source is trade press; primary paper not checked — [GEN](https://www.genengnews.com/topics/bioprocessing/hybrid-model-predicts-and-optimizes-cho-cell-mab-production/)
- Narayanan, von Stosch, Feidl, Sokolov, Morbidelli, Butté (2023), "Hybrid modeling for biopharmaceutical processes: advantages, opportunities, and implementation", Front Chem Eng 5:1157889, DOI 10.3389/fceng.2023.1157889 — [Frontiers](https://www.frontiersin.org/journals/chemical-engineering/articles/10.3389/fceng.2023.1157889/full)
  - Covers serial architectures and the "degree of hybridization" concept. Data-driven parts used in the literature: ANN/MLP, PCR, PLSR, trees, GPs, DNNs and symbolic regression.
  - Claims hybrids need "much fewer data" than PLSR/ANNs and extrapolate better, citing Narayanan 2019/2021/2022, but gives **no numbers** in the review.
- Schweidtmann, Zhang, von Stosch (2024), "A review and perspective on hybrid modeling methodologies", Digital Chemical Engineering 10:100136, DOI 10.1016/j.dche.2023.100136 (open access). It counts more than 150 publications over 20+ years — [TU Delft repository](https://repository.tudelft.nl/record/uuid:5079278e-7034-4365-8737-7ebc669a1e63)
- Sansana, Joswiak, Castillo, Wang, Rendall, Chiang, Reis (2021), "Recent trends on hybrid modeling for Industry 4.0", Comput Chem Eng 151:107365, DOI 10.1016/j.compchemeng.2021.107365 [background, not re-fetched] — [DOI](https://doi.org/10.1016/j.compchemeng.2021.107365)
- Herrera-Ruiz et al. (2025), "Hybrid Modeling for Bioprocesses: Architectures, Applications, and Perspectives", Engineering Reports, DOI 10.1002/eng2.70502 (architectural review) — [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/eng2.70502)

### Inferences
- The training recipe that recurs across the strongest CHO results has four parts:
  1. Integrate the whole trajectory from the initial condition (single shooting through the ODE), with sensitivity or adjoint gradients.
  2. Use a weighted least-squares loss scaled by measurement variance.
  3. Regularize: few hidden nodes, dropout or early stopping, repeated random batch-level splits, model averaging.
  4. Hold out whole batches (leave-batch-out), never time points.

  This matches the hackathon "t=0 + feeds → trajectory" task exactly.
- For a 5-state problem (VCD, dead cells = TCD−VCD, Glc, Lac, V), the literature suggests that **predicting only a handful of rates** works well: μ, k_d, q_Glc, q_Lac, plus optionally lysis. Pinto 2023 predicted 30 rates from 30 states with ~2,900 weights on 20 training runs. Bayer 2021 used only 4 hidden neurons. A small network of roughly 1–2 layers and 5–20 units is appropriate for "dozens of batches".
- Unknown-composition feeds: Bayer 2021 fed CPPs (e.g., feed glucose concentration) directly as NN inputs. An analogous approach is to give the NN cumulative or added amounts of each unknown feed, or to add latent "nutrient" states that receive the bolus feeds with learned feed concentrations. The latter is an extrapolation of published practice, not a published CHO result found here.
- Deep vs shallow: the 24% test-error improvement in Pinto 2023 came with 20 training runs and 30 measured species. With only 4–5 measured species, the benefit of depth is likely smaller and overfitting risk larger. This is an inference, not tested in the literature found.

### Gaps
- Per-state test errors (e.g., %RMSE for VCD, glucose, lactate on unseen batches) are rarely reported. Pinto 2023 reports only aggregate WMSE. Bayer 2023 NRMSEs and Doyle 2026 numbers were paywalled (HTTP 403).
- No CHO paper found that directly compares serial vs parallel hybrid structures on the same data in 2022–2026.
- The exact run count and errors of Narayanan 2019 could not be verified; the full text was not accessible.
- Kotidis & Kontoravdi: no grey-box/ML CHO hybrid paper by them found. Their mechanistic CHO model, adapted from Kotidis et al. 2019, underlies the Yu et al. 2026 EnKF transfer work (Section 5).

---

## 2. Universal differential equations (UDEs), neural ODEs and physics-informed neural networks (PINNs) for bioreactors / CHO

### Takeaway
UDEs are the SciML name for the same serial hybrid idea: an ODE right-hand side with embedded NNs, trained by differentiating through the solver. They are mathematically the same as the "hybrid semi-parametric" models of Section 1. Bioprocess-specific UDE work in 2024–2026 is mostly microbial or methodological, while CHO neural-ODE and PINN papers exist but are small and rarely report hard out-of-batch numbers. PINNs (collocation-based) are less suited to the "predict unseen batch from t=0 + feeds" task than ODE-integrating hybrids, because a vanilla PINN learns one trajectory. CHO PINN papers instead use them to fuse small-scale mechanistic knowledge with pilot data.

### Cited Findings
- Rackauckas et al. (2020), "Universal Differential Equations for Scientific Machine Learning", arXiv:2001.04385. Defines UDEs, implemented in the Julia SciML stack (DifferentialEquations.jl / DiffEqFlux.jl) [background, not re-fetched] — [arXiv](https://arxiv.org/abs/2001.04385)
- A 2025 *npj Systems Biology and Applications* review, "Current state and open problems in universal differential equations for systems biology", says UDE training is hard because of stiff dynamics and noisy, sparse data, and describes a systematic training pipeline. It credits Julia SciML for much of the progress — [Nature npj SBA](https://www.nature.com/articles/s41540-025-00550-w)
- "Functional and parametric identifiability for universal differential equations" (arXiv 2510.14140, 2025) addresses identifiability of UDE components. This is important when one wants to interpret the learned rate function — [arXiv](https://arxiv.org/pdf/2510.14140)
- Menard (U Waterloo thesis), "Universal Differential Equations Applied to Bioprocesses". Uses Julia DiffEqFlux and DiffEqParamEstim, and notes that "hybrid neural differential equation" and UDE name the same model class — [thesis PDF](https://uwaterloo.ca/computational-mathematics/sites/default/files/uploads/documents/menard_jyler_0.pdf)
- Herrera-Ruiz et al. 2025 review (Engineering Reports) reports a UDE batch fermentation example in which a neural ODE was added additively to biomass/ethanol kinetics. It gave **3.6-fold and 1.7-fold better performance** than the parametric model, at the cost of longer training — [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/eng2.70502)
- Martínez, Rocha Calvette, Péré, Barrientos, Ossandón (2025/2026), "Hybrid modeling framework for bioprocesses with minimal data", bioRxiv DOI 10.1101/2025.11.14.688550 — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.14.688550.full); code at [GitHub](https://github.com/carlosmvond/hybrid-modelling-of-batch-bioreactors)
  - **Structure:** UDE-style. Two FFNNs per state, f_i (growth-like, multiplied by state ξ_i for non-negativity) and g_i (ReLU output for non-negative production). Zero-invariance constraints remove g_i for biomass and substrate. **No stoichiometry or rate laws assumed.**
  - **Training:** JAX/Optax ADAM. Minibatch = one whole batch experiment; 10 restarts.
  - **Synthetic *E. coli* overflow model:** trained on only **3 batches**. Test R² is mostly 0.9–0.99, but extrapolation to a lower initial glucose (5 g/L) gives negative R² for glucose and acetate.
  - **Interpretability:** the learned g for acetate is near zero below a growth threshold and linear above it, recovering overflow metabolism.
- Cui, Bertalan, Ndahiro, Khare, Betenbaugh, Maranas, Kevrekidis (2023), "Data-driven and Physics Informed Modelling of Chinese Hamster Ovary Cell Bioreactors", arXiv:2305.03257. Compares:
  - black-box neural ODEs,
  - white-box parameter fitting,
  - grey-box models where an NN learns partially unknown kinetic expressions inside mass balances, with the **overdetermined metabolic (MFA-like) convex optimization encoded as a differentiable layer**.

  A later journal version exists (Penn State Pure listing); venue not verified. Numbers are not in the abstract — [arXiv](https://arxiv.org/abs/2305.03257); [PSU Pure](https://pure.psu.edu/en/publications/data-driven-and-physics-informed-modeling-of-chinese-hamster-ovar/)
- "A neural ordinary differential equation model for predicting the growth of Chinese Hamster Ovary cell in a bioreactor system", Biotechnology and Bioprocess Engineering, Sept 2024. It models VCD, Glc, Lac, pH and DO with inequality constraints in estimation. A secondary summary says it needs less data and extrapolates more reliably than mechanistic or pure NN models; numbers and authors not verified — [ResearchGate](https://www.researchgate.net/publication/383645325_A_neural_ordinary_differential_equation_model_for_predicting_the_growth_of_Chinese_Hamster_Ovary_cell_in_a_bioreactor_system); [Altmetric](https://altmetric.com/details/117992615/citations)
- Chiu et al. (2025), "Neural ordinary differential equation-based model predictive controller for regulating glucose concentration in a fed-batch CHO cell bioreactor", Can J Chem Eng, DOI 10.1002/cjce.25623. Simulated fed-batch CHO; neural-ODE MPC benchmarked against NN- and LSTM-based MPC — [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/cjce.25623)
- Yang, Fahey, Truccollo, Browning, et al. (2024), "Hybrid Modeling of Fed-Batch Cell Culture Using Physics-Informed Neural Network", Ind Eng Chem Res, DOI 10.1021/acs.iecr.4c01459. Hybrid PINN for a **large-scale pilot CHO fed-batch** that combines first-principles knowledge from small-scale studies with deep learning on pilot data. Full text 403; quantitative results not obtained — [Bohrium listing](https://www.bohrium.com/paper-details/hybrid-modeling-of-fed-batch-cell-culture-using-physics-informed-neural-network/1055851808313835568-4160); [NSF PAR copy](https://par.nsf.gov/servlets/purl/10537019)
- Xu, Chen, Qu (2025), "An improved physics-informed neural network with scheduled sampling for modeling the cell culture process for production of monoclonal antibodies" (SS-IHPINN), Chinese J Biotechnol 42(2):942–954, DOI 10.13345/j.cjb.250654. Physics-guided initialization plus scheduled sampling for scarce data; claims transfer when process parameters vary. MeSH suggests hybridoma, not CHO — [Europe PMC query](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=(TITLE%3A%22neural%20ODE%22%20OR%20TITLE%3A%22neural%20ordinary%20differential%22%20OR%20TITLE%3A%22universal%20differential%22%20OR%20TITLE%3A%22physics-informed%22%20OR%20TITLE%3A%22SINDy%22%20OR%20TITLE%3A%22sparse%20identification%22%20OR%20TITLE%3A%22symbolic%20regression%22)%20AND%20(ABSTRACT%3A%22bioprocess%22%20OR%20ABSTRACT%3A%22bioreactor%22%20OR%20ABSTRACT%3A%22fed-batch%22%20OR%20ABSTRACT%3A%22CHO%22)%20AND%20PUB_YEAR%3A%5B2022%20TO%202026%5D&resultType=core&format=json&pageSize=20)
- Li et al. (2026), "Physics-encoded transfer learning for scale-up modeling of CHO cell bioreactors", AIChE J, DOI 10.1002/aic.70542. Physics-informed FiLM (feature-wise linear modulation) design for cross-scale transfer under data scarcity. Only the search snippet was seen (403) — [Wiley](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.70542)
- Weir et al. (2025), "Noise aware parameter estimation in bioprocesses: Using neural network surrogate models with nonuniform data sampling", AIChE J, DOI 10.1002/aic.18634. Estimates μ_max and the primary death rate from real fed-batch CHO data — [Wiley](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.18634)
- AIChE 2023 abstract "Hybrid Modeling of CHO Cell Culture Using Physics Informed Neural Networks" and AIChE 2025 "Hybrid Modeling of Industrial Bioreactors Using PINNs" (adaptive selection among candidate kinetic expressions). Abstracts only — [AIChE 2023](https://aiche.org/conferences/aiche-annual-meeting/2023/proceeding/paper/374b-hybrid-modeling-cho-cell-culture-using-physics-informed-neural-networks); [AIChE 2025](https://proceedings.aiche.org/conferences/aiche-annual-meeting/2025/proceeding/paper/hybrid-modeling-industrial)
- "A Physics-Informed Neural Network (PINN) framework for generic bioreactor modelling" (2025, ResearchGate) — [ResearchGate](https://www.researchgate.net/publication/394557507_A_Physics-Informed_Neural_Network_PINN_framework_for_generic_bioreactor_modelling)

### Inferences
- For "t=0 + feeds → full trajectory" on unseen batches, an **ODE-integrated hybrid/UDE** (NN inside the RHS, trained through the solver) is the natural choice. Collocation-style PINNs fit a time-indexed network per trajectory and need extra conditioning on t=0 and feeds, as in operator-learning or FiLM variants, to generalize across batches.
- Bolus feeds are easy in a UDE: integrate between feed events and apply discrete jump maps for V and concentrations at each bolus. In Julia this is done with `DiscreteCallback`; in JAX/diffrax/torchdiffeq by segmenting the integration.
- The Martínez 2025 constraints (non-negativity via multiplication by the state, zero-invariance, ReLU-positive production terms) are cheap, interpretable priors that should transfer directly to VCD/Glc/Lac.

### Gaps
- No peer-reviewed 2022–2026 paper found that applies Julia SciML UDEs specifically to **CHO fed-batch with held-out-batch metrics**. The closest are Cui et al. 2023 (grey-box neural ODE, CHO) and the 2024 Biotechnol Bioprocess Eng neural-ODE paper, whose numbers were not accessible.
- PINN CHO papers (Yang 2024 IECR; Li 2026 AIChE J) were paywalled. Their accuracy on unseen batches is unknown from this research.

---

## 3. SINDy and symbolic regression for interpretable bioprocess rate laws

### Takeaway
Plain polynomial SINDy is a poor fit for bioprocess kinetics, which are saturating, inhibitory and switching. Successful variants use rational or implicit SINDy or physiologically informed libraries (Monod/Haldane terms). The more mature bioprocess line is **symbolic regression inside a hybrid backbone**:
- Functional-Hybrid (Narayanan et al. 2022),
- Forster et al. 2024 (two-step SR without repeated integration),
- Riezzo et al. 2026 (automated data-efficient SR with a macroscopic backbone).

These report similar interpolation accuracy to NN hybrids and better extrapolation, but validation so far is mainly in silico or microbial, with no CHO fed-batch held-out benchmarks.

### Cited Findings
- Narayanan, Cruz Bournazou, Guillén-Gosálbez, Butté (2022), "Functional-Hybrid modeling through automated adaptive symbolic regression for interpretable mathematical expressions", Chem Eng J 430(4):133032 (bioRxiv 10.1101/2021.06.09.447664) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1385894721046088); [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.06.09.447664v1)
  - Replaces the ANN in a serial hybrid with SR over a **library of domain-specific knowledge blocks**: ranked "functional beliefs" such as Monod and inhibition terms.
  - Performs similarly to an ANN hybrid in interpolation and gives **superior extrapolation**.
  - Tested on 4 benchmark systems plus a microbial fermentation. The authors describe it as only "to some extent" interpretable.
- Forster, Vázquez, Müller, Guillén-Gosálbez (2024), "Machine learning uncovers analytical kinetic models of bioprocesses", Chem Eng Sci vol. 300 (Dec 2024). The repository lists 2024-08-30 online. It uses two-step SR (fit smooth state profiles and derivatives, then regress rate expressions) to avoid repeated ODE integration, and slightly outperformed NN benchmarks on numerical examples — [URL repository](https://dau.url.edu/handle/20.500.14342/4620); [Forster ETH dissertation](https://research-collection.ethz.ch/bitstream/handle/20.500.11850/701976/1/Dissertation_TimForster.pdf)
- Riezzo, Rogers, Kay, Zhang (2026), "Automated Data-Efficient Symbolic Regression for Interpretable Bioprocess Model Development", Biotechnol Bioeng, DOI 10.1002/bit.70328 — [Europe PMC query](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=(TITLE%3A%22neural%20ODE%22%20OR%20TITLE%3A%22neural%20ordinary%20differential%22%20OR%20TITLE%3A%22universal%20differential%22%20OR%20TITLE%3A%22physics-informed%22%20OR%20TITLE%3A%22SINDy%22%20OR%20TITLE%3A%22sparse%20identification%22%20OR%20TITLE%3A%22symbolic%20regression%22)%20AND%20(ABSTRACT%3A%22bioprocess%22%20OR%20ABSTRACT%3A%22bioreactor%22%20OR%20ABSTRACT%3A%22fed-batch%22%20OR%20ABSTRACT%3A%22CHO%22)%20AND%20PUB_YEAR%3A%5B2022%20TO%202026%5D&resultType=core&format=json&pageSize=20)
  - A generic macroscopic kinetic backbone plus SR for key kinetic terms. Two strategies compared: SR embedded in the backbone, and first estimating **time-varying parameter profiles** then applying SR.
  - A "local iterative structural correction" improves convergence to the true kinetics and was more data-efficient than model-based DoE.
  - In silico yeast fermentation only.
- A "KAN-SR" (Kolmogorov–Arnold-network symbolic regression) framework has been described as data-efficient SR for automated discovery of interpretable bioprocess models. The exact paper was not resolved in search — [search context, Frontiers Chem Eng 2024](https://www.frontiersin.org/journals/chemical-engineering/articles/10.3389/fceng.2024.1494244/full)
- "Interpretable-AI-Based Model Structural Transfer Learning to Accelerate Bioprocess Model Construction" (2025) combines SR with ANN feature attribution to transfer model structure between processes — [ResearchGate](https://www.researchgate.net/publication/393793279_Interpretable-AI-Based_Model_Structural_Transfer_Learning_to_Accelerate_Bioprocess_Model_Construction)
- "A reinforcement learning-based hybrid modeling framework for bioprocess kinetics identification" (PMC10092184) — kinetic structure identification via RL. Not fetched — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10092184/)
- SINDy for biology: Mangan et al. extended SINDy to **rational nonlinearities (implicit SINDy)** and recovered Michaelis–Menten kinetics and yeast glycolysis. This is the relevant variant for saturating bioprocess kinetics [arXiv 1605.08368, 2016, foundational] — [arXiv](https://arxiv.org/pdf/1605.08368)
- A PLOS Comput Biol study of a mixotrophic *Clostridium* coculture in perfusion pairs mechanistic modeling with SINDy. It notes that **standard SINDy polynomial libraries fail to capture saturation kinetics and metabolic switching**, so it uses a physiologically informed library — [PLOS CB](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1014759)
- "On the Number of Reactions and Stoichiometry of Bioprocess Macroscopic Models: an Implicit Sparse Identification Approach" — sparse identification applied to macroscopic reaction schemes (number of reactions, stoichiometry) — [ResearchGate](https://www.researchgate.net/publication/375850915_On_the_Number_of_Reactions_and_Stoichiometry_of_Bioprocess_Macroscopic_Models_an_Implicit_Sparse_Identification_Approach)
- Nyande et al. (2024), "Data-driven identification of crystallization kinetics", AIChE J, DOI 10.1002/aic.18333. Iterates PLS and sparsity steps and reports a **minimum-data threshold** for identifying kinetics (non-bio but methodologically relevant) — [Wiley](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.18333)
- OASIS, operable adaptive sparse identification (Bhadriraju/Kwon line) — adaptive SINDy for process systems — [NSF PAR](https://par.nsf.gov/servlets/purl/10188321)
- Aydin & Arkun (Oct 2026), "Autonomous Structural Discovery of Bioprocess Dynamics and Kinetics via LLM-Driven Agents", arXiv:2610.35391 — [arXiv](https://arxiv.org/abs/2610.35391)
  - Combines an LLM agent with moving-horizon estimation, log-decomposition of multiplicative vs additive rate terms, RAG kinetic libraries, and active DoE.
  - Recovered Haldane/Luong/autocatalysis terms in a simulated CSTR with **<6% parameter error**.
  - Argues SINDy yields high-dimensional polynomial approximations lacking biological meaning. No head-to-head numbers.

### Inferences
- For VCD/Glc/Lac with a few dozen batches, the pragmatic interpretable route is two-stage:
  1. Fit a small NN hybrid (or estimate per-interval specific rates from the data, as in Riezzo's "time-varying parameter profiles").
  2. **Distil** the learned μ(Glc, Lac, …), q_Glc(·) and q_Lac(·) into Monod/Haldane/Luedeking–Piret-type expressions, using SR or SINDy with a rational, bioprocess-specific library.

  This is what the 2022–2026 SR literature does, but with in silico data.
- Bolus feeding makes derivative estimation from daily samples hard because of jumps at feeds. Two-step SR/SINDy therefore requires **segment-wise smoothing between feed events** or integral (weak-form) formulations. This is an inference; no bioprocess paper found addresses it explicitly besides cGPSSM (Section 1).

### Gaps
- No peer-reviewed SINDy or SR application to **CHO fed-batch** with held-out-batch accuracy was found.
- Ensemble-SINDy and SINDYc have not been applied to bioprocesses in any paper found; these are absent from the bioprocess literature surfaced.

---

## 4. Bayesian / GP hybrids, hierarchical (mixed-effects) models, batch-to-batch variability and UQ

### Takeaway
GP-based hybrids are the main 2025–2026 development for UQ in CHO fed-batch. Doyle et al. (2026) used GP-in-mass-balance on intensified fed-batch and found it outperforms a Monod model, while flagging **transient bolus-feed effects as unresolved**. Wang, Harcum & Xie (2026) built a multi-scale stochastic hybrid with calibrated prediction intervals. Explicit **hierarchical Bayesian or mixed-effects** treatments of batch-to-batch variability exist mainly in microbial work (Helleckes 2022 murefi; Bayesian non-parametric mixed-effects growth curves) and are a clear opening for CHO.

### Cited Findings
- Doyle, Yang, Colarusso, Bano, Glennon, Jiménez del Val (2026), "Hybrid Dynamic Modelling With Gaussian Process Regression for Intensified Fed-Batch CHO Cell Culture Processes", Biotechnol Bioeng, DOI 10.1002/bit.70339 — [Europe PMC abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Gaussian%20Process%22%20AND%20TITLE%3A%22Intensified%20Fed-Batch%22&resultType=core&format=json); thesis background in [UCD repository](https://researchrepository.ucd.ie/entities/publication/95db1e50-999c-413f-b1ec-73ee1f386d60)
  - GPR combined with material balances to predict growth and metabolism.
  - Two contrasting intensified fed-batch CHO processes, development and manufacturing scale. Compared against a **Monod-based kinetic model**; studied incremental data availability and data quality.
  - Hybrid outperforms mechanistic. Data quality and DoE matter for reliability. **Transient bolus feeding effects remain a challenge requiring model-structure work.** The PhD thesis also compared GPR vs ANN for the specific rates.
  - Numbers paywalled.
- Wang, Harcum, Xie (2026), "Multi-Scale Hybrid Modeling to Predict Cell Culture Process With Metabolic Phase Transitions", Biotechnol Bioeng 123(7):1745–1770, DOI 10.1002/bit.70205 (arXiv 2412.03883) — [PMC13309251](https://pmc.ncbi.nlm.nih.gov/articles/PMC13309251/); code at [GitHub](https://github.com/kw48792/Multi-Scale-Hybrid-Model-with-Metabolic-Phase-Transitions)
  - **Structure:** single-cell Michaelis–Menten metabolic network with SDE exchange rates, plus a sigmoid **phase-transition probability** (growth → stationary → decline) driven by culture age, qO2 and pH, plus population mass-balance SDEs. Fitted by EM / maximum likelihood with Euler–Maruyama.
  - **Data:** CHO-K1 VRC01 IgG1, **9 ambr250 fed-batch runs** (3 conditions × triplicate).
  - **Look-ahead WAPE:** 5.66–8.37% (1-day), 7.39–10.13% (3-day), 8.51–12.07% (5-day). 95% PI coverage is about 91.5–94.3%.
  - Whole-trajectory leave-one-batch-out extrapolation is shown only graphically. **No baselines** (LSTM/GP/mechanistic) are quantified.
- Helleckes, Osthege, Wiechert, von Lieres, Oldiges (2022), "Bayesian calibration, process modeling and uncertainty quantification in biotechnology", PLoS Comput Biol 18(3):e1009223, DOI 10.1371/journal.pcbi.1009223. Python packages **calibr8** (measurement/calibration models) and **murefi** (multi-replicate ODE fitting). A **hierarchical Monod ODE** across replicate *C. glutamicum* microbioreactor batches is fitted by ML and Bayesian inference — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Bayesian%20calibration%2C%20process%20modeling%20and%20uncertainty%20quantification%22%20OR%20TITLE%3A%22Transfer%20Learning%20Approaches%20in%20Bioprocess%20Engineering%22&resultType=core&format=json)
- Bayesian non-parametric mixed-effects model of microbial growth curves (PLoS Comput Biol 2020, DOI 10.1371/journal.pcbi.1008366). Batch effects are random effects from a shared distribution, and **ignoring batch effects biases estimates** — [PMC7644099](https://pmc.ncbi.nlm.nih.gov/articles/PMC7644099/)
- "A Multi Clone Kinetic Model for characterizing Chinese hamster ovary cell line variability" (2025, with Kotidis). Derives 13 kinetic parameters from one fed-batch run per clone (49 data points). This gives per-unit parameter sets suitable for a hierarchical layer — [PMC12486387](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12486387/)
- Hutter, von Stosch, Cruz Bournazou, Butté (2021), hybrid **GP** models with entity embeddings (Biotechnol Bioeng 118(11):4389–4401, DOI 10.1002/bit.27907). See Section 5 — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Knowledge%20transfer%20across%20cell%20lines%22%20OR%20TITLE%3A%22Physics-encoded%20transfer%20learning%22%20OR%20TITLE%3A%22transfer%20learning%20approach%20for%20predictive%20modeling%20of%20bioprocesses%22&resultType=core&format=json)
- "Multi-fidelity Gaussian Process for Biomanufacturing Process Modeling with Small Data", arXiv 2211.14493 (2022) — [arXiv](https://arxiv.org/pdf/2211.14493)
- Cruz-Bournazou et al. 2021 cGPSSM (Section 1). GP derivative uncertainty is available, but propagating it through integration was left out of scope — [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.12.27.474269.full)
- Yu, del Rio Chanona, Kontoravdi (2026), ensemble Kalman filter for joint state/parameter estimation. Ensemble parameter spread gives UQ and shows which parameters shift under new conditions (Section 5) — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Knowledge%20transfer%20across%20cell%20lines%22%20OR%20TITLE%3A%22Physics-encoded%20transfer%20learning%22%20OR%20TITLE%3A%22transfer%20learning%20approach%20for%20predictive%20modeling%20of%20bioprocesses%22&resultType=core&format=json)

### Inferences
- With "dozens of batches", a **mixed-effects grey-box** is a well-motivated, under-explored design:
  - population-level NN or Monod rate laws, plus per-batch random effects on a few parameters (e.g., μ_max scaling, an initial-condition offset like Martínez's α_j, or the lactate switch threshold);
  - predicting unseen batches at the population mean, with posterior-predictive bands from the random-effect variance.

  It directly addresses batch-to-batch variability that the feeds and t=0 state cannot explain. This is an inference; no CHO paper found does exactly this.
- GP hybrids (Doyle 2026) give UQ cheaply for low-dimensional rate inputs (Glc, Lac, VCD → μ, q). With 4–5 states and a few hundred daily samples, exact GPs are computationally trivial.
- The bolus-feed transient problem flagged by Doyle 2026 suggests adding a short-lived post-feed response (e.g., a feed-event state or cumulative-feed input) to the rate model.

### Gaps
- No CHO-specific hierarchical Bayesian or mixed-effects kinetic or hybrid paper with held-out-batch validation was found in 2022–2026.
- Bayesian neural-network hybrids for CHO: none found. The Europe PMC query "hybrid AND Bayesian" in titles with bioprocess terms, 2021–2026, returned zero hits.
- The Doyle 2026 quantitative results (and whether GP beats ANN) could not be retrieved.

---

## 5. Transfer learning and "foundation-model-like" approaches (embeddings of process conditions, cross-product / cross-scale hybrids), 2021–2026

### Takeaway
Bioprocess transfer learning is still small-model and hybrid-centric. The main approaches are:
- **entity embeddings** of product or cell line fed to a hybrid GP (Hutter 2021),
- fine-tuning general-to-specific models (Rogers 2022),
- physics-encoded FiLM transfer across scales (Li 2026),
- sequential Bayesian (EnKF) adaptation of mechanistic CHO models across cell lines and scales from a single new run (Yu, del Rio Chanona, Kontoravdi 2026).

No true "foundation model" for bioprocess trajectories was found. A 2026 Biotech Bioeng review lists benchmark datasets and TL-plus-hybrid as future directions.

### Cited Findings
- Hutter, von Stosch, Cruz Bournazou, Butté (2021), "Knowledge transfer across cell lines using hybrid Gaussian process models with entity embedding vectors", Biotechnol Bioeng 118(11):4389–4401, DOI 10.1002/bit.27907 (arXiv 2011.13863). Product identity is encoded as **learned embedding vectors** input to a GP. Better than one-hot encoding on a **simulated** cross-product task; embeddings capture product similarity — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Knowledge%20transfer%20across%20cell%20lines%22%20OR%20TITLE%3A%22Physics-encoded%20transfer%20learning%22%20OR%20TITLE%3A%22transfer%20learning%20approach%20for%20predictive%20modeling%20of%20bioprocesses%22&resultType=core&format=json); [arXiv](https://arxiv.org/abs/2011.13863)
- Rogers, Vega-Ramon, Yan, del Río-Chanona, Jing, Zhang (2022), "A transfer learning approach for predictive modeling of bioprocesses using small data", Biotechnol Bioeng 119(2):411–422, DOI 10.1002/bit.27980. Compares transfer strategies and topologies on two case studies, benchmarked against kinetic and pure data-driven models. The optimal transfer structure reflects mechanism, which aids interpretability — [same Europe PMC record](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Knowledge%20transfer%20across%20cell%20lines%22%20OR%20TITLE%3A%22Physics-encoded%20transfer%20learning%22%20OR%20TITLE%3A%22transfer%20learning%20approach%20for%20predictive%20modeling%20of%20bioprocesses%22&resultType=core&format=json)
- Yu, del Rio Chanona, Kontoravdi (2026), "Model-Enabled Knowledge Transfer Across Cell Lines, Culture Scales and Conditions", Biotechnol Bioeng 123(10):2613–2628, DOI 10.1002/bit.70269 (bioRxiv 10.64898/2025.11.30.691385) — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Knowledge%20transfer%20across%20cell%20lines%22%20OR%20TITLE%3A%22Physics-encoded%20transfer%20learning%22%20OR%20TITLE%3A%22transfer%20learning%20approach%20for%20predictive%20modeling%20of%20bioprocesses%22&resultType=core&format=json); [bioRxiv](https://www.biorxiv.org/content/10.64898/2025.11.30.691385v1.full)
  - Reduced Kotidis-2019-derived CHO model with 8 states and 24 kinetic parameters. Dual-ensemble EnKF with ensemble sizes 50–75.
  - Six datasets: 2 cell lines; shake flask vs 1.5 L STR; temperature shift; three feed regimes.
  - Transfers with **one dataset per target system, no reparameterization**. Forecasts improve as data accrue; for one feed condition, accurate Glc/Gln/Glu/Asn forecasts by day 9.
  - Captured a ~4× titer shift (about 2.5 g/L) that the base model missed. RMSEs are only in supplementary tables.
- Li et al. (2026), physics-encoded transfer learning with a physics-informed FiLM design for CHO scale-up, AIChE J, DOI 10.1002/aic.70542 (snippet only) — [Wiley](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.70542)
- Bayer et al. 2021: a hybrid trained on shake flasks (bolus feed) plus 3 center points transferred to 15 L (continuous feed) with test NRMSE of VCC 10.92% and titer 17.79% (Section 1) — [Frontiers](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2021.740215/full)
- Barón Díaz, Drommershausen, Grünberger, Holtmann (2026), "Transfer Learning Approaches in Bioprocess Engineering: Opportunities and Challenges", Biotechnol Bioeng 123(6):1417–1431, DOI 10.1002/bit.70186. Names **data heterogeneity and transferability** as the main obstacles, and lists TL plus hybrid/physics-informed models, **standardized benchmark datasets** and public-data mining as future directions — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Bayesian%20calibration%2C%20process%20modeling%20and%20uncertainty%20quantification%22%20OR%20TITLE%3A%22Transfer%20Learning%20Approaches%20in%20Bioprocess%20Engineering%22&resultType=core&format=json)
- "Machine Learning Methods for Small Data and Upstream Bioprocessing Applications: A Comprehensive Review" (arXiv 2506.12322, 2025). It lists hybrid GP with entity embeddings as a TL example across cell lines and process conditions — [arXiv](https://arxiv.org/pdf/2506.12322)
- Foundation models: searches surfaced only cell-biology (omics) foundation models, not bioprocess-trajectory models. A recent preprint notes that **simple models can outperform single-cell foundation models** on downstream benchmarks — [Nature perspective](https://www.nature.com/articles/s41586-025-08710-y); [arXiv 2602.16696](https://arxiv.org/pdf/2602.16696)
- An LLM-guided parameter-optimization study for mechanistic CHO bioreactor models exists (not fetched) — [PMC13665592](https://pmc.ncbi.nlm.nih.gov/articles/PMC13665592)

### Inferences
- For a single-process hackathon dataset, the transferable ideas are:
  - **batch or condition embeddings**: a learned low-dimensional vector per batch, as in Hutter 2021, set to the population mean for unseen batches. This is the ML analogue of mixed-effects random effects (Section 4).
  - **sequential assimilation**: if early-run data of a test batch were ever allowed (not the case here), EnKF-style updating as in Yu 2026.
- "Foundation-model" approaches are not mature for bioprocess trajectories and are unlikely to help in this small-data regime.

### Gaps
- No public "pre-trained" bioprocess trajectory model or cross-company foundation model found.
- Could not verify a 2024 "Rogalla/Hutter" embedding paper suggested in the prompt. Searches surfaced Hutter 2021 (Biotechnol Bioeng) and Rogers et al. 2022 instead.

---

## 6. Head-to-head comparisons: hybrid vs mechanistic vs data-driven (PLS, RNN/LSTM, GP) on unseen CHO fed-batch runs

### Takeaway
Every CHO comparison found reports hybrid ≥ mechanistic in accuracy, and hybrid > statistical/black-box for whole-trajectory prediction from initial conditions plus process inputs. Hard numbers are scarce and rarely cover the same metrics:
- Bayer 2021: test NRMSE 11–14% VCC, 18–21% titer.
- Pinto 2023: deep vs shallow hybrid −24% WMSE.
- Ramos 2024 (HEK293): R² 0.93.
- Wang 2026: 6–12% WAPE for 1–5-day look-ahead.

Data needs run from about 4 well-chosen experiments (Bayer 2023) to about 20 training runs (Pinto 2023, Ramos 2024). No study benchmarks a pure LSTM/transformer against a hybrid on held-out CHO fed-batch runs with t=0-only inputs.

### Cited Findings
- Narayanan et al. 2019 (Biotechnol Bioeng, DOI 10.1002/bit.27097): hybrid > state-of-the-art statistical models (PLS-type) for predicting full time profiles from **initial and process conditions only**; better robustness and extrapolation — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22new%20generation%20of%20predictive%20models%22%20AND%20AUTH%3A%22Narayanan%22&resultType=core&format=json)
- Bayer et al. 2023 (Biotechnol J, DOI 10.1002/biot.202200381): hybrid **more accurate than mechanistic on all 4 data partitions** of 33 CHO fed-batch runs. The mechanistic model is more robust to which data are used. **4 well-chosen experiments** can represent the 4-D design space — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Comparison%20of%20mechanistic%20and%20hybrid%20modeling%20approaches%22&resultType=core&format=json)
- Bayer et al. 2021 (Front Bioeng Biotechnol): hybrid test NRMSE of VCC 10.92% and titer 17.79% (cross-scale), and VCC 13.75% and titer 21.13% (iDoE → static) — [Frontiers](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2021.740215/full)
- Pinto et al. 2023 (CHO-K1, 20 train / 4 test × 10 permutations): test WMSE deep 1.88 vs shallow 2.46, a −23.6% change. It summarizes Bayer et al. 2022 as finding shallow hybrids outperform mechanistic-only models — [PMC10515724](https://pmc.ncbi.nlm.nih.gov/articles/PMC10515724/)
- Ramos et al. 2024 (HEK293, 20 runs): LSTM-hybrid beats FFNN-hybrid; test R² 0.93 — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Deep%20hybrid%20modeling%20of%20a%20HEK293%22%20OR%20TITLE%3A%22general%20deep%20hybrid%20model%20for%20bioreactor%22&resultType=core&format=json)
- Doyle et al. 2026: GP-hybrid beats Monod mechanistic on intensified CHO fed-batch; numbers not accessible — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Gaussian%20Process%22%20AND%20TITLE%3A%22Intensified%20Fed-Batch%22&resultType=core&format=json)
- Narayanan et al. 2023 review: hybrids outperform PLSR and ANNs/RNNs in fermentation and cell culture, need "much fewer data", and give **~40% lower error than lumped kinetic mechanistic models for protein-A capture** (chromatography, not cell culture) — [Frontiers](https://www.frontiersin.org/journals/chemical-engineering/articles/10.3389/fceng.2023.1157889/full)
- Herrera-Ruiz 2025 review: UDE hybrid 3.6× and 1.7× better than parametric kinetics in batch fermentation (microbial) — [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/eng2.70502)
- Forster et al. 2024: SR kinetic models "slightly outperformed" NN benchmarks on numerical examples — [URL repository](https://dau.url.edu/handle/20.500.14342/4620)
- Rogers et al. 2022: TL models benchmarked against kinetic and data-driven models (small data; numbers not retrieved) — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Knowledge%20transfer%20across%20cell%20lines%22%20OR%20TITLE%3A%22Physics-encoded%20transfer%20learning%22%20OR%20TITLE%3A%22transfer%20learning%20approach%20for%20predictive%20modeling%20of%20bioprocesses%22&resultType=core&format=json)
- Caution: a vendor blog (bioprocesstools.com) claims hybrid R² 0.91–0.97 vs 0.80–0.92 for pure ML on CHO titer **without primary sources**. Do not use — [blog](https://bioprocesstools.com/blog/digital-twin-bioprocess/)

### Inferences
- Across studies, the consistent pattern is:
  - mechanistic models are data-robust but biased;
  - hybrids are the most accurate when the training design covers the test conditions;
  - pure black-box sequence models (LSTM/GRU) are not competitive for t=0-only trajectory prediction in small data.

  For the hackathon, a sensible benchmark ladder is: (i) Monod-type mechanistic, (ii) serial hybrid with a small NN or GP for rates, (iii) hybrid plus batch random effects, plus (iv) a black-box baseline (e.g., a neural ODE without mass balances, or GP/PLS on time and cumulative feeds) to quantify the hybrid gain.
- Typical expected error levels: about 10–15% NRMSE for VCD and about 15–20% for titer-like outputs on held-out runs (Bayer 2021). Lactate is the hardest state (R² 0.36 in Richelle 2025's kinetic sub-model; it needs switch behaviour, which Pinto 2023 captured with a deep hybrid).

### Gaps
- No found study reports a head-to-head **LSTM/transformer vs hybrid on unseen CHO fed-batch runs with t=0-only inputs**, with numbers.
- Per-variable errors for glucose and lactate on unseen CHO batches are almost never tabulated in accessible text.

---

## 7. Interpretability approaches for grey-box models

### Takeaway
Interpretability in this literature comes from four mechanisms:
1. **Mass-balance structure**: the NN only outputs specific rates, so conservation and dilution are exact.
2. **Inspecting learned rate functions** vs states, e.g., acetate or lactate overflow thresholds recovered from the NN.
3. **Symbolic distillation** of the NN part (Functional-Hybrid, SR, KAN-SR).
4. **Structurally interpretable stochastic or mechanistic hybrids**: phase-transition probabilities, EnKF parameter trajectories.

### Cited Findings
- Imposing mass balances yielded **biologically consistent glucose consumption patterns**, while a purely statistical model did not. Lumped kinetics (glucose consumption, maintenance rates) were extracted from a hybrid model (Narayanan 2019, 2021d, 2022b, as summarized in the 2023 review) — [Frontiers](https://www.frontiersin.org/journals/chemical-engineering/articles/10.3389/fceng.2023.1157889/full)
- Martínez et al. 2025: plotting learned f and g against growth rate recovers overflow metabolism (acetate production only above a μ threshold, roughly linear above it) and the ethanol switch in *X. dendrorhous* — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.14.688550.full)
- Pinto 2023: deep hybrid captures the **lactate production→consumption switch** in CHO-K1. The learned rates can be inspected — [PMC10515724](https://pmc.ncbi.nlm.nih.gov/articles/PMC10515724/)
- Functional-Hybrid SR (Narayanan 2022) and Riezzo 2026 SR yield closed-form kinetic expressions. Riezzo's route estimates time-varying parameter profiles first, then fits expressions — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1385894721046088); [Europe PMC query](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=(TITLE%3A%22neural%20ODE%22%20OR%20TITLE%3A%22neural%20ordinary%20differential%22%20OR%20TITLE%3A%22universal%20differential%22%20OR%20TITLE%3A%22physics-informed%22%20OR%20TITLE%3A%22SINDy%22%20OR%20TITLE%3A%22sparse%20identification%22%20OR%20TITLE%3A%22symbolic%20regression%22)%20AND%20(ABSTRACT%3A%22bioprocess%22%20OR%20ABSTRACT%3A%22bioreactor%22%20OR%20ABSTRACT%3A%22fed-batch%22%20OR%20ABSTRACT%3A%22CHO%22)%20AND%20PUB_YEAR%3A%5B2022%20TO%202026%5D&resultType=core&format=json&pageSize=20)
- Hutter 2021: learned product embeddings encode an **interpretable notion of product similarity** — [arXiv](https://arxiv.org/abs/2011.13863)
- Wang, Harcum, Xie 2026: an interpretable phase-transition model (sigmoid in culture age, qO2, pH). Lactate and glutamine terms did not significantly improve the transition fit — [PMC13309251](https://pmc.ncbi.nlm.nih.gov/articles/PMC13309251/)
- Yu et al. 2026: evolving EnKF parameter ensembles reveal **which kinetic parameters matter under new conditions and when** — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Knowledge%20transfer%20across%20cell%20lines%22%20OR%20TITLE%3A%22Physics-encoded%20transfer%20learning%22%20OR%20TITLE%3A%22transfer%20learning%20approach%20for%20predictive%20modeling%20of%20bioprocesses%22&resultType=core&format=json)
- Khare et al. 2025: kinetic-oriented dFBA hybrid with Michaelis–Menten reaction-level parameters. Reversible kinetics reproduce lactate/alanine shifts without a redox variable — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.10.28.685134v1.full)
- UDE identifiability (arXiv 2510.14140): learned NN components may not be functionally identifiable. Interpretation should be checked against identifiability — [arXiv](https://arxiv.org/pdf/2510.14140)

### Inferences
- For a 5-state dataset, the most communicable outputs are:
  - partial-dependence plots of learned μ(Glc, Lac), k_d(Glc, Lac or time), q_Glc(Glc) and q_Lac(Glc, Lac), including the sign change of q_Lac (lactate consumption);
  - a symbolic fit of each, e.g., μ = μ_max·Glc/(K+Glc)·K_I/(K_I+Lac);
  - the implied feed contributions of each unknown feed, if feed concentrations are learned parameters.
- Restricting the NN's inputs to physically meaningful states (not time) improves interpretability and extrapolation. Several CHO hybrids nonetheless include time or culture age (Richelle 2025, Wang 2026), which helps capture ageing or phase transitions at an interpretability cost.

### Gaps
- No systematic benchmark found of interpretability vs accuracy trade-offs for CHO hybrids, e.g., SR-distilled vs NN hybrid on held-out batches.

---

## 8. Public benchmark datasets and competitions for fed-batch CHO trajectory prediction

### Takeaway
There is **no established public benchmark or competition** for CHO fed-batch trajectory prediction. The main open real dataset is the **AstraZeneca/Cambridge (Gangadharan et al. 2021, Data in Brief)** collection of about 106 CHO cultures, 5–500 L, 2010–2016, ≥25 offline variables up to 17 days. Most methodological papers use proprietary industrial data (GSK, Sartorius, Boehringer, etc.) or in silico simulators. The 2026 TL review explicitly calls for standardized benchmarks.

### Cited Findings
- AstraZeneca dataset (Gangadharan et al. 2021). Contents:
  - CHO lines producing various antibodies, **2010–2016**, **106 cultures from 5 L bench-top to 500 L manufacturing**;
  - **≥25 offline parameters** (VCD, pH, osmolality, metabolites, titer) **for up to 17 days**;
  - original and imputed versions;
  - a separate subset of 45 cultures with only mid-run and harvest values, including about 10% failed runs;
  - no Raman data.

  Linked to Gangadharan's Cambridge PhD with AstraZeneca. Described as freely downloadable. Exact DOI not verified (DOAJ 403) — [arXiv 2512.03460](https://arxiv.org/html/2512.03460); [arXiv 2310.09991 review](https://arxiv.org/pdf/2310.09991); [Cambridge repository appendix](https://www.repository.cam.ac.uk/handle/1810/347248)
- "Learning From Limited Data and Feedback for Cell Culture Process Monitoring: A Comparative Study" (arXiv 2512.03460; Chem Eng J 2026) uses this public dataset to benchmark monitoring methods. It is a monitoring/soft-sensor benchmark, not t=0 trajectory forecasting — [arXiv](https://arxiv.org/html/2512.03460); [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1385894726027233)
- Anti-HER2 fed-batch dataset: 12 operating conditions (DO, pH, 2 media, temperature shifts) × 3 replicates, 12 days, daily sampling. Public raw-data availability unclear — [PMC11222931](https://pmc.ncbi.nlm.nih.gov/articles/PMC11222931/)
- The ML-in-biopharma review (arXiv 2310.09991) notes open upstream cell culture data are very limited, so many studies **generate synthetic data with mechanistic models** — [arXiv](https://arxiv.org/pdf/2310.09991)
- Synthetic benchmarks used in hybrid papers: Robitaille et al. (2015) CHO model with 9 CCD runs (Pinto 2023), and a DataHow macro-kinetic simulator with 30 LHS runs (Cruz-Bournazou 2021) — [PMC10515724](https://pmc.ncbi.nlm.nih.gov/articles/PMC10515724/); [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.12.27.474269.full)
- Code available:
  - Wang/Harcum/Xie multi-scale hybrid (CHO VRC01): [GitHub](https://github.com/kw48792/Multi-Scale-Hybrid-Model-with-Metabolic-Phase-Transitions). Data on request.
  - Martínez et al. UDE-style (microbial): [GitHub](https://github.com/carlosmvond/hybrid-modelling-of-batch-bioreactors)
- Barón Díaz et al. 2026 review: standardized benchmark datasets are a stated future need — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Bayesian%20calibration%2C%20process%20modeling%20and%20uncertainty%20quantification%22%20OR%20TITLE%3A%22Transfer%20Learning%20Approaches%20in%20Bioprocess%20Engineering%22&resultType=core&format=json)

### Inferences
- The hackathon dataset, with dozens of batches, 4 bolus feeds and VCD/TCD/Glc/Lac/V, resembles the AstraZeneca dataset in the states measured. That dataset could be used for pre-training or sanity checks, but feed records and units would need harmonization.
- In the absence of a community benchmark, results should be reported with **leave-batch-out cross-validation** and per-state NRMSE/WAPE so they are comparable with Bayer 2021 (NRMSE) and Wang 2026 (WAPE).

### Gaps
- No Kaggle-style or academic competition on CHO fed-batch trajectory prediction found, so no "winning approach" can be reported.
- The DOI and licence of the AstraZeneca Data in Brief article were not confirmed directly; the DOAJ page returned 403.

---

## 9. Synthesis: which approaches are state of the art (2022–2026), and fit to the hackathon setting

### Takeaway
State of the art for **predicting unseen CHO fed-batch trajectories from initial conditions plus feeds** is the **serial hybrid**: mass-balance ODEs, with an ML model for specific rates, integrated from t=0. Two variants are emerging:
- **(a) deep or regularized-NN hybrids**: Pinto 2023, Ramos 2024 LSTM-hybrid, Richelle 2025 digital twin;
- **(b) GP or stochastic hybrids with UQ**: Doyle 2026, Wang 2026.

Symbolic-regression distillation (Narayanan 2022, Forster 2024, Riezzo 2026) is state of the art for interpretable rate-law discovery but is not yet validated on CHO. Mixed-effects or Bayesian hierarchical grey-box models for batch variability, and t=0-only LSTM benchmarks, are open gaps.

### Cited Findings
- Deep hybrid (FFNN/LSTM) with ADAM, dropout, minibatch and sensitivities: −23.6% test WMSE vs shallow on 24-run CHO-K1, and R² 0.93 on 20-run HEK293 — [PMC10515724](https://pmc.ncbi.nlm.nih.gov/articles/PMC10515724/); [Europe PMC (Ramos 2024)](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Deep%20hybrid%20modeling%20of%20a%20HEK293%22%20OR%20TITLE%3A%22general%20deep%20hybrid%20model%20for%20bioreactor%22&resultType=core&format=json)
- GP-hybrid > Monod in intensified CHO fed-batch, with bolus-feed transients unresolved (Doyle 2026) — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Gaussian%20Process%22%20AND%20TITLE%3A%22Intensified%20Fed-Batch%22&resultType=core&format=json)
- Stochastic multi-scale hybrid with calibrated PIs: 5.7–12.1% WAPE (1–5-day look-ahead), 9 runs (Wang 2026) — [PMC13309251](https://pmc.ncbi.nlm.nih.gov/articles/PMC13309251/)
- Hybrid > mechanistic on all partitions; 4 well-chosen runs suffice to span a 4-D design space (Bayer 2023) — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE%3A%22Comparison%20of%20mechanistic%20and%20hybrid%20modeling%20approaches%22&resultType=core&format=json)
- UDE-style constrained hybrids work with as few as 3 training batches in microbial systems but fail when extrapolating to out-of-range initial substrate (Martínez 2025) — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.14.688550.full)

### Inferences (concrete design for the hackathon dataset; these are recommendations drawn from the literature, not published results)
- **States:** Xv = VCD; Xd = TCD − VCD (or lysis-aware: viable, dead, lysed, as in Richelle 2025); Glc; Lac; V. Optionally add 1–3 latent nutrient states fed by the unknown feeds.
- **Balances between feeds:** dXv/dt = (μ − k_d)·Xv; dXd/dt = k_d·Xv − k_lys·Xd; dGlc/dt = −q_Glc·Xv; dLac/dt = q_Lac·Xv. Volume is constant between samples unless there is continuous feed or sampling.
- **At each bolus:** V⁺ = V + ΔV_k, and C⁺ = (V·C + ΔV_k·C_feed,k)/V⁺. Glucose concentration is known for the glucose feed. For the 3 unknown feeds, learn C_feed,k for the latent nutrient(s), or pass cumulative feed volumes as NN inputs (cf. Bayer 2021 CPP inputs).
- **Rate model:** a small MLP or GP, [Glc, Lac, Xv, (latent nutrients or cumulative feeds)] → [μ, k_d, q_Glc, q_Lac], with sign/positivity constraints (softplus, or multiplying by the state as in Martínez 2025). q_Lac must be allowed to change sign to capture the lactate switch.
- **Training:** whole-trajectory integration from t=0 with weighted least squares across all batches; leave-batch-out CV; ensembles or restarts. Add batch-level random effects or embeddings, set to the population mean for prediction, to absorb unexplained batch variability.
- **Interpretability:** plot learned rate surfaces, then distil them via SR or SINDy with a Monod/Haldane library and compare held-out accuracy of the distilled vs NN version.
- **Baselines:** Monod/Luedeking–Piret mechanistic; black-box neural ODE or GP on time plus cumulative feeds.

### Gaps
- No public head-to-head on an open CHO dataset exists to rank these approaches objectively. The ranking above rests on per-paper comparisons with different data and metrics.
