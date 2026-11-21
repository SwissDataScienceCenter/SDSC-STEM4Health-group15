# Unstructured / macroscopic kinetic (mass-balance ODE) models of CHO and mammalian fed-batch culture: classic structures and 2022–2026 developments

*How these notes were made: I read full texts (open access, via Europe PMC JATS XML) of Ben Yahia et al. 2015, Kornecki & Strube 2018, Okamura et al. 2024, Xing (Z.) et al. 2024, Sietaram et al. 2025 (MCKM), Kuchemüller et al. 2023 and 2026, Richelle et al. 2026 (two papers), Yu, del Rio Chanona & Kontoravdi 2026, Kim et al. 2026, Khare et al. 2026, Wang, Harcum & Xie 2026, and Ranpura et al. 2025. Xing et al. 2010, Craven et al. 2013, Kontoravdi et al. 2005, Zalai et al. 2015, Hartley et al. 2018, Mulukutla et al. 2012, Bayer et al. 2023, Narayanan et al. 2019, López-Meza et al. 2016 and Kyriakopoulos et al. 2018 are paywalled, so I saw only their abstracts (plus equations or parameters quoted in the open papers). In the tags, "Classic" means before 2022 and "Cutting edge" means 2022–2026.*

---

## Q1. What are the canonical model structures, and what are their exact equations for growth, death, substrate uptake (Pirt / Luedeking–Piret) and byproduct formation?

### Takeaway
The usual model is the unstructured, unsegregated **multiplicative-Monod** model. In it, μ = μmax · Π (limiting-substrate saturation terms) · Π (inhibitor terms K_I/(K_I+I)). Death μd is either constant or a function of byproducts and/or substrates. Substrate uptake follows a Pirt form, q_S = μ/Y_X/S + m_S. Lactate is usually tied to glucose uptake through a yield Y_Lac/Glc, and product formation follows Luedeking–Piret. The CHO model cited most often is Xing et al. 2010, which is fully written out in Kornecki & Strube 2018. Logistic models have fewer parameters but cannot include feeds, so they are a poor fit for bolus fed-batch prediction.

### Cited findings

**Status of the field (reviews)**
- **(Classic review)** The simple unstructured, unsegregated approach with "empirical Monod-type kinetics based on limiting substrates and inhibitory metabolites is commonly used due to its traceability and simple formalism". Most available kinetic models are small to moderate in size. — [Kyriakopoulos et al. 2018, Biotechnol J 13:e1700229, doi:10.1002/biot.201700229](https://doi.org/10.1002/biot.201700229)
- **(Classic review)** The general Monod form is `μ = μmax · Π_i S_i/(S_i + K_Si)`, with inhibition added through a term in the denominator. Under balanced growth "all other rates can be related to μ by yield coefficients". The review notes it is "often difficult to define which formulation is the best". It also warns that setting K_S to arbitrarily small values to get balanced growth in batch "will not allow to transfer such a model to continuous or fed-batch processes without readjustment of these constants". — [Ben Yahia, Malphettes & Heinzle 2015, Appl Microbiol Biotechnol 99:7009–7024, doi:10.1007/s00253-015-6743-6](https://doi.org/10.1007/s00253-015-6743-6) (OA: [PMC4536272](https://pmc.ncbi.nlm.nih.gov/articles/PMC4536272/))
- **(Cutting edge review)** For CHO monitoring, mechanistic models use "Monod-like equations to describe cell growth rates, toxicity, cell death rates, or shifts in consumption and production". Their main weakness is oversimplification and "the static nature" of their parameters. — [Ranpura et al. 2025, Comput Struct Biotechnol J, doi:10.1016/j.csbj.2025.06.035](https://doi.org/10.1016/j.csbj.2025.06.035)

**Growth-rate forms catalogued in Ben Yahia et al. 2015, Table 2 (with the reported parameters)** — [Ben Yahia 2015](https://doi.org/10.1007/s00253-015-6743-6)
- Bree et al. 1988 (hybridoma): `μ = μmax·Gln/(kGln+Gln)·kLac/(kLac+Lac)·kAmm/(kAmm+Amm)`, with μmax = 0.125 h⁻¹, kGln = 0.8 mM, kLac = 8 mM, kAmm = 1.05 mM.
- Glacken et al. 1988 (hybridoma): a cell-density-dependent saturation constant `k_s0·X^(−β)`, a Contois-like idea (k_s0 = 26.5, β = 0.21), together with an inhibition term in Amm² (kAmm = 26 mM²).
- Ozturk & Palsson 1991 (hybridoma): Monod in serum, `μ = μmax·Serum/(k_Serum+Serum)`, with μmax = 0.053 h⁻¹ and k_Serum = 0.0139 v/v. This is an early "lumped, unmeasured nutrient" formulation.
- de Tremblay et al. 1992: double Monod `μ = μmax·Glc/(kGlc+Glc)·Gln/(kGln+Gln)`, with μmax = 0.045 h⁻¹, kGlc = 1 mM, kGln = 0.3 mM.
- Dhir et al. 2000: `μ = μmax·Glc/(kGlc+Glc)·Gln/(kGln+Gln)·kLac/(kLac+Lac) − k_rd·Lac/(k_dLac+Lac) − k_A·Amm`. Here the death terms sit inside the net growth rate.
- Jang & Barford 2000 (hybridoma, batch and fed-batch): four-factor `μ = μmax·Glc/(kGlc+Glc)·Gln/(kGln+Gln)·kLac/(kLac+Lac)·kAmm/(kAmm+Amm)`, with μmax = 0.065 h⁻¹, kGlc = 0.75 mM, kGln = 0.075 mM, kLac = 90 mM, kAmm = 15 mM. Original: [Jang & Barford 2000, Biochem Eng J 4:153–168, doi:10.1016/S1369-703X(99)00041-8](https://doi.org/10.1016/S1369-703X(99)00041-8)
- **Xing et al. 2010 (CHO, 5000 L fed-batch)**: the same four-factor form, with μmax = 0.028 h⁻¹, kGlc = 0.084 mM, kGln = 0.047 mM, kLac = 43 mM, kAmm = 6.51 mM.
- Pörtner et al. 1996 (hybridoma): `μ = μmax·Gln/(kGln+Gln)`, with μmax = 0.036 h⁻¹ and kGln = 0.06 mM.
- Borchers et al. 2013 (AGE1.HN): `μ = μmax·Glc/(kGlc+Glc)`, with μmax = 0.019 h⁻¹ and kGlc = 1.45 mM.

**Logistic alternative (classic)**
- Goudar's generalised logistic models are `X(t) = A/(e^{Bt} + C·e^{−Dt})` for cells, `P(t) = A/(1 + C·e^{−Dt})` for increasing metabolites and `N(t) = A/(e^{Bt} + C)` for decreasing nutrients. They use fewer parameters and need no limiting-substrate knowledge, but "seem less suited for extrapolation and not well suited to incorporate additional information on metabolism". — [Ben Yahia 2015](https://doi.org/10.1007/s00253-015-6743-6); original: [Goudar et al. 2005, Biotechnol Prog 21:1109, doi:10.1021/bp050018j](https://doi.org/10.1021/bp050018j)
- **(Classic head-to-head)** Craven et al. compared a Monod kinetic model, a logistic model and a statistical regression model for CHO in batch, bolus fed-batch and continuous fed-batch at 3 L and 15 L. The Monod model "predicted cell density, glucose glutamine, lactate, and ammonia concentrations well" but was computationally demanding because it has many parameters. The logistic and regression models struggled with nutrient and metabolite concentrations because of "the inability of both models to incorporate a feed". — [Craven, Shirsat, Whelan & Glennon 2013, Biotechnol Prog 29:186–196, doi:10.1002/btpr.1664](https://doi.org/10.1002/btpr.1664)

**Xing et al. 2010 (classic CHO model)**
- The model has six ODEs (VCD, glucose, glutamine, ammonia, lactate, product) and 18 parameters. Six of these were calculated directly from the data and 12 were estimated by MCMC from a training set of 7 runs at 5000 L. It was validated on 2 perturbed datasets. — [Xing, Bishop, Leister & Li 2010, Biotechnol Prog 26:208–219, doi:10.1002/btpr.284](https://doi.org/10.1002/btpr.284) (abstract via [PubMed 19834967](https://pubmed.ncbi.nlm.nih.gov/19834967/))
- **Full equation set, Xing-based, as written in Kornecki & Strube 2018 (classic; daily bolus fed-batch CHO).** Feed and volume terms were added but are not printed. — [Kornecki & Strube 2018, Bioengineering 5:25, doi:10.3390/bioengineering5010025](https://doi.org/10.3390/bioengineering5010025) (OA: [PMC5874891](https://pmc.ncbi.nlm.nih.gov/articles/PMC5874891/))
  - (1) `dXv/dt = (μ − μd)·Xv`
  - (2) `μ = μmax · GLC/(Kglc+GLC) · GLN/(Kgln+GLN) · KIlac/(KIlac+LAC) · KIamm/(KIamm+AMM)`
  - (3) `μd = kd · LAC/(KDlac+LAC) · AMM/(KDamm+AMM)`
  - (4) `dGLC/dt = −[(μ − μd)/Y_X/GLC + m_glc]·Xv`
  - (5) `dGLN/dt = −[(μ − μd)/Y_X/GLN + m_gln]·Xv`, with (6) `m_gln = a1·GLN/(a2 + GLN)`
  - (7) `dLAC/dt = Y_LAC/GLC·[(μ − μd)/Y_X/GLC + m_glc]·Xv` (lactate is purely coupled to glucose, with no consumption term)
  - (8) `dAMM/dt = Y_AMM/GLN·[(μ − μd)/Y_X/GLC]·Xv − r_Amm·Xv` (printed with Y_X/GLC)
  - (9) `dmAb/dt = Q_mAb·Xv·(1 − μ/μmax)` (product formation that is not growth-associated)

**Pörtner / Möller / Kuchemüller school (classic → cutting edge)**
- Möller et al. used an "unstructured, unsegregated saturation type model, adapted from literature" for CHO-XM-111 fed-batch. Parameters were fitted with Nelder–Mead and the model was used for model-assisted DoE. — [Möller et al. 2015, BMC Proc 9(Suppl 9):P42, doi:10.1186/1753-6561-9-S9-P42](https://doi.org/10.1186/1753-6561-9-S9-P42). The mDoE found the same optimal strategies with 4 experiments as with 16 (batch) or 29 (fed-batch) in classic DoE. — [Möller et al. 2019, Bioprocess Biosyst Eng 42:867–882, doi:10.1007/s00449-019-02089-7](https://doi.org/10.1007/s00449-019-02089-7)
- The same group's equation template, adapted from their CHO model to batch L929 cells:
  - `dXt/dt = μ·Xv − K_Lys·(Xt − Xv)`
  - `μ = μmax · c_LS/(c_LS + K_S,LS) · (Xmax − Xv)/Xmax`
  - `q_Glc = μ/Y_X/Glc · c_Glc/(c_Glc + k_Glc) · (Xmax − Xv)/Xmax`
  - lactate is yield-coupled to glucose and ammonia to glutamine
  - an **unmeasured limiting substrate** `dc_LS/dt = −q_LS,max·c_LS/(c_LS + k_LS)·Xv`
  - — [Kuchemüller, Pörtner & Möller 2023, Eng Life Sci 23:e2200059, doi:10.1002/elsc.202200059](https://doi.org/10.1002/elsc.202200059)

**Threshold Monod (classic, CHO)**
- Growth fits Monod-like kinetics only "if a threshold substrate concentration value of [S]t = 0.58 g/L (for recombinant cells) and [S]t = 0.96 g/L (for naïve cells), below which growth is not observed, was considered". Y_X/S depended on substrate concentration. mAb followed Luedeking–Piret, `d[mAb]/dt = α·d[X]/dt + β·[X]`, with α = 7.65×10⁻⁷ µg/cell and β = 7.68×10⁻⁸ µg/cell/h. — [López-Meza et al. 2016, Cytotechnology 68:1287–1300, doi:10.1007/s10616-015-9889-2](https://doi.org/10.1007/s10616-015-9889-2)

**Modern "smooth switch" growth factors (cutting edge, Sartorius)**
- `μ_eff = μmax · θ_Φbio · Π θ_sub · Π θ_inh · Π θ_quad`, where:
  - `θ_sub = [tanh(2·z/k_sub)]²` (substrate limitation)
  - `θ_inh = 1/((z/(2·k_inh))³ + 1)` (inhibition)
  - `θ_quad = exp(−(1/25)·((z − k_quad)/k_quad,spread)²)` (pH or temperature optimum)
  - These forms are "adapted from structured CHO cell models developed in previous work of our group".
  - — [Kuchemüller et al. 2026, Bioprocess Biosyst Eng, doi:10.1007/s00449-026-03362-2](https://doi.org/10.1007/s00449-026-03362-2) (OA: [PMC13582380](https://pmc.ncbi.nlm.nih.gov/articles/PMC13582380/))
- A similar glucose uptake form: `v_Glc = Y_G·μ_eff·tanh(Glc/K_G)²·1/((Lac/K_I)³ + 1) + m_Glc`. — [Richelle et al. 2026, Comput Struct Biotechnol J 35:0078, doi:10.34133/csbj.0078](https://doi.org/10.34133/csbj.0078) (OA: [PMC13136614](https://pmc.ncbi.nlm.nih.gov/articles/PMC13136614/))

**Other macroscopic structures**
- **(Cutting edge)** A HEK293 macroscopic model is built on four macroscopic reactions: substrate consumption with lactate production, lactate consumption, biomass death and maintenance. The authors state it is also valid for cultures without lactate consumption. — [Pimentel et al. 2023, CoDIT 2023, doi:10.1109/CODIT58514.2023.10284210](https://doi.org/10.1109/CODIT58514.2023.10284210) (abstract via [ORBi UMONS](https://orbi.umons.ac.be/handle/20.500.12907/46828))
- **(Cutting edge)** Kim et al. 2026 built their baseline from Craven et al. 2013, took the death rate from Xing et al. 2010 "to better represent cell death kinetics driven by byproduct toxicity", and added a glutamine production term `k_Gln,prod·K_Gln,prod/(K_Gln,prod + Gln)`. — [Kim et al. 2026, Biotechnol Bioeng, doi:10.1002/bit.70300](https://doi.org/10.1002/bit.70300) (OA: [PMC13665592](https://pmc.ncbi.nlm.nih.gov/articles/PMC13665592/))
- **(Classic GSA study)** Kontoravdi et al. ran a Sobol' global sensitivity analysis on a dynamic mAb cell-culture model. "most parameters have low sensitivity indices and exhibit strong interactions with one another", and sensitivity varies over culture time. — [Kontoravdi et al. 2005, Biotechnol Prog 21:1128–1135, doi:10.1021/bp050028k](https://doi.org/10.1021/bp050028k)
- **(Classic)** Kornecki & Strube note that "excess glucose leads to substrate inhibition". This is the motivation for Haldane/Andrews forms. — [Kornecki & Strube 2018](https://doi.org/10.3390/bioengineering5010025)

### Inferences
- The user's dataset measures only VCD, TCD, one substrate, one byproduct and volume. The glutamine and ammonia parts of the Xing/Kornecki/Kontoravdi models are therefore unobservable and would be poorly identifiable. A reduced model is the natural starting point:
  - states Xv, Xd (or Xt), Glc, Lac, V
  - `μ = μmax·Glc/(Kglc+Glc)·KIlac/(KIlac+Lac)·f(N)`, where f(N) is an optional latent-nutrient term (see Q4)
  - μd from Q3
  - `q_Glc = μ/Y_X/Glc + m_Glc`
  - q_Lac with a switch (see Q2)
- Haldane/Andrews (`μ = μmax·S/(K_S + S + S²/K_I)`) and Contois (`μ = μmax·S/(K_S·X + S)`) are directly transferable from microbial modelling. A Contois-like X-dependence (Glacken 1988) or a capacity term (Xmax − Xv)/Xmax (Pörtner school) gives a phenomenological way to stop growth at high density without knowing the real limiting nutrient. This is attractive when feeds B/C/D have unknown composition.
- Logistic models (Goudar) can serve as a baseline for VCD only. They cannot respond to feed changes, so they are unsuitable for predicting unseen feed profiles (Craven 2013).

### Gaps
- I could not access the original equation listing of Xing et al. 2010 (paywalled). The Kornecki & Strube 2018 version is labelled as Xing-derived (parameters cite ref. [49] = Xing 2010) but may differ in details such as the ammonium and lactate terms.
- I did not retrieve the full equations of Jang & Barford 2000, Frahm et al. 2002 (OLFO hybridoma model), Pörtner & Schäfer 1996 (comparison of hybridoma models, [doi:10.1016/0168-1656(96)01535-0](https://doi.org/10.1016/0168-1656(96)01535-0)) or Kontoravdi et al. 2005/2007. Their death-rate forms are therefore not verified here. From recall (not verified), Kontoravdi-type models use `μd = μd,max/(1 + (Kd,amm/Amm)^n)` with lysis `k_lys·Xd`. Please check before citing.
- I found no CHO-specific fed-batch paper that uses Haldane/Andrews or Contois kinetics as its main growth law.

---

## Q2. How do models capture the lactate metabolic shift (production → consumption), and which forms work best?

### Takeaway
At least eight formulations exist: pure yield coupling with no shift, a glucose threshold, a time or phase switch, a sigmoid in glutamine with regime-dependent growth, an overflow/oxidative-capacity limit, activation constants, reversible LDH kinetics, and separate macroscopic consumption reactions. No head-to-head benchmark exists. Lactate is consistently the worst-predicted state even in the 2024–2026 models (NRMSE 33–44%, R² 0.36–0.77). The mechanistic triggers reported include glucose depletion, glutamine depletion, reduced glycolytic flux or growth cessation, high lactate, temperature or pH shifts and dissolved oxygen. They depend on the cell line and the process.

### Cited findings

**Biology of the trigger**
- **(Classic)** In NS0 fed-batch, "the shift to lactate consumption occurs upon the cessation of rapid growth and under conditions of low glycolysis flux and high extracellular lactate concentrations". High lactate inhibits PFK. Lactate consumption "is an outcome of reduced glycolysis flux". — [Mulukutla, Gramer & Hu 2012, Metab Eng 14:138–149, doi:10.1016/j.ymben.2011.12.006](https://doi.org/10.1016/j.ymben.2011.12.006)
- **(Classic review)** The mechanism "remains enigmatic". The review discusses pH, NAD⁺/NADH, pyruvate availability and mitochondrial function, and puts forward a cellular-redox hypothesis. — [Hartley, Walker, Chung & Morten 2018, Biotechnol Bioeng 115:1890–1903, doi:10.1002/bit.26603](https://doi.org/10.1002/bit.26603)
- **(Classic)** In a CHO fed-batch DoE, the switch to lactate uptake "was mainly triggered by the decrease in the glycolytic flux" and could be controlled over a broad range by the timing of pH and temperature shifts. Late lactate uptake correlated with TCA activity. — [Zalai et al. 2015, Biotechnol Prog 31:1657–1668, doi:10.1002/btpr.2179](https://doi.org/10.1002/btpr.2179)
- **(Classic)** Lactate consumption "is observed during the stationary phase upon depletion of glutamine". — [Kornecki & Strube 2018](https://doi.org/10.3390/bioengineering5010025)
- **(Cutting edge)** At high DO, Q_lac and Q_gln fell as glutamine dropped. Q_gln stabilised near 0 at about 1 mM Gln, after which "the specific lactate production rate slowed, followed by a phase of lactate consumption". At low DO, both rates were insensitive to glutamine. — [Okamura et al. 2024, Biotechnol Prog 40:e3486, doi:10.1002/btpr.3486](https://doi.org/10.1002/btpr.3486) (OA: [PMC11659809](https://pmc.ncbi.nlm.nih.gov/articles/PMC11659809/))
- **(Cutting edge)** The shift "is prompted by the depletion of glutamine", which reduces TCA influx and NADH. Lactate inhibits HK and PFK and also glutaminase. — [Wang, Harcum & Xie 2026, Biotechnol Bioeng, doi:10.1002/bit.70205](https://doi.org/10.1002/bit.70205) (OA: [PMC13309251](https://pmc.ncbi.nlm.nih.gov/articles/PMC13309251/))
- **(Cutting edge)** Glucose depletion was identified as the main cause, citing Hartley 2018 and Martínez 2013. In 656 Ambr15 runs, lactate correlated significantly with glucose profiles. — [Sietaram, Kotidis, Finka & Lapkin 2025, J Ind Microbiol Biotechnol 52:kuaf029, doi:10.1093/jimb/kuaf029](https://doi.org/10.1093/jimb/kuaf029) (OA: [PMC12486387](https://pmc.ncbi.nlm.nih.gov/articles/PMC12486387/))
- **(Cutting edge)** Net lactate consumption appeared after a temperature downshift to 32 °C. The shift is cell-line specific: Cell Line B consumed lactate under the same feed under which Cell Line A accumulated it. — [Yu, del Rio Chanona & Kontoravdi 2026, Biotechnol Bioeng, doi:10.1002/bit.70269](https://doi.org/10.1002/bit.70269) (OA: [PMC13576805](https://pmc.ncbi.nlm.nih.gov/articles/PMC13576805/))

**Formulations (equations as published)**
1. **No shift (yield coupling).** `dLAC/dt = Y_LAC/GLC·q_GLC·Xv` ([Kornecki & Strube 2018](https://doi.org/10.3390/bioengineering5010025), Xing-derived). Okamura et al. note that such models "cannot follow the shifts in lactate metabolic behavior" ([Okamura 2024](https://doi.org/10.1002/btpr.3486)).
2. **Glucose threshold (MCKM, cutting edge).** "a minimum glucose threshold below which cells transition to lactate consumption". The threshold a_glc is a fixed constant, set as the average glucose concentration at lactate-consumption onset across the 656 cultures. The model captures parabolic lactate profiles and also cell lines without a switch. It "only partially" captures a double switch. Lactate NRMSE was 33.5–44.4% across campaigns. The authors state "no kinetic model in the literature of CHO incorporates an accurate mathematical description of the lactate switch". — [Sietaram et al. 2025](https://doi.org/10.1093/jimb/kuaf029)
3. **Time switch + Monod consumption (cutting edge).** Production phase: `dc_Lac/dt = Y_Lac/Glc·[ (r·Xt − (kD + kL)·Xv)/Y_Xv/Glc + m_Glc·Xv ]`. After day 5 a consumption term `− q_Lac·Xv` is added, with `q_Lac = q_Lac,max·c_Lac/(K_Lac + c_Lac)`. — [Xing Z., Nguyen, Kanai-Bai, Yamano-Adachi & Omasa 2024, Cytotechnology, doi:10.1007/s10616-023-00598-8](https://doi.org/10.1007/s10616-023-00598-8) (OA: [PMC10828271](https://pmc.ncbi.nlm.nih.gov/articles/PMC10828271/))
4. **Phase-wise parameter sets (cutting edge).** Estimation is split at Day 7.6 "because constant parameters are insufficient to capture the dynamic behavior and metabolic shifts of CHO cells, particularly for lactate and ammonium (Ben Yahia et al. 2017)" ([Kim et al. 2026](https://doi.org/10.1002/bit.70300)). Similarly, Wang et al. used "two sets of parameter values" before and after a temperature shift (as cited in [Okamura 2024](https://doi.org/10.1002/btpr.3486)).
5. **Sigmoid in glutamine + regime-dependent growth, with lactate as a carbon source (cutting edge).** — [Okamura et al. 2024](https://doi.org/10.1002/btpr.3486)
   - `Q_lac = k_lac11/(k_lac12 + exp(−k_lac13·GLN)) + k_lac14` at high DO, and `Q_lac = k_lac21` at low DO. Q_gln uses the analogous form.
   - μ is piecewise:
     - `μmax1·GLC/(Kglc+GLC)·1/(1+exp(−GLN/Kgln))·KIlac1/(KIlac1+LAC)` when DO is high and Q_lac > 0
     - **`μmax1·GLC/(Kglc+GLC)·LAC/(Klac+LAC)`** when DO is high and Q_lac ≤ 0 (lactate enhances growth once it is being consumed)
     - `μmax2·GLC/(Kglc+GLC)·KIlac2/(KIlac2+LAC)` at low DO
   - Fitted values (Exp. A): k_lac11 = 2.63×10⁻¹¹ mmol/cell/h, k_lac12 = 0.065, k_lac13 = 1.19 L/mmol, k_lac14 = −6.41×10⁻¹¹ mmol/cell/h, Klac = 6.46 mM.
6. **Overflow / oxidative-capacity limit (cutting edge).** — [Richelle et al. 2026 CSBJ](https://doi.org/10.34133/csbj.0078)
   - `v_Lac,Prod = max(0, v_Glc − v_Ox,max)`
   - `v_Lac,Cons = max(0, v_Ox,max − v_Glc)·tanh(Lac/K_Lac)²`
   - `v_Lac = Y_Lac,Prod·v_Lac,Prod − Y_Lac,Cons·v_Lac,Cons`
   - Across 23 fed-batch runs, lactate had R² = 0.36, versus 0.91–0.98 for the other metabolites. The authors attribute this to "limited ability to capture lactate re-accumulation observed during the later culture stages".
7. **Activation constants (Kotidis/Kontoravdi model, cutting edge use).** Lac_max1 and Lac_max2 "regulate the activation of lactate consumption", and Y_X/Lac represents growth-coupled lactate use. The two activation constants show posterior correlation, so their individual contributions "cannot be fully resolved independently from data alone". — [Yu et al. 2026](https://doi.org/10.1002/bit.70269); base model: [Kotidis et al. 2019, Biotechnol Bioeng 116:1612–1626, doi:10.1002/bit.26960](https://doi.org/10.1002/bit.26960)
8. **Reversible LDH kinetics in a hybrid kinetic–MFA model (cutting edge).** The kinetic-oriented model "successfully predicts the shift from production to consumption for both lactate and alanine … through the inclusion of reverse reaction terms" (vmax2r, Km2lac), without the earlier discontinuous "redox variable". It captured the trend "albeit with underpredictions of the levels". — [Khare et al. 2026, npj Syst Biol Appl, doi:10.1038/s41540-026-00703-5](https://doi.org/10.1038/s41540-026-00703-5) (OA: [PMC13550590](https://pmc.ncbi.nlm.nih.gov/articles/PMC13550590/))
9. Alhuthali & Kontoravdi used "a variable lactate production term as a function of glucose consumption and a constant lactate consumption term" (as cited in [Okamura 2024](https://doi.org/10.1002/btpr.3486)). Pimentel et al. 2023 use a separate macroscopic "lactate consumption" reaction ([Pimentel 2023](https://doi.org/10.1109/CODIT58514.2023.10284210)).

**Predictive accuracy on lactate**
- Okamura 2024: fit R² = 0.772; prediction on an unseen run R² = 0.674. Lactate was the second-worst state after DNA. — [Okamura 2024](https://doi.org/10.1002/btpr.3486)
- MCKM: lactate NRMSE 33.5–44.4% (fits, not held-out). — [Sietaram 2025](https://doi.org/10.1093/jimb/kuaf029)
- Richelle 2026: lactate R² = 0.36 (direct validation). — [Richelle 2026 CSBJ](https://doi.org/10.34133/csbj.0078)
- Kornecki 2018: lactate prediction R² ≥ 0.90, in a process without a strong switch. — [Kornecki & Strube 2018](https://doi.org/10.3390/bioengineering5010025)

### Inferences
- The user's dataset measures glucose but neither glutamine nor DO-dependent rates. Glucose-triggered switches (MCKM) or growth- or nutrient-state-triggered switches (Mulukutla 2012) are the only directly usable triggers. Glutamine-triggered switches (Okamura, Wang 2026) need a latent nutrient state to stand in for glutamine and amino acids.
- Two useful refinements of MCKM's hard threshold:
  - A smooth, differentiable version: `q_Lac = Y_L/G·q_Glc·H(Glc) − q_Lac,max·Lac/(K_L + Lac)·(1 − H(Glc))`, with `H(Glc) = Glc^n/(K_sw^n + Glc^n)`. This gives stable gradients and identifiable parameters (K_sw, n).
  - Making the switch depend on growth (`1 − μ/μmax`) or on a latent nutrient N. This matches Mulukutla's "cessation of rapid growth" and would matter if glucose is kept high by bolus feeding, so that glucose never depletes.
- The overflow form (Richelle) is the most mechanistically motivated single-parameter switch (v_Ox,max). With the user's feeds it needs few parameters, but its reported R² = 0.36 shows that late re-accumulation is not captured.
- Okamura's sigmoid gives net production of about 0.34 pmol/cell/h at high Gln and net consumption of about 0.04 pmol/cell/h at Gln = 0. These numbers are my own calculation from their parameters and give a sense of the size of the production and consumption rates.
- Since no benchmark exists, the switch form should be chosen empirically with leave-batch-out validation. The comparison should include at least: (a) no switch, (b) glucose-Hill switch, (c) growth-dependent switch, (d) overflow. Choose by lactate RMSE on held-out batches.

### Gaps
- I found no study that benchmarks several lactate-switch formulations on the same CHO dataset. The MCKM authors make the same point ([Sietaram 2025](https://doi.org/10.1093/jimb/kuaf029)). The BioPreDyn-bench CHO problem is a batch process in which lactate is only a byproduct, so it does not test the switch (per search snippet of [Villaverde et al., arXiv:1407.5856](https://arxiv.org/pdf/1407.5856)).
- I could not see the equations of Zalai et al. 2015 or Hartley et al. 2018, which are paywalled. Hartley is a review, and Zalai combined MFA with PLS-R. Neither appears to propose a compact ODE switch term usable here.
- The MCKM a_glc threshold value is not given in the text I accessed.

---

## Q3. How are cell death and lysis modelled so that TCD vs VCD (dead-cell accumulation, lysis) are reproduced?

### Takeaway
Death-rate forms are: (i) constant; (ii) byproduct-driven with Monod factors in Lac and Amm (Xing 2010); (iii) threshold-ammonia-driven (Kontoravdi 2007, Xing 2024); (iv) substrate-depletion-driven (Okamura 2024, Pörtner school); and (v) cutting-edge "toxicity/age" forms driven by an accumulated biomaterial (∫Xv dt) or by lysed cells (Sartorius 2026). Dead cells are tracked as Xd (or Xt) with lysis that is first-order in Xd, Michaelis–Menten in Xd, or direct from viable cells. Trypan-blue TCD counts dead but not lysed cells.

### Cited findings

**Death-rate forms**
- **Byproduct-driven (classic Xing-type).** `μd = kd·LAC/(KDlac + LAC)·AMM/(KDamm + AMM)`, with kd = 0.004 h⁻¹ (experimental), KDlac = 45.8 mM and KDamm = 6.51 mM (from Xing 2010). — [Kornecki & Strube 2018](https://doi.org/10.3390/bioengineering5010025). Kim et al. 2026 adopted Xing's death rate to capture "cell death kinetics driven by byproduct toxicity" ([Kim 2026](https://doi.org/10.1002/bit.70300)).
- **Byproduct + glucose-starvation (cutting edge).** — [Okamura 2024](https://doi.org/10.1002/btpr.3486)
  - `μd = kd·LAC/(KDlac + LAC)·AMM/(KDamm + AMM)·KDI_glc/(KDI_glc + GLC)`, with separate kd and KDlac values at high and low DO.
  - Fitted values: kd1 = 0.0515 / 0.0449 h⁻¹, KDlac1 = 12.6 / 9.61 mM, KDamm = 6.96 / 6.03 mM, KDI_glc = 0.582 / 0.806 mM (Exp. A / C).
- **Ammonia-threshold, phase-dependent (cutting edge, after Kontoravdi 2007).** During the death phase, if c_Amm > c_Amm,cr = 5 mM, then `kD = kD,dead·(K_DAmm + c_Amm − c_Amm,cr)/K_DAmm`; otherwise `kD = kD,min`. — [Xing Z. et al. 2024](https://doi.org/10.1007/s10616-023-00598-8)
- **Ammonium only (cutting edge).** Death is a Monod-type function of ammonium alone. Lactate is excluded, and lysis is not modelled. Mean kd = 0.0116–0.0185 h⁻¹ against a literature value of 0.016 h⁻¹. — [Sietaram 2025](https://doi.org/10.1093/jimb/kuaf029)
- **Substrate-dependent (Pörtner school).** As extracted from Kuchemüller 2023: `μd = μd,min + μd,max·c_LS/(c_LS + K_d,LS)`, with c_LS an unmeasured limiting substrate. — [Kuchemüller 2023](https://doi.org/10.1002/elsc.202200059). *Caution:* as transcribed, death rises with substrate. The original may use K_d,LS/(K_d,LS + c_LS), i.e. death rising on depletion. Check the paper.
- **Toxicity/age from accumulated biomaterial (cutting edge).** — [Richelle 2026 CSBJ](https://doi.org/10.34133/csbj.0078)
  - `dXv/dt = (μ_eff − μd − μl)·Xv − (F/V)·Xv`
  - `dXd/dt = μd·Xv − (F/V)·Xd`
  - `dXl/dt = μl·Xv − (F/V)·Xl`
  - `dγ_X/dt = Xv − (F/V)·γ_X`
  - `μd = kd + k_Td·γ_X`, `μl = kl + k_Tl·Xl`
  - "Cell death and lysis are driven by the accumulation of metabolic by-products, represented by a catch-all 'biomaterial' variable, γ_X." Because "trypan blue exclusion does not distinguish between dead and lysed cells", Xl is kept as an internal, unmeasured variable.
  - *Note:* the dilution signs above follow the mass balance; the extracted text lost brackets.
- **Lysed-cell toxicity (cutting edge, HEK293).** `μdeath = k_death + k_toxic·x_LCD`, and an accumulated biomaterial Φ_bio (`dΦ_bio/dt = x_VCD − dilution`) inhibits growth through θ_Φbio. — [Kuchemüller et al. 2026](https://doi.org/10.1007/s00449-026-03362-2)
- **Death inside the net growth rate (classic, Dhir 2000).** `μ = … − k_rd·Lac/(k_dLac + Lac) − k_A·Amm`. — [Ben Yahia 2015](https://doi.org/10.1007/s00253-015-6743-6)

**Lysis formulations**
- **First-order on (Xt − Xv) (Pörtner school).** `dXt/dt = μ·Xv − K_Lys·(Xt − Xv)`. — [Kuchemüller 2023](https://doi.org/10.1002/elsc.202200059)
- **Michaelis–Menten lysis of dead cells (cutting edge).** `d(V·Xd)/dt = μd·V·Xv − r_dcell·V·Xd/(K_ldcell + Xd)`, coupled to HCP and DNA release with yields Y_hcp/Xd and Y_dna/Xd. Dead-cell density fit R² = 0.966 and prediction R² = 0.955 on an unseen run. — [Okamura 2024](https://doi.org/10.1002/btpr.3486)
- **Two lysis routes, from dead and from viable cells (cutting edge, after Kroll 2017).** — [Xing Z. et al. 2024](https://doi.org/10.1007/s10616-023-00598-8)
  - `dXd/dt = kD·Xv − kDL·Xd`
  - `dXl/dt = kL·Xv + kDL·Xd`
  - `dXv/dt = r·Xt − (kD + kL)·Xv`
  - with a logistic total-cell growth `dXt/dt = r·Xt`, `r = r_max·(1 − Xt/Xt,max)`
  - Fitted: kDL = 6.9–8.8×10⁻³ h⁻¹, kL = 1.3–1.4×10⁻³ h⁻¹, kD,min = 1.7–2.5×10⁻⁴ h⁻¹, kD,dead = 3.3–5.1×10⁻⁴ h⁻¹, K_DAmm = 0.6 mM. Viable and dead cell densities were reproduced with R² > 0.95.
- **Dead and lysed populations with dilution (cutting edge).** `dx_DCD/dt = μdeath·x_VCD − (k_lyse + F_bleed/V + dV/(V·dt))·x_DCD` and `dx_LCD/dt = k_lyse·x_DCD − (…)·x_LCD`. — [Kuchemüller 2026](https://doi.org/10.1007/s00449-026-03362-2)
- **Measuring lysis.** Lysis can be quantified from supernatant dsDNA. Xing et al. 2024 adapted the method of [Klein et al. 2015, J Biotechnol 207:67–76, doi:10.1016/j.jbiotec.2015.04.021](https://doi.org/10.1016/j.jbiotec.2015.04.021) ([Xing Z. 2024](https://doi.org/10.1007/s10616-023-00598-8)). Turbidity (total cells) stopped tracking VCD after about 234 h, once viability fell more than 10%. — [Kornecki & Strube 2018](https://doi.org/10.3390/bioengineering5010025)

### Inferences
- For the user's states (VCD, TCD), the minimal set-up is `dXv/dt = (μ − μd)·Xv`, `dXd/dt = μd·Xv − k_lys·Xd`, with TCD = Xv + Xd (lysed cells disappear from counts), plus bolus dilution of both. A late TCD plateau or decline is then explained by k_lys. Without k_lys, TCD can only increase, which typically fails in late culture.
- Which death driver to choose depends on what is measured. Ammonium is not measured, so Xing/Kontoravdi ammonia terms cannot be identified. The candidates are:
  - (a) lactate-driven: `kd·Lac/(KD + Lac)`
  - (b) glucose-starvation: `kd·KDI/(KDI + Glc)` (Okamura)
  - (c) age/toxicity: `kd + k_T·∫Xv dt`, i.e. an IVCD-driven term (Richelle, Kuchemüller 2026), which can stand in for unmeasured ammonium and other inhibitors
  - (d) latent-nutrient exhaustion
  - Additive combinations such as `μd = kd0 + kd,lac·f(Lac) + kd,starv·g(Glc)` are more robust than products of Monod terms, because a product forces death to zero when one factor is zero.
- The IVCD-based "biomaterial" term is attractive for interpretability and transfer. It is a single scalar state, needs no unknown feed composition, and both Sartorius 2026 papers use it.

### Gaps
- I found no comparative study of death-rate formulations on the same CHO data.
- I could not verify the original Kontoravdi 2005/2007 or Jang & Barford 2000 death and lysis equations (paywalled).

---

## Q4. How are fed-batch mass balances written for bolus feeds with volume change and sampling, and how are feeds of unknown composition handled?

### Takeaway
Models either write balances on total amounts, d(V·C)/dt (Okamura), or use concentration form with a dilution term −(F/V)·C or −(dV/dt)/V·C (Kuchemüller 2026, Richelle 2026). Bolus feeds are handled as impulse additions V_F,i·c_F/V (Xing 2024), or by integrating between events and re-initialising (MCKM). Sampling changes volume but not concentrations. Unknown or unmeasured feed components are handled by a lumped "limiting substrate" state (Pörtner school; serum in Ozturk & Palsson), a catch-all accumulated "biomaterial" state (Sartorius), or are simply left out, which is acknowledged as a source of error. Exact feed volumes are critical: averaging feeds destroyed glucose prediction in MCKM.

### Cited findings
- **Amount-based balances, general fed-batch/perfusion (cutting edge).** — [Okamura 2024](https://doi.org/10.1002/btpr.3486); a review table of the same generic balances is in [Ranpura et al. 2025](https://doi.org/10.1016/j.csbj.2025.06.035)
  - `d(V·Xv)/dt = (μ − μd)·V·Xv − F_bleed·Xv`
  - `d(V·GLC)/dt = −(μ/Y_Xv/glc + m_glc)·V·Xv + F_in·c_in,glc + F_suppl·c_suppl − (F_harvest + F_bleed)·GLC`
  - `d(V·LAC)/dt = Q_lac·V·Xv − (F_harvest + F_bleed)·LAC`
  - `dV/dt = F_in + F_suppl − (F_harvest + F_bleed)`
- **Concentration-based balances with explicit dilution (cutting edge).** — [Kuchemüller 2026](https://doi.org/10.1007/s00449-026-03362-2); [Richelle 2026 CSBJ](https://doi.org/10.34133/csbj.0078)
  - `dx_VCD/dt = (μ_eff − μdeath − F_bleed/V − dV/(V·dt))·x_VCD`
  - `dGlc/dt = v_Glc·Xv + (F/V)·(Glc_in − Glc)`
  - `dLac/dt = v_Lac·Xv − (F/V)·Lac`
- **Impulsive bolus terms (cutting edge).** `dc_Glc/dt = −[growth + maintenance] + V_F,i·c_M/V + V_G,i·c_G/V`. Here V_F,i and V_G,i are the volumes of feed medium and of glucose supplement at the i-th feeding, and c_M, c_G are their glucose concentrations. The authors describe this as "modified from a study by Xing et al." — [Xing Z. et al. 2024](https://doi.org/10.1007/s10616-023-00598-8)
- **Segment-wise integration (cutting edge).** "the fed-batch problem is solved as successive batch segments between the seven sampling times". Feed is added at each sampling time and the objective is the sum of the segment errors. The authors used one run's feeding schedule for all runs, and glucose fits were poor: mean R² = −0.24 ± 0.48 and NRMSE about 50% for mAb-A. They list sensitivity to "averaged feed amounts" as a limitation. — [Sietaram 2025](https://doi.org/10.1093/jimb/kuaf029)
- **Pseudo-concentrations (cutting edge).** MetRaC "transforms raw concentration measurements into pseudo-concentrations that account for distortions caused by bioreactor volume changes (e.g., feed additions, sample withdrawals)". It models rates as linear combinations of basis functions inside a Bayesian / nested-sampling framework, and stays robust under sparse, noisy data. — [Richelle et al. 2026, Bioprocess Biosyst Eng 49:2147–2161, doi:10.1007/s00449-026-03382-y](https://doi.org/10.1007/s00449-026-03382-y) (OA: [PMC13424222](https://pmc.ncbi.nlm.nih.gov/articles/PMC13424222/))
- **Feed effects outside the model (classic).** "Since the cell cultures were performed in fed-batch mode with daily bolus feed additions, the model equations were extended by feeding terms. Consequently, the bioreactor volume was considered as well." Glucose prediction in the feed phase was "significantly more challenging, since physiological effects of the feed media on the cellular state … are not described by this model". — [Kornecki & Strube 2018](https://doi.org/10.3390/bioengineering5010025)
- **Logistic and regression models cannot include a feed (classic).** — [Craven 2013](https://doi.org/10.1002/btpr.1664)
- **Lumped unmeasured limiting substrate (classic → cutting edge).**
  - `dc_LS/dt = −q_LS·Xv` with `q_LS = q_LS,max·c_LS/(c_LS + k_LS)`, entering μ as `c_LS/(c_LS + K_S,LS)`, "unmeasured, e.g., serum". — [Kuchemüller 2023](https://doi.org/10.1002/elsc.202200059)
  - Serum-Monod μ (Ozturk & Palsson 1991, k_Serum = 0.0139 v/v). — [Ben Yahia 2015](https://doi.org/10.1007/s00253-015-6743-6)
- **Catch-all accumulated variable (cutting edge).** γ_X or Φ_bio integrates Xv and is diluted by feeds. It represents "accumulation of unknown biomaterial" that inhibits growth or drives death. — [Richelle 2026 CSBJ](https://doi.org/10.34133/csbj.0078); [Kuchemüller 2026](https://doi.org/10.1007/s00449-026-03362-2)
- **Unrepresented proprietary feed (cutting edge).** A glucose misalignment at 32 °C "may be attributed to uncertainty in the composition of the proprietary Feed X … its compositional uncertainty was not explicitly represented in the EnKF". — [Yu et al. 2026](https://doi.org/10.1002/bit.70269)
- **Closest published analogue to the user's dataset (cutting edge).** — [Richelle 2026 CSBJ](https://doi.org/10.34133/csbj.0078)
  - 23 Ambr15 CHO-S fed-batch runs.
  - FMA and FMB carry the full nutrient complement, split only to avoid solubility issues, and are added daily from day 3. FMG is a 400 g/L glucose stock that keeps glucose above 5 g/L.
  - 8 FMA+FMB formulations were tested, with volumes varied across runs. "each of the 23 fed-batch cultures represents a unique process condition".
- **A second analogue (cutting edge).** — [Kim 2026](https://doi.org/10.1002/bit.70300)
  - 14-day Ambr15 runs with bolus feeds of 5% v/v on days 2.6, 4.6, 6.6, 8.6, 10.5 and 12.6, and glucose continuously supplemented to stay at or above 3 g/L.
  - 8 batches with different pH and temperature setpoints.

### Inferences
- **Exact bolus jump conditions** (my derivation from the mass balance). At a bolus event t_k with volumes ΔV_j,k of feeds j ∈ {S, B, C, D}:
  - `V⁺ = V⁻ + Σ_j ΔV_j,k`
  - `C⁺ = (C⁻·V⁻ + Σ_j c_j·ΔV_j,k)/V⁺` for each dissolved species
  - `X⁺ = X⁻·V⁻/V⁺` for Xv and Xd (feeds carry no cells)
  - At a sampling event, V⁺ = V⁻ − V_sample and concentrations are unchanged.
  - Integrate the continuous ODE between events with a stiff or adaptive solver, restarting at every event (e.g. scipy `solve_ivp` segments, diffrax, or SUNDIALS with discontinuities). This is equivalent to MCKM's segment-wise approach, but uses the true per-run feed volumes.
  - Because cell concentrations drop on dilution, VCD and TCD trajectories have sawtooth drops at feed times. Fitting requires knowing whether samples were taken before or after the feed (Wang 2026 sampled "prior to feeding", [Wang 2026](https://doi.org/10.1002/bit.70205)).
- **Latent nutrient pool for feeds B/C/D of unknown composition.** Add a state N (dimensionless, N(0) = 1 for basal medium) with `dN/dt = −(μ/Y_X/N)·Xv` (optionally plus a maintenance term). Each bolus of feed j adds `γ_j·ΔV_j/V⁺`, where γ_j is an unknown "nutrient richness" of feed j relative to basal medium. N enters μ as N/(K_N + N) and/or drives death and the lactate switch.
  - Identifiability: Y_X/N and the γ_j trade off against N(0), so fix N(0) = 1 or fix one γ_j. Only γ_B, γ_C, γ_D and K_N remain.
  - Variant: a feed that may be inhibitory (e.g. high osmolality) can enter as a separate inhibitor state, again with an unknown per-volume loading.
- If feeds B/C/D contain glucose, their glucose content enters the jump condition for the measured substrate. It can be estimated as an extra unknown concentration c_Glc,j, or fixed to 0 if a dedicated substrate feed exists, as in the user's case.

### Gaps
- I found no CHO paper that explicitly estimates the "nutrient value" of several unknown-composition feeds as latent parameters. The latent-pool construction above is an inference built from the Pörtner limiting-substrate and Sartorius biomaterial ideas.

---

## Q5. What parameter values and ranges are reported for CHO (μmax, kd, Ks, Y_X/glc, Y_lac/glc, maintenance, etc.)?

### Takeaway
For CHO fed-batch: μmax ≈ 0.025–0.045 h⁻¹ (0.6–1.1 d⁻¹), kd ≈ 0.004–0.02 h⁻¹, Kglc from below 0.1 mM to a few mM (poorly identifiable), Y_X/Glc ≈ 0.36–0.49×10⁹ cells/mmol, Y_Lac/Glc ≈ 0.3 g/g (≈0.57 mol/mol) to 0.7 mol/mol, ammonium death constant ≈ 6–7 mM, and dead-cell lysis ≈ 7–9×10⁻³ h⁻¹. Values are strongly model- and clone-specific.

### Cited findings (CHO unless noted)

| Parameter | Value(s) | Context / source |
|---|---|---|
| μmax | 0.028 h⁻¹ | Xing 2010, 5000 L CHO ([Ben Yahia 2015 Table 2](https://doi.org/10.1007/s00253-015-6743-6)) |
| μmax | 0.039 h⁻¹ (exp.) | Kornecki & Strube 2018 ([link](https://doi.org/10.3390/bioengineering5010025)) |
| μmax | 0.0258 ± 0.0083 h⁻¹; bounds 0.01–0.1 | MCKM, 656 Ambr15 runs, mAb-A; "consistent across clones" ([Sietaram 2025](https://doi.org/10.1093/jimb/kuaf029)) |
| r_max (logistic on Xt) | 0.042–0.044 h⁻¹ | CHO-S VLP, 2 bioreactors ([Xing Z. 2024](https://doi.org/10.1007/s10616-023-00598-8)) |
| μmax1 / μmax2 | 0.271–0.282 / 0.296 h⁻¹ (paired with Kglc 14–37 mM and a Gln sigmoid, so the effective μ is far lower) | CHO-MK high-density, 50 L shaken tank ([Okamura 2024](https://doi.org/10.1002/btpr.3486)) |
| μmax (hybridoma, for comparison) | 0.036–0.125 h⁻¹ | [Ben Yahia 2015 Table 2](https://doi.org/10.1007/s00253-015-6743-6) |
| kd | 0.004 h⁻¹ (max, multiplied by Lac/Amm Monod terms) | [Kornecki & Strube 2018](https://doi.org/10.3390/bioengineering5010025) |
| kd | 0.018 ± 0.019 h⁻¹ (mAb-A); campaign means 0.0116–0.0185 h⁻¹; literature value 0.016 h⁻¹ | [Sietaram 2025](https://doi.org/10.1093/jimb/kuaf029) |
| kd1 / kd2 | 0.0515 / 0.0315 h⁻¹ (max) | [Okamura 2024](https://doi.org/10.1002/btpr.3486) |
| kD,min / kD,dead | 1.7–2.5×10⁻⁴ / 3.3–5.1×10⁻⁴ h⁻¹ | [Xing Z. 2024](https://doi.org/10.1007/s10616-023-00598-8) |
| Lysis of dead cells kDL | 6.9–8.8×10⁻³ h⁻¹ | [Xing Z. 2024](https://doi.org/10.1007/s10616-023-00598-8) |
| Direct lysis of viable cells kL | 1.3–1.4×10⁻³ h⁻¹ | [Xing Z. 2024](https://doi.org/10.1007/s10616-023-00598-8) |
| Kglc | 0.084 mM | Xing 2010 ([Ben Yahia 2015](https://doi.org/10.1007/s00253-015-6743-6)) |
| Kglc | 1.00 mM | Kornecki (from lit.) ([link](https://doi.org/10.3390/bioengineering5010025)) |
| Kglc | 1.05 ± 2.60 g/L (≈5.8 mM; very variable); bounds 0.1–10 g/L | [Sietaram 2025](https://doi.org/10.1093/jimb/kuaf029) |
| Kglc | 36.7 / 14.1 mM (Exp. A / C, −61.6%) | [Okamura 2024](https://doi.org/10.1002/btpr.3486) |
| Glucose growth threshold S_t | 0.58 g/L (recombinant), 0.96 g/L (naive) | [López-Meza 2016](https://doi.org/10.1007/s10616-015-9889-2) |
| Kgln | 0.047 mM | Xing 2010 ([Kornecki](https://doi.org/10.3390/bioengineering5010025)) |
| KIlac (growth) | 43 mM (Xing-type); 2.7–3.9 mM (Okamura, different structure) | [Kornecki](https://doi.org/10.3390/bioengineering5010025); [Okamura](https://doi.org/10.1002/btpr.3486) |
| KIamm (growth) | 6.51 mM | Xing 2010 ([Kornecki](https://doi.org/10.3390/bioengineering5010025)) |
| KDlac (death) | 45.8 mM (Xing); 9.6–12.6 mM (Okamura) | as above |
| KDamm (death) | 6.51 mM (Xing); 6.47 ± 3.14 mM (MCKM; noted as higher than the 1.44–4.5 mM in the literature); 6.0–7.0 mM (Okamura) | [Kornecki](https://doi.org/10.3390/bioengineering5010025); [Sietaram](https://doi.org/10.1093/jimb/kuaf029); [Okamura](https://doi.org/10.1002/btpr.3486) |
| KDI_glc (starvation death) | 0.58–0.81 mM | [Okamura 2024](https://doi.org/10.1002/btpr.3486) |
| Critical ammonia for accelerated death | 5 mM (Kontoravdi 2007) | [Xing Z. 2024](https://doi.org/10.1007/s10616-023-00598-8) |
| Y_X/Glc | 0.357×10⁹ cells/mmol | [Kornecki](https://doi.org/10.3390/bioengineering5010025) |
| Y_X/Glc | 2.03 ± 1.29×10⁹ cells/g (bounds 0.1–10) | [Sietaram 2025](https://doi.org/10.1093/jimb/kuaf029) |
| Y_X/Glc | 4.89 / 4.83×10⁸ cells/mmol | [Okamura 2024](https://doi.org/10.1002/btpr.3486) |
| Y_X/S (glucose, López-Meza) | 0.72–2.79×10⁶ cells/mL per (g/L), recombinant; depends on S | [López-Meza 2016](https://doi.org/10.1007/s10616-015-9889-2) |
| Y_Lac/Glc | 0.70 mol/mol | [Kornecki](https://doi.org/10.3390/bioengineering5010025) |
| Y_Lac/Glc | 0.286 ± 0.158 g/g (bounds 0.1–2; "comparatively stable") | [Sietaram 2025](https://doi.org/10.1093/jimb/kuaf029) |
| m_glc | 69.2×10⁻¹² mmol/cell/h (Xing) | [Kornecki](https://doi.org/10.3390/bioengineering5010025) |
| m_glc | 1.5×10⁻⁶ ± 2.1×10⁻³ (not reliably estimated) | [Sietaram 2025](https://doi.org/10.1093/jimb/kuaf029) |
| Lactate sigmoid (Okamura) | k_lac11 2.63×10⁻¹¹ mmol/cell/h; k_lac12 0.065; k_lac13 1.19 L/mmol; k_lac14 −6.41×10⁻¹¹ mmol/cell/h; K_lac (growth on lactate) 6.46 mM | [Okamura 2024](https://doi.org/10.1002/btpr.3486) |
| Y_Amm/Gln, r_amm, a1, a2 | 0.67 mol/mol; 6.3×10⁻¹² mmol/cell/h; 3.2×10⁻¹² mmol/cell/h; 2.1 mM | Xing 2010 ([Kornecki](https://doi.org/10.3390/bioengineering5010025)) |
| Luedeking–Piret mAb | α = 7.65×10⁻⁷ µg/cell; β = 7.68×10⁻⁸ µg/cell/h | [López-Meza 2016](https://doi.org/10.1007/s10616-015-9889-2) |

### Inferences
- **Cross-check (my calculation).** MCKM's Y_X/Glc of 2.03×10⁹ cells/g equals about 0.37×10⁹ cells/mmol (×0.180 g/mmol). This agrees with Kornecki's 0.357×10⁹ cells/mmol. MCKM's Y_Lac/Glc of 0.286 g/g is about 0.57 mol/mol, compared with Kornecki's 0.70 mol/mol. These give reasonable priors: Y_X/Glc ~ 0.3–0.5×10⁹ cells/mmol and Y_Lac/Glc ~ 0.5–1.5 mol/mol. Values above 1 can occur if lactate also comes from glutamine.
- **Suggested prior ranges** for a Bayesian or bounded fit: μmax 0.01–0.06 h⁻¹; kd 10⁻⁴–0.03 h⁻¹; k_lys 10⁻³–2×10⁻² h⁻¹; Kglc 0.05–10 mM (log-uniform); KIlac 5–100 mM.
- Okamura's very large μmax (0.27 h⁻¹) paired with a very large Kglc (36.7 mM) is a typical sign of μmax–Kglc compensation (see Q6). Compare raw parameter values across papers only together with the model structure they belong to.

### Gaps
- I could not access a consolidated CHO parameter table from Kyriakopoulos et al. 2018 or Ben Yahia et al. 2017 (paywalled). The table above is assembled from the open papers only.

---

## Q6. What is known about parameter identifiability and about generalising to unseen batches with batch-to-batch variability (mixed-effects / hierarchical)?

### Takeaway
Identifiability is the main weak point. μmax–Ks, yield pairs (Y_Lac/Glc–Y_X/Lac) and lactate-switch constants are collinear, and most parameters show low sensitivity with strong interactions. Parameters shift by more than 35% between scales and reactors. Published "validation" is mostly per-run refitting rather than true out-of-sample prediction. The 2026 approaches to transfer are recursive Bayesian updating (EnKF), per-clone or per-batch parameter distributions (MCKM), leave-batch-out-validated hybrids and stochastic multi-scale models. I found **no CHO paper applying formal nonlinear mixed-effects (NLME) or hierarchical Bayesian ODE models**. That remains an open niche, with methods available from pharmacometrics and microbial growth modelling.

### Cited findings

**Identifiability and sensitivity**
- **(Classic)** In GSA, most parameters "have low sensitivity indices and exhibit strong interactions with one another" and can be fixed at nominal values. Sensitivity changes with culture time. — [Kontoravdi 2005](https://doi.org/10.1021/bp050028k)
- **(Classic)** Structural identifiability of yield coefficients when the reaction rates are unknown. — [Chen & Bastin 1996, Math Biosci 132:35–67, doi:10.1016/0025-5564(95)00048-8](https://doi.org/10.1016/0025-5564(95)00048-8) (title as cited in [Ben Yahia 2015](https://doi.org/10.1007/s00253-015-6743-6))
- **(Cutting edge)** Collinearity indices for MCKM: μmax–Kglc CI = 14.33, below the threshold of 20 but "slight co-linearity"; Y_Lac/Glc–Y_X/Lac CI = 5.8×10⁶; Y_Gln/Glu–Y_X/Gln CI = ∞. An automated balancing step fixes μmax or Kglc when the R² of a fit falls below 0.90. m_glc is "not reliably estimated". Across clones, μmax is consistent ("host property") while Kglc, Y_X/Gln and Y_X/Glu vary widely. — [Sietaram 2025](https://doi.org/10.1093/jimb/kuaf029)
- **(Classic)** The "experimentally determined parameters had the greatest influence on model performance. They changed with scale and mode of operation, but were easily calculated". The remaining parameters fitted by differential evolution "were not as crucial". — [Craven 2013](https://doi.org/10.1002/btpr.1664)
- **(Classic)** μmax "may be the key parameter of this macroscopic model". The authors suggest MCMC for parameters that are hard to obtain (m_glc, Kgln, Kglc). — [Kornecki & Strube 2018](https://doi.org/10.3390/bioengineering5010025)
- **(Cutting edge)** Between reactor types, Kglc changed by −61.6% and KDI_glc by +38.5%. Standard errors were obtained from the inverse Hessian. — [Okamura 2024](https://doi.org/10.1002/btpr.3486)
- **(Cutting edge)** Parameter values are "highly system-specific due to biological variability across cell lines and operating conditions". The broad search spaces this requires lead to local minima. The authors use LHS + CMA-ES + L-BFGS-B with LLM-guided bound updates. Each of the 8 batches was "calibrated independently" with two phases. — [Kim et al. 2026](https://doi.org/10.1002/bit.70300)
- **(Cutting edge)** Parameters that show little reduction in ensemble uncertainty "may be weakly identifiable or insensitive". Posterior correlation means individual contributions "cannot be fully resolved independently from data alone". — [Yu et al. 2026](https://doi.org/10.1002/bit.70269)
- **(Cutting edge review)** Measurement-error models combined with mechanistic models for likelihood-based identifiability and prediction (Murphy, Maclaren & Simpson 2024, J R Soc Interface, [doi:10.1098/rsif.2023.0402](https://doi.org/10.1098/rsif.2023.0402)), as cited in [Ranpura 2025](https://doi.org/10.1016/j.csbj.2025.06.035).

**Generalisation and out-of-sample evidence**
- **(Classic)** "The extrapolation power of a model cannot be predicted a priori". The probability of good extrapolation increases "if physically meaningful functions are used". The review distinguishes direct validation from validation on an independent dataset. — [Ben Yahia 2015](https://doi.org/10.1007/s00253-015-6743-6)
- **(Cutting edge)** True held-out prediction, with parameters fitted on Exp. A and used on Exp. B: VCD R² = 0.707, glucose 0.832, lactate 0.674, dead cells 0.955, mAb 0.950. — [Okamura 2024](https://doi.org/10.1002/btpr.3486)
- **(Cutting edge)** MCKM has no held-out validation. Each run is regressed on its own data (biomass R² 0.96 ± 0.07, NRMSE 6–7%). — [Sietaram 2025](https://doi.org/10.1093/jimb/kuaf029)
- **(Cutting edge)** Two-model comparison on 33 CHO fed-batch experiments in a 4-factor design with four data partitions. The mechanistic model shows "the advantage of prior knowledge … relatively independently of the utilized data partition". The hybrid model "displayes a higher data dependency but simultaneously yielded a higher accuracy on all data partitions". Four well-chosen initial experiments can represent the full design space. — [Bayer, Duerkop, Pörtner & Möller 2023, Biotechnol J 18:e2200381, doi:10.1002/biot.202200381](https://doi.org/10.1002/biot.202200381)
- **(Classic)** On a 3.5 L CHO fed-batch dataset, hybrid models predicted the time evolution of process variables "using only the initial and process conditions" better than statistical models, with better robustness and extrapolation. — [Narayanan, Sokolov, Morbidelli & Butté 2019, Biotechnol Bioeng 116:2540–2549, doi:10.1002/bit.27097](https://doi.org/10.1002/bit.27097)
- **(Cutting edge)** EnKF dual state–parameter estimation moved a Kotidis-2019-based model (8 states, 24 parameters) across 6 datasets: scale, temperature downshift, a second cell line and three feeds. It did this "without reparameterization, using only a single experimental dataset". Open-loop forecasts improved as more data arrived, started at days 7, 9 and 12. This requires in-run measurements and is not a prediction from t = 0. — [Yu et al. 2026](https://doi.org/10.1002/bit.70269)
- **(Cutting edge)** For 23 runs, the authors used leave-one-batch-out (LOBO) and leave-one-media-group-out (LOMO) validation. Generalisation "remained comparable to that obtained under direct validation". The digital twin predicts VCD, metabolites and titre "solely from initial" conditions and feeds. — [Richelle 2026 CSBJ](https://doi.org/10.34133/csbj.0078)
- **(Cutting edge)** A stochastic multi-scale model with probabilistic metabolic-phase transitions gives "uncertainty-aware estimates of batch-to-batch variation" and predicts from online measurements and initial conditions. — [Wang, Harcum & Xie 2026](https://doi.org/10.1002/bit.70205)
- **(Cutting edge)** Kinetic-oriented model parameters fitted on a control run predicted an alternative dataset with different initial conditions and feeding, VCD and antibody "with reasonable accuracy". Slight parameter adjustments were needed under lactate supplementation. — [Khare et al. 2026](https://doi.org/10.1038/s41540-026-00703-5)
- **(Classic)** Product identity is represented by learned embedding vectors in GP-based hybrid models for cross-product transfer. — [Hutter, von Stosch, Cruz Bournazou & Butté 2021, Biotechnol Bioeng 118:4389–4401, doi:10.1002/bit.27907](https://doi.org/10.1002/bit.27907)

**Mixed-effects / hierarchical tools from neighbouring fields**
- Bayesian calibration and uncertainty quantification for bioprocess data, with Python packages `calibr8` and `murefi` (high-throughput microbial cultivations). — [Helleckes et al. 2022, PLoS Comput Biol 18:e1009223, doi:10.1371/journal.pcbi.1009223](https://doi.org/10.1371/journal.pcbi.1009223)
- NLME separates population parameters from batch effects in transfection kinetics. — [Fröhlich et al. 2018, npj Syst Biol Appl, doi:10.1038/s41540-018-0079-7](https://www.nature.com/articles/s41540-018-0079-7)
- NLME for cell-to-cell variability (yeast Mig1), framed as the pharmacokinetics method family. — [Llamosi et al. 2016, PLoS Comput Biol (PMC4404321)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4404321/)
- Hierarchical Bayesian growth-curve models with batch and replicate random effects that "correct for batch effects". — [Tonner et al. 2020, PLoS Comput Biol (PMC7644099)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7644099/)

### Inferences
- **The prediction task is out-of-sample from t = 0 using only feeds.** Random effects for a new batch cannot be estimated from that batch. A hierarchical model therefore helps in three ways:
  - (i) It gives better population (typical-batch) parameters, by shrinking noisy per-batch estimates.
  - (ii) It gives calibrated predictive uncertainty, by integrating over η ~ N(0, Ω).
  - (iii) If batch-level covariates exist (e.g. total feed volumes, inoculum viability, initial glucose), it can explain part of the variability through covariate models such as `log μmax,b = log μmax,pop + β·covariate_b + η_b`.
- **Practical structure.** Use `θ_b = θ_pop·exp(η_b)`, with η_b only on 2–4 parameters known to vary: μmax (or a growth-capacity term), kd or k_tox, the lactate-switch threshold, and Y_X/Glc. Keep all other parameters global. Fit with Stan, NumPyro or diffrax+BlackJAX, or with a two-stage approach (MCKM-style per-batch fits, then a population distribution).
- **Collinearity to manage.** Fix or put a tight prior on Kglc (glucose is mostly far above Ks in fed-batch, so Kglc is weakly identifiable). Do not let both Y_Lac/Glc and a lactate-uptake yield float freely. Fix N(0) of any latent nutrient pool.
- **Evaluation.** Use leave-one-batch-out or grouped k-fold over batches, with metrics per state (RMSE or NRMSE on VCD, TCD, Glc, Lac). Most published numbers are in-sample fits, which would overstate performance for this task.

### Gaps
- I found no CHO or mammalian fed-batch study that uses NLME or hierarchical Bayesian ODE kinetics for batch-to-batch variability. One extended search returned only single-cell and microbial analogues.
- There is no quantitative comparison of the generalisation error of different unstructured models on a common CHO held-out set.

---

## Q7. Which recent (2022–2026) papers propose improved unstructured models or benchmark several of them on the same data, and which is considered best?

### Takeaway
There is no consensus "best" unstructured CHO model and no standard benchmark. Recent work goes in five directions:
- (a) generalised single-structure kinetic models fitted at scale (MCKM 2025, 656 runs)
- (b) regime-switching models with explicit lactate-shift and lysis/impurity terms (Okamura 2024, Xing Z. 2024, Kim 2026)
- (c) catch-all "biomaterial"/toxicity states and smooth tanh/cubic switch functions (Sartorius 2026)
- (d) adaptive or Bayesian parameter handling (EnKF 2026, MetRaC 2026, LLM-guided optimisation 2026)
- (e) hybrid models, which beat purely mechanistic models in accuracy when enough data exist (Bayer 2023; Narayanan 2019 classic; Richelle 2026)

Direct comparisons (Craven 2013; Bayer 2023; Khare 2026) favour Monod-type kinetic structures over logistic or statistical ones, and kinetic-oriented over stoichiometric-oriented models. Hybrids are more accurate but depend more on data.

### Cited findings (chronological; all cutting edge unless marked)
- **Bayer et al. 2023, Biotechnol J.** Mechanistic vs hybrid on the same 33-run CHO DoE. Hybrid more accurate on all partitions; mechanistic more robust to the data partition. — [doi:10.1002/biot.202200381](https://doi.org/10.1002/biot.202200381)
- **Kuchemüller, Pörtner & Möller 2023, Eng Life Sci.** mDoE with a Pörtner-type model: unmeasured limiting substrate, death, lysis to Xt, and a capacity term. Fit R² = 0.80 for Xv and above 0.4 for lactate and ammonia. — [doi:10.1002/elsc.202200059](https://doi.org/10.1002/elsc.202200059)
- **Pimentel et al. 2023, CoDIT.** Four-reaction macroscopic model with lactate consumption (HEK293). — [doi:10.1109/CODIT58514.2023.10284210](https://doi.org/10.1109/CODIT58514.2023.10284210)
- **Okamura et al. 2024, Biotechnol Prog.** DO- and glutamine-dependent lactate shift (sigmoid), lactate as a growth substrate, glucose-starvation death, Michaelis–Menten lysis, HCP/DNA release. Held-out R² for VCD 0.71, glucose 0.83, lactate 0.67, dead cells 0.96. — [doi:10.1002/btpr.3486](https://doi.org/10.1002/btpr.3486)
- **Xing Z. et al. 2024, Cytotechnology.** Logistic Xt growth chosen "to prevent overfitting because it had fewer parameters than Monod-type equations". Death, two lysis routes, a time-switched lactate consumption, and bolus impulse terms. — [doi:10.1007/s10616-023-00598-8](https://doi.org/10.1007/s10616-023-00598-8)
- **Sietaram et al. 2025 (MCKM), J Ind Microbiol Biotechnol.** 13-parameter generalised model with a glucose-threshold lactate switch and Pirt/Luedeking–Piret forms. Fitted per run to 656 Ambr15 runs, with collinearity analysis. Biomass NRMSE 6–7%, which the authors compare with 10–20% for hybrid and transfer models (an in-sample versus out-of-sample caveat applies). — [doi:10.1093/jimb/kuaf029](https://doi.org/10.1093/jimb/kuaf029)
- **Ranpura et al. 2025, CSBJ.** Review of CHO modelling in the digital-biomanufacturing era. — [doi:10.1016/j.csbj.2025.06.035](https://doi.org/10.1016/j.csbj.2025.06.035)
- **Richelle et al. 2026, Bioprocess Biosyst Eng (MetRaC).** Bayesian rate estimation with pseudo-concentrations that correct for feeds and sampling. — [doi:10.1007/s00449-026-03382-y](https://doi.org/10.1007/s00449-026-03382-y)
- **Richelle et al. 2026, CSBJ.** Hybrid digital twin: NN growth rate + ODE metabolite model (tanh/cubic kinetics, overflow lactate) + PC-dFBA. Covers 23 runs with LOBO/LOMO validation. The pure ODE layer had lactate R² of only 0.36. — [doi:10.34133/csbj.0078](https://doi.org/10.34133/csbj.0078)
- **Kuchemüller et al. 2026, Bioprocess Biosyst Eng.** HEK293 mechanistic model with viable, dead and lysed cells, lysed-cell toxicity and an accumulated-biomaterial inhibitor. — [doi:10.1007/s00449-026-03362-2](https://doi.org/10.1007/s00449-026-03362-2)
- **Yu, del Rio Chanona & Kontoravdi 2026, Biotechnol Bioeng.** EnKF knowledge transfer across cell lines, scales and feeds. — [doi:10.1002/bit.70269](https://doi.org/10.1002/bit.70269)
- **Kim et al. 2026, Biotechnol Bioeng.** Craven 2013 + Xing 2010 death model with two-phase parameters, using LLM-guided global optimisation on 8 Ambr15 bolus-fed batches. — [doi:10.1002/bit.70300](https://doi.org/10.1002/bit.70300)
- **Khare et al. 2026, npj Syst Biol Appl.** Kinetic-oriented vs stoichiometric-oriented hybrid CHO models. "KOM outperformed the SOM in predicting viable cell density, antibody production, and multiple amino acids and metabolites" across two fed-batch conditions, and captured the lactate and alanine production-to-consumption shifts. — [doi:10.1038/s41540-026-00703-5](https://doi.org/10.1038/s41540-026-00703-5)
- **Wang, Harcum & Xie 2026, Biotechnol Bioeng.** Multi-scale stochastic hybrid with probabilistic growth → stationary → decline phase transitions. Predicts long-term trajectories from online data and initial conditions, with uncertainty. — [doi:10.1002/bit.70205](https://doi.org/10.1002/bit.70205)
- **Craven et al. 2013 (classic benchmark).** Monod > logistic and regression for fed-batch, because the latter cannot incorporate feeds. — [doi:10.1002/btpr.1664](https://doi.org/10.1002/btpr.1664)

### Inferences

**Recommended interpretable model for the user's task.** This is a synthesis and needs validating on the user's data. It combines the most transferable elements from the literature:

1. **States:** Xv, Xd, Glc, Lac, V, plus optionally N (latent nutrient from feeds B/C/D) and I = ∫Xv dt (biomaterial/age).
2. **Growth:** `μ = μmax · Glc/(Kglc + Glc) · N/(K_N + N) · K_IL/(K_IL + Lac) · 1/(1 + (I/K_I)^m)`
   - Drop factors that do not improve held-out error.
   - Kglc can be fixed (e.g. about 1 mM) because of collinearity.
3. **Death:** `μd = kd0 + kd,L·Lac/(K_DL + Lac) + kd,S·K_DS/(K_DS + Glc) [+ k_T·I]`
4. **Dead-cell lysis:** `dXd/dt = μd·Xv − k_lys·Xd`, with TCD = Xv + Xd.
5. **Glucose:** `q_Glc = μ/Y_X/Glc + m_Glc` (Pirt).
6. **Lactate:** `q_Lac = Y_L/G·q_Glc·H − q_L,max·Lac/(K_L + Lac)·(1 − H)`, with H a Hill function of Glc (or of μ/μmax or N). Alternatively use overflow, `q_Lac,prod = max(0, q_Glc − q_ox)`.
7. **Feeds:** bolus jump conditions as in Q4. Feeds B/C/D are parameterised by unknown nutrient loadings γ_B, γ_C, γ_D (and glucose content if needed).
8. **Variability:** hierarchical random effects on μmax, kd (or k_T), the lactate switch and Y_X/Glc. Population-mean prediction for unseen batches, with predictive intervals.

**Model ladder for leave-batch-out selection.**
- M0: Xing/Kornecki reduced to Glc and Lac
- M1: + lactate switch
- M2: + dead-cell lysis
- M3: + latent nutrient N for B/C/D
- M4: + biomaterial/age toxicity
- M5: + hierarchical parameters

Published evidence suggests the largest gains will come from:
- exact handling of the feeds (MCKM's averaged-feed glucose failure)
- a lactate switch (lactate is the weakest state in all recent papers)
- a lysis term so that TCD is reproduced

If a hybrid is allowed later, Bayer 2023 and Narayanan 2019 indicate that a data-driven correction on μ and/or q_Lac (keeping the mass balances) is the most accurate next step. Richelle 2026 shows this can keep interpretability through mass balances.

### Gaps
- No standardised public benchmark dataset exists for CHO fed-batch unstructured models, and I found no multi-model leaderboard. Statements about which model is "best" are therefore setting-specific.
- I could not verify full-text details of Narayanan 2019 or Bayer 2023 (dataset sizes beyond the abstracts, and the exact mechanistic equations used by Bayer 2023). These are paywalled.
- Bayer et al. 2023 appears to use the Möller/Pörtner mechanistic model, given the co-authors, but I have not verified this.
