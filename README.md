# Deconfounding Actor-Critic Network with Policy Adaptation forDynamic Treatment Regimes

This repository contains the official PyTorch implementation of the following paper:

> **Deconfounding Actor-Critic Network with Policy Adaptation forDynamic Treatment Regimes**<br>
>
> **Abstract:** *Despite intense efforts in basic and clinical research, an individualized ventilation strategy for critically ill patients remains a major challenge. Recently, dynamic treatment regime (DTR) with reinforcement learning (RL) on electronic health records (EHR) has attracted interest from both the healthcare industry and machine learning research community. However, most learned DTR policies might be biased due to the existence of confounders. Although some treatment actions non-survivors received may be helpful, if confounders cause the mortality, the training of RL models guided by long-term outcomes (e.g., 90-day mortality) would punish those treatment actions causing the learned DTR policies to be suboptimal. In this study, we develop a new deconfounding actor-critic network (DAC) to learn optimal DTR policies for patients. To alleviate confounding issues, we incorporate a patient resampling module and a confounding balance module into our actor-critic framework. To avoid punishing the effective treatment actions non-survivors received, we design a short-term reward to capture patients' immediate health state changes. Combining short-term with long-term rewards could further improve the model performance. Moreover, we introduce a policy adaptation method to successfully transfer the learned model to new-source small-scale datasets. The experimental results on one semi-synthetic and two different real-world datasets show the proposed model outperforms the state-of-the-art models. The proposed model provides individualized treatment decisions for mechanical ventilation that could improve patient outcomes.*


# Framework

Paired survivor and non-survivor patients are resampled with similar estimated mortality risks to build balanced mini-batches.
DAC adopt an actor-critic model to learn the optimal DTR policies. The longitudinal patients' data are sent to a long short-term memory network (LSTM) \cite{lstm} to generate the health state sequences.
To further remove the confounding bias, a dynamic inverse probability of treatment weighting method is introduced to assign weights to the rewards at each time step for each patient and train the actor network with the weighted rewards.

<img src="src/framework.png" width=80%>



# Data preprocessing

## List of used variables

Static variables :
Age,Gender,Weight,Readmission to intensive, care, Elixhauser score (premorbid status)


Time-varying variables:
Modified SOFA, SIRS, Glasgow coma scale, Heart rate, systolic, mean and diastolic, 
blood pressure, shock index, Respiratory rate, SpO2, Temperature 
Potassium, sodium, chloride, Glucose, BUN, creatinine, Magnesium, calcium, 
ionized calcium, carbon dioxide, SGOT, SGPT, total bilirubin, albumin, Hemoglobin, 
White blood cells count, platelets, count, PTT, PT, INR, pH, PaO2, PaCO2, base excess, 
bicarbonate, lactate, PaO2/FiO2 ratio, Mechanical ventilation, FiO2, 
IV fluid intake over 4h, vasopressor over 4h, Urine output over 4h, 
Cumulated fluid balance since admission (includes preadmission data when available)

Treatment actions:
positive end-expiratory pressure (PEEP), fraction of inspired oxygen (FiO2), 
ideal body weight-adjusted tidal volume (Vt) 

Outcome: Hospital mortality, 90-day mortality 



### MIMIC-III dataset
```
cd preprocessing
python extract_mimic_data.py
python mimic3_dataset.py
python mech_dataset.py
python find_patients_wth_mechvent.py
```


### Synthetic dataset
Simulate the all covariates, treatments and outcomes
```
cd preprocessing
python synthetic_mimic.py
```

# Train and test DAC
### Split dataset
```
cd code
python split.py
```

### Estimate mortality rate
```
cd code
python estimate_mortality.py
```


### Model training
```
cd code
python main.py
```

# Visualization of results

- Visualization of the action distribution in the 3-dimensional action space on MIMIC-III dataset.

<img src="src/MIMIC-III_PEEP_action_distribution.png" width=30%> <img src="src/MIMIC-III_TidalVolume_action_distribution.png" width=30%> <img src="src/MIMIC-III_FiO2_action_distribution.png" width=30%>


- Visualization of the action distribution in the 3-dimensional action space on AmsterdamUMCdb dataset.

<img src="src/AmsterdamUMCdb_PEEP_action_distribution.png" width=30%> <img src="src/AmsterdamUMCdb_TidalVolume_action_distribution.png" width=30%> <img src="src/AmsterdamUMCdb_FiO2_action_distribution.png" width=30%>

- The relations between mortality rates and mechanical ventilation setting difference (recommended setting - actual setting) on MIMIC-III dataset.

<img src="src/mimic.FiO2.diff.png" width=30%> <img src="src/mimic.PEEP.diff.png" width=30%> <img src="src/mimic.TidalVolume.diff.png" width=30%>

- The relations between mortality rates and mechanical ventilation setting difference (recommended setting - actual setting) on AmsterdamUMCdb dataset.

<img src="src/ast.FiO2.diff.png" width=30%> <img src="src/ast.PEEP.diff.png" width=30%> <img src="src/ast.TidalVolume.diff.png" width=30%>

- The positive correlations between estimated mortality rate and predicted mortality probability on MIMIC-III and AmsterdamUMCdb datasets. 

<img src="src/mimic.predicted_mortality.png" width=30%> <img src="src/ast.predicted_mortality.png" width=30%>

- Mortality-expected-return curve computed by the learned policies

<img src="src/mimic.q_value.png" width=30%> <img src="src/ast.q_value.png" width=30%>

