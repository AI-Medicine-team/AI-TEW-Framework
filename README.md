# AI-TEW-Framework
An Artificial Intelligence-Powered Tiered Early Warning Framework Addressing Class Imbalance and High False Alarm Rates for In-Hospital Mortality Prediction in the Emergency Department.

## 1. ML Training and Feature Selection
#### 1.1. ML_feature_selection_trend.ipynb:
Machine learning model training and feature selection for IHM prediction.

## 2. Downsampling Ratio Analysis
#### 2.1. plot_downsampling_trend.ipynb:
Analyze the impact of changes in the downsampling ratio of unbalanced datasets on the evaluation metrics of the model.

## 3. Patient Risk Stratification Method
#### 3.1. get_risk_stratification_result.ipynb:
Analyze changes in NPV, PPV, and related clinical metrics under different risk-stratification thresholds.
#### 3.2. youden_for_risk_level.py:
Youden index and index calculation under different thresholds.

## 4. Risk Stratification Decision Analysis
#### 4.1. plot_DCA_and threshold.ipynb: 
The three-dimensional profit function design for risk stratification and the calculation and drawing of the corresponding surfaces in the research.
#### 4.2. plot_risk_stractification_result.ipynb:
Compare the changes in NPV and PPV before and after stratification.

## 5. LLM False Alarm Filtering Layer
#### 5.1. high_risk_alert_filter_llms.ipynb:
Six large language models are used to filter out false alarms for high-risk patients.
#### 5.2. plot_llms_alert_ppv.ipynb:
Evaluate the effect of filtering false alarms in the large language model.
#### 5.3. llm_classify_risk_level.ipynb:
Directly use the large language model for risk classification.
