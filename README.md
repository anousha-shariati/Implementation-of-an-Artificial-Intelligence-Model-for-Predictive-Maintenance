# Implementation-of-an-Artificial-Intelligence-Model-for-Predictive-Maintenance

This project focuses on developing a predictive maintenance (PdM) model to anticipate repair needs for industrial equipment, particularly electric motors. Predictive maintenance leverages AI and deep learning algorithms to analyze real-time data from sensors, alerting technicians to potential issues before they result in costly downtime. This project aims to reduce maintenance costs, minimize unexpected breakdowns, and improve operational efficiency.

Predictive maintenance relies on AI and Time-series analysis to assess equipment health and predict breakdowns. Sensors track crucial parameters such as temperature, vibration, and current, which provide key insights into equipment health. Vibration, in particular, often shows the earliest signs of impending failure, making it a critical parameter in predictive models.

Project Objectives:

1. Data analysis & parameter monitoring: work with industrial sensor streams (e.g., vibration, temperature, current) to assess machine health.

2. AI model design: develop anomaly-detection and early-warning models (classical + deep learning).

3. Optimal parameter selection: identify the most predictive sensors/features for motor failures.

4. Sampling strategy: reason about window length, stride, and sampling rate vs. performance/latency.

Methods at a glance:

1. Feature selection: RandomForest importances (+ optional permutation importance) to pick top sensors.

2. Anomaly detection: z-score outliers, moving-average gradient anomalies, threshold sweep for early warnings.

3. Ensemble detectors: Mahalanobis (Empirical Covariance), PCA (T² + SPE), LOF (novelty mode) with recall-first policy and event-aware metrics (FP/day, lead time).

4. LSTM early-warning: sequence model on top features; label alarm=2 in the pre-failure window (e.g., 24h→1min), alarm=0 otherwise; 


Dataset:

Multivariate time-series with columns like:

timestamp, machine_status ∈ {NORMAL, RECOVERING, BROKEN}

sensor_00, sensor_04, sensor_06, sensor_11, sensor_12, …

In Colab, the notebook reads sensor.csv from Google Drive and can download it via kagglehub if missing.

Prerequisites:

1. Python 3.10+

2. ML/DS: TensorFlow 2.x, scikit-learn, NumPy, Pandas, Matplotlib, Seaborn

3. Access to an industrial time-series sensor dataset (sensor.csv)



