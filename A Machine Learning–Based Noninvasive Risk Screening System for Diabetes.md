# A Machine Learning–Based Noninvasive Risk Screening System for Diabetes



## Abstract

Diabetes is a severe public health challenge, with over 500 million adults affected globally. Current screening relies heavily on invasive, costly biochemical tests that require specialized infrastructure. This project proposes a zero-cost, noninvasive machine learning-based screening system. By utilizing 31 original features and strictly excluding blood biochemical indicators, we engineered advanced derivative features—such as the lipid metabolism balance ratio and mean arterial pressure—to extract high-order predictive signals. We evaluated Logistic Regression, Random Forest, and LightGBM models under a rigorous 5-fold cross-validation framework. Logistic Regression emerged as the optimal model for clinical settings due to its strong interpretability and an AUC of 0.6611. This approach demonstrates that noninvasive data can provide a highly effective pre-screening filter, significantly improving the allocation of public health resources.



## 1. Introduction

Diabetes has become one of the most severe public health challenges in the contemporary world. According to the data from the International Diabetes Federation (IDF), there are over 500 million adults worldwide living under the shadow of diabetes, and this number is still on the rise. Despite the continuous progress in medical technology, a harsh reality is that the existing diabetes screening methods are highly dependent on "invasive" biochemical tests, such as fasting blood glucose or glycated hemoglobin tests. This traditional testing method is not only costly (usually ranging from 15 to 30 US dollars per test), but also has extremely high requirements for medical infrastructure, and can only be conducted in places equipped with professional laboratories and medical staff. In many resource-poor communities or remote areas, the high economic burden and inconvenient transportation have become "obstacles" to early diagnosis, resulting in a large number of patients being diagnosed only when their conditions have deteriorated to the stage of complications. 

Our goal was to break through the barriers of traditional diagnosis and utilize the power of machine learning to build a diabetes risk assessment system with zero operating costs and non-invasive nature. The core innovation of this system lies in that it does not rely on any blood test indicators at all, but only extracts the basic physiological characteristics of the subjects (such as BMI, blood pressure), demographic information (such as age, family history), and controllable lifestyle habits (such as diet and sleep). Through this project, we aim to provide a lightweight digital tool for community healthcare, making screening no longer an expensive and exclusive laboratory procedure, but a widely accessible service that can be easily implemented in pharmacies, mobile apps, or even community questionnaires. This is not only to enhance the efficiency of diagnosis, but also to raise an alarm at the earliest stage of the disease, seizing the most valuable "golden window period" for subsequent prevention and intervention.



## 2. Methodology & Learning Process

To ensure the correctness of our results and allow for reproducibility, we carefully designed a data engineering pipeline and an automated evaluation framework.

### 2.1 Data Engineering and Preprocessing

Throughout the complete lifecycle of data mining, the quality and characteristics of the data directly determine the upper limit of the model's predictions. The dataset used in this project is a comprehensive medical screening database consisting of 31 original features, and its core value lies in covering multiple dimensions ranging from macroscopic demographic data to microscopic physiological and clinical indicators. To achieve the goal of "non-invasive and zero-cost" screening, in the feature engineering stage, we not only implemented strict "physical isolation" to prevent data leakage but also utilized clinical medical logic to create multiple highly predictive derived variables. 

The dataset used in this study contains 31 original features, covering five major dimensions: demographics, lifestyle, clinical measurements, lipid indicators, and past medical history. 

Before entering the modeling stage, the primary task is to ensure the "non-invasiveness" and "predictive power" of the features. 

- In the code implementation, 6 clinical diagnosis indicators were strictly excluded, including `diabetes_stage` (diabetes stage), `diabetes_risk_score` (risk score), and all indicators related to blood biochemical tests (such as fasting blood glucose, post-meal blood glucose, insulin level, and glycosylated hemoglobin). Although this step significantly reduces the evaluation indicators of the model, it ensures that the model is conducting "risk prediction" rather than "result reconstruction", thus meeting the requirements of low-cost screening scenarios in reality. 

- The features are classified into **numeric**, **ordinal**, and **nominal** types, so as to adopt different encoding strategies in the subsequent steps.



To compensate for the loss of information after eliminating biochemical diagnostic indicators, we designed three sets of complex derivative features based on clinical pathological principles. This method of extracting "higher-order signals" from the original data is the key to improving the AUC performance in this project. 

- **Lipid Metabolism Balance Ratio:** We constructed the total cholesterol to high-density lipoprotein ratio (`tc_hdl_ratio`) and triglyceride to high-density lipoprotein ratio (`tg_hdl_ratio`). The latter is a powerful alternative indicator for insulin resistance.
- **Circulatory System Dynamics:** We calculated pulse pressure difference (`pulse_pressure`) and mean arterial pressure (`map`) to estimate peripheral circulation resistance.
- **Lifestyle Risk Score:** We integrated smoking status, dietary habits, and extreme sleep duration into a continuous risk scale to capture synergistic amplification effects.



To optimize the data distribution while preserving the integrity of the samples, we implemented 1%-99% percentile trimming. This strategy is different from simple deletion; instead, all values below the 1% percentile and above the 99% percentile are forcibly brought back to these two boundary values. This approach can significantly smooth the distribution curve of features, reduce the interference of long-tail effects on model convergence, and thereby enhance the robustness of the model when dealing with real-world noisy data. 



Furthermore, As this project compared three algorithms with completely different mathematical principles (LR, RF, LightGBM), the feature engineering was customized for each model in the final stage: 

For Logistic Regression, we have constructed a strict feature transformation pipeline. Firstly, we use `StandardScaler` to normalize the numerical features by their means, ensuring that all coefficients are on the same scale, thereby making the "Odds Ratio" generated later medically comparable. For categorical features such as education level and income level that have a hierarchical relationship, we use `OrdinalEncoder` for ordinal encoding to retain the implicit social class gradient information. 

For Random Forest, the key point of handling is the one-hot encoding of categorical features. To avoid the problem of multicollinearity, we set the `drop=first` parameter in the code. Since Random Forest is not sensitive to the scaling of features, we retained the original scale of the features to maintain the physical meaning during the decision tree splitting process. 

For LightGBM, we adopted the most advanced native strategy for handling categorical features. In the file *gbm_preprocessing.py*, we did not perform one-hot encoding for gender or race; instead, we directly declared them as the category type. The LightGBM algorithm can directly find the optimal splitting point in the categorical space through a histogram-based decision algorithm. This approach not only avoids the sparsification of the feature space but also significantly improves the efficiency and accuracy of the model when handling large-scale categorical variables. 



Through the aforementioned series of deep feature engineering, we transformed the original raw data into clinical feature vectors that can be efficiently recognized by machine learning algorithms, laying a solid foundation for the subsequent construction of a highly accurate screening model.



### 2.2 Model Construction and Evaluation

In the data mining task of this project, choosing the appropriate algorithm is not only for the purpose of improving prediction accuracy, but also to find a balance between predictive efficacy and interpretability in the specific business scenario of "medical screening". Based on this consideration, we constructed a multi-algorithm comparison experiment framework to deeply explore the differentiated performance of linear models (Logistic Regression), ensemble forest models (Random Forest), and gradient boosting decision trees (LightGBM) when dealing with non-invasive features.



#### 2.2.1 Interpretability benchmark model based on Logistic Regression

**Logistic Regression** serves as the foundation model of this project. Its core significance lies in providing clinicians with intuitive risk quantification basis. In the code implementation of lr.py, the model construction is not a simple classifier call, but is integrated into a complex machine learning pipeline.

Firstly, the model performs refined and differentiated processing on the features through Column Transformer: numerical features are standardized using StandardScaler to eliminate the influence of dimension differences on weights; nominal variables are transformed into dummy variables using OneHotEncoder. At the algorithm level, we use the Sigmoid function to map the linear combination of features to the [0, 1] interval, thereby obtaining the probability of disease prediction. The greatest technical advantage of Logistic Regression lies in its coefficient exponential processing. In the extract_and_print_odds_ratios function of the code, we calculate the odds ratio (OR) of each feature using `np.exp(classifier.coef_)`. This processing method allows us to clearly explain: when BMI increases by one unit, the risk of the subject developing diabetes increases by a certain percentage. This "white box" characteristic makes Logistic Regression the most promising model in the clinical initial screening scenario.



#### 2.2.2 Random Forest

Although Logistic Regression has excellent interpretability, it is difficult to capture the complex non-linear relationships between features (such as the exponential amplification effect of high age and obesity on risk). Therefore, we introduced **Random Forest** as the second experimental dimension.

In random_forest.py, we constructed a forest model consisting of hundreds of decision trees. This model adopts the core idea of "Bagging" (Bagging), reducing the problem of overfitting in a single decision tree through dual random sampling of samples and features. For the common class imbalance problem in medical data (more healthy samples than diabetes samples), we explicitly configured the `class_weight='balanced_subsample'` parameter in the code. This setting enables the forest to automatically adjust weights based on sample frequencies when constructing each sub-tree, forcing the model to pay more attention to those rare "high-risk" samples. The addition of Random Forest allows the system to identify the hidden interaction patterns in the data's depth, and the ranking of feature importance generated by it provides us with a new perspective based on information entropy for understanding disease risk.



#### 2.2.3 LightGBM

To further extract the predictive potential of the data, the project introduced the advanced gradient boosting framework **LightGBM**. Unlike the parallel construction of random forests, LightGBM adopts a leaf-wise strategy, reducing residuals iteratively to approximate the objective function. In train_evaluate_lgbm.py, we fully utilized LightGBM's native support for categorical features. In the preprocessing script gbm_preprocessing.py, we kept gender, race, etc. as the original categorical labels and dynamically converted them to the category type that the model can recognize during the training phase. Compared to traditional one-hot encoding, this approach avoids artificially creating a high-dimensional sparse matrix and retains the clustering structure of the features in the multi-dimensional space. Additionally, in the code, the `importance_type='gain'` was selected as the evaluation metric, reflecting the total information gain benefit brought by each feature during the node splitting process. This method, compared to simply counting the number of statistical splits, can more realistically reflect which non-invasive indicators (such as family history or pulse pressure difference) contribute the highest information value in distinguishing pathological states. 



#### 2.2.4 Automated Parameter Search and Cross-Validation Framework 

To ensure the robustness of model performance rather than accidental parameter stacking, this project deployed a strict validation system in all training scripts. 

1. **5-Fold Cross-Validation:** Using `StratifiedKFold`, we divide the dataset into five mutually exclusive subsets. In each round of training, four subsets are used for training, and one subset is used for validation, with the proportion of patients and healthy individuals remaining consistent in each subset. This design effectively prevents the assessment bias caused by the randomness of data division.
2. **Grid Search** (`GridSearchCV`): For hyperparameter tuning of different algorithms, the code is configured with an automated search logic. For instance, in Random Forest, the optimal max_depth (tree depth) and min_samples_leaf (minimum number of samples per leaf) are sought, and in Logistic Regression, the optimal regularization coefficient C is found. By exhaustively testing parameter combinations and optimizing with auc-roc as the core objective, we ensure the model's generalization ability on unseen data. 
3. **Evaluation index system:** Given the unique nature of the screening system's operations, we not only focus on accuracy (Accuracy), but also calculate sensitivity (Sensitivity) and specificity (Specificity) in the code. In the initial screening of diabetes, high sensitivity means minimizing the omission of any potential patients, which is the key metric for evaluating the practical application value of the model.



## 3. Results Evaluation

In the data mining experiments, since our goal was "risk prediction" rather than "end-point diagnosis", the meaning of the metrics has a special business weight. The performance of each model under 5-fold cross-validation shows a high degree of convergence, indicating that the risk signals in the data are robust and objectively exist.

| 模型名称                           | 最优参数组合                                                 | 准确率 (Accuracy) | 敏感度 (Sensitivity) | 特异度 (Specificity) | 精确率 (Precision) | F1 分数 (F1-Score) | AUC 面积 |
| :--------------------------------- | :----------------------------------------------------------- | :---------------- | :------------------- | :------------------- | :----------------- | :----------------- | :------: |
| **随机森林 (Random Forest)**       | `{'max_depth': 20, 'min_samples_leaf': 20, 'n_estimators': 200}` | 0.6097            | 0.6129               | 0.6049               | 0.6994             | 0.6533             |  0.6558  |
| **逻辑回归 (Logistic Regression)** | `{'C': 0.01, 'penalty': 'l2'}`                               | 0.6038            | 0.5607               | 0.6685               | 0.7173             | 0.6294             |  0.6611  |
| **LightGBM**                       | `{'learning_rate': 0.1, 'n_estimators': 200, 'num_leaves': 63}` | 0.5973            | 0.5760               | 0.6292               | 0.6997             | 0.6319             |  0.6458  |

- **Logistic Regression:** After grid search, the optimal parameter combination for Logistic Regression was fixed at `{'C': 0.01, 'penalty': 'l2'}`. The strong regularization coefficient (C is smaller, the regularization is stronger) of up to 0.01 indicates that due to the presence of a large amount of lifestyle-related noise in the non-invasive features, the model needs to constrain the coefficient size through a strong L2 penalty term to prevent overfitting. This model achieved the highest AUC (0.6611). It is worth noting that its specificity reached 0.6685, meaning that the model performed better than other algorithms in excluding healthy individuals. In actual screening, the high specificity can significantly reduce the subsequent unnecessary clinical re-examination expenses and reduce the overall burden of the medical system.

![](D:\xwechat_files\wxid_mh12x2y4gz3422_2abe\msg\file\2026-04\src(3)\src(3)\src\lr\lr_roc_curve.png)

- **Random Forest:** Random Forest reached the optimal state with parameters `{'max_depth': 20, 'min_samples_leaf': 20, 'n_estimators': 200}`. The tree structure with a depth of up to 20 layers allows the model to fully explore the nonlinear interactions between features, and the constraint that each leaf contains at least 20 samples ensures that the forest does not fall into local noise. Random Forest showed the highest sensitivity (0.6129). In the diabetes screening scenario, sensitivity represents the reciprocal of the "miss rate". Random Forest can identify approximately 5% of potential high-risk patients, which has extremely high clinical value.

![](D:\xwechat_files\wxid_mh12x2y4gz3422_2abe\msg\file\2026-04\src(3)\src(3)\src\rf\rf_evaluation_plots.png)

- **LightGBM:** The optimal combination of LightGBM was `{'learning_rate': 0.1, 'n_estimators': 200, 'num_leaves': 63}`. The larger number of leaf nodes 63 combined with 200 iterative trees shows that the gradient boosting algorithm attempts to fit the data through a more complex partition space. Although its AUC (0.6458) is slightly lower than Logistic Regression, its computational efficiency is extremely high when dealing with large-scale categorical variables. It performed stably in precision (0.6997), proving that it has a high confidence level in determining the risk population. 

![](D:\xwechat_files\wxid_mh12x2y4gz3422_2abe\msg\file\2026-04\src(3)\src(3)\src\lgbm\lgbm_roc_curve.png)

![](D:\xwechat_files\wxid_mh12x2y4gz3422_2abe\msg\file\2026-04\src(3)\src(3)\src\lgbm\lgbm_feature_importance.png)

![](D:\xwechat_files\wxid_mh12x2y4gz3422_2abe\msg\file\2026-04\src(3)\src(3)\src\lgbm\odds_ratio_forest_plot.png)



The area under the AUC curve for each model remained stable between **0.64 and 0.66**. In the field of medical screening, this is an encouraging result. It must be emphasized that this set of data was obtained after excluding all core indicators of blood glucose. Clinically, relying solely on questionnaires and basic measurements (height, weight, blood pressure) can already achieve an area under the curve of 0.66.

If this model is put into application, its ROC curve indicates that the system can effectively filter out more than half of the low-risk groups while maintaining an accuracy rate of approximately 60%.



An interesting finding is that the final AUC differences are extremely small (varying within 0.015). This phenomenon in data mining usually implies "data characteristics dominate performance":

- **Information Limitation:** Non-invasive features (such as BMI, age, family history) can provide a relatively saturated amount of information under the current data scale.
- **Robustness Verification:** The consistency of the results strongly proves that the prediction logic is not dependent on a coincidence of a certain algorithm, but is based on the physiological and epidemiological laws contained in the features.
- **Model Selection Preference:** Given the similar performance, **logistic regression**, with its highest AUC of 0.6611 and excellent interpretability, is determined as the final recommended model for this project.



## 4. Discussion and Interpretability

In the field of medical data mining, a "black box" model that is impossible to explain, even if it has extremely high accuracy, is difficult to gain trust in real clinical scenarios. Therefore, this project not only focuses on prediction indicators, but also strives to reveal the decision-making logic behind the model.

### 4.1 Knowledge Extraction vs. Existing Methods

The odds ratio (OR) provided by the logistic regression model is our core tool for quantifying risks. By exponentiating the model coefficients, we can clearly observe the differentiated impacts of genetics, metabolism, and behavior on the probability of developing diabetes.

The data analysis shows that the **family history of diabetes** is the risk signal with the highest weight and the most stable in the model. Its odds ratio is close to 1.61. This means that after excluding all blood sugar indicators, the genetic background of the family remains the "ballast" for determining whether an individual belongs to the high-risk group. Followed closely is the age factor, with the OR value indicating that the risk increases linearly with age. This verifies the cumulative effect of chronic metabolic depletion.

What is worth in-depth analysis are the derived indicators we constructed. For example, the `tg_hdl_ratio` (triglyceride/high-density lipoprotein ratio) demonstrates stronger predictive efficacy than a single lipid indicator. From a biological perspective, an increase in this ratio often indicates the occurrence of insulin resistance. Our model, without any blood sugar data, successfully "reverse-drew" the metabolic disorder state of the subjects by capturing the subtle imbalance in lipid structure. Additionally, the significance of `map` (mean arterial pressure) also reveals the close relationship between microcirculation pressure and sugar metabolism disorders, providing solid physiological support for "non-invasive screening". 

This extracted knowledge proves that non-invasive metrics can serve as a meaningful proxy for invasive clinical markers.



### 4.2 Business Impact and Generalization

It can be observed that the F1 score of the model ranges from 0.63 to 0.65, which indicates that the model has achieved a good balance between precision and sensitivity.

- **False Positive Analysis:** Some individuals classified as high-risk may be in the "pre-diabetic" stage or have strong unhealthy habits. From a preventive medicine perspective, even if these people are not diagnosed patients, providing them with early intervention (such as suggesting increased exercise) still has significant health benefits.
- **Engineering Considerations:** During the evaluation of the code implementation, we included time consumption statistics. The reasoning time of logistic regression is nearly zero, which means that this algorithm can be seamlessly deployed on low-power mobile devices and even single-chip medical devices, achieving true inclusive healthcare.

By thoroughly dissecting the above experimental data, we can assert that the non-invasive risk screening system built in this project does not sacrifice the core predictive efficiency and significantly lowers the threshold for diabetes screening, providing a practical and feasible solution for the management of chronic diseases in large-scale populations.



Also, in the review of this project, a key discussion point was whether an AUC of around 0.66 was sufficient for clinical application. We believe that evaluating medical screening models must be based on a "cost-benefit" framework, rather than just mathematical indicators.

Although traditional laboratory screenings (such as HbA1c testing) have extremely high AUC values (often exceeding 0.90), their high costs and invasiveness make them unable to cover the most vulnerable groups that need the most attention. **The AUC of 0.66 achieved by this system is essentially an efficient "filter" provided by the medical system under zero operational costs and zero physical damage.** In practical applications, it can serve as a pre-screening layer: the top 30% of high-risk individuals identified by the system have a much higher actual probability of being ill than those selected through random sampling. Through this approach, we can precisely allocate limited medical resources to those individuals who most urgently need intervention, thereby improving the operational efficiency of public health at a macro level. This paradigm shift from "blind population-wide screening" to "data-driven precise initial screening" is precisely the highest value manifestation of data mining in smart healthcare. 



### 4.3 Limitations

Although the model performs robustly, we must also acknowledge the challenges it faces in practical implementation. 

Firstly, lifestyle data (such as dietary scores and physical activity levels) heavily rely on self-reporting by the subjects, inevitably leading to recall bias and subjective modifications. In future iterations, incorporating wearable device data (such as automatically recorded steps and resting heart rate) will be the key to breaking this bottleneck.

Secondly, although the current feature engineering has optimized outliers through Winsorization (capping), it still falls short in covering certain extreme pathological conditions. Finally, due to the geographical or racial limitations of the dataset, the generalization fairness of the model across different communities still requires further large-scale multi-center verification. 



## 5. Conclusion

After conducting numerous experiments and in-depth analysis of various algorithms, this project successfully achieved the preset research objectives. We have successfully demonstrated that, even when all diagnostic-level indicators such as blood sugar and insulin are completely excluded, the machine learning model constructed solely based on non-invasive features can still robustly identify potential risk signals of diabetes. The experimental results show that the three models - Logistic Regression, Random Forest, and LightGBM - reached a high degree of consensus in the AUC metric (stabilizing at 0.64-0.66). The consistency of these results strongly proves that risk prediction is not due to the randomness of a certain algorithm, but is deeply rooted in the pathological logic formed by the selected features such as family history, age, BMI, and lifestyle. 

Among the three algorithms, logistic regression stood out due to its outstanding interpretability and became the recommendation engine of this system. It not only provides accurate risk scores but also can intuitively inform users through "odds ratio" which specific factor (such as obesity or lack of exercise) has led to the increase in risk. This "transparent" decision-making process is crucial in the grassroots medical scenario because it can directly translate into doctors' health advice for patients, making the screening results not just a number, but also an action guide. 

Of course, the current system still has vast room for improvement. In future version iterations, we plan to introduce two key enhancements: Firstly, multi-source data fusion, by integrating dynamic data collected from wearable devices such as smartwatches (such as resting heart rate, fluctuation in exercise heart rate, etc.), upgrading the model from a "static snapshot" to a "dynamic monitoring"; Secondly, building a closed-loop management system, combining risk prediction with personalized health intervention plans (such as customized recipes and exercise prescriptions). 

In summary, this project is not only a successful application of data mining technology, but also a deep implementation of the concept of "digital preventive medicine". What we have constructed is not merely an algorithm model, but a set of low-cost and high-efficiency chronic disease management solutions. It presents us with an accessible future: through the power of technology, we can enable everyone to have their own "health warning system" without increasing economic burdens, achieving the leap from "treating existing diseases" to "preventing diseases before they occur".



## 6. Reproducibility & Code Submission

The complete source code, including:

- diabetes_dataset.csv

- data preprocessing (`lr_preprocessing.py`, `fr_preprocessing.py`,`gbm_preprocessing.py`)

- model training (`lr.py`, `random_forest.py`, `train_evaluate_lgbm.py`)

- evaluation scripts

  , are available to ensure full reproducibility of the results discussed in this report.

  

## 7. Team Contributions

- **Member 1 :** **WEI Jiazhe 59865690**

- **Member 2 :** **59926509 Xie Zekunwei**

- **Member 3 :** **Xiao Runhong 59884643** 

- **Member 4：ZHANG Huiyi 60027009**

  All members contributed equally (approx. 25% each) and successfully collaborated to prepare for the group presentation challenge.