# DrinkingWater


![Map-Future-of-Water_1778px_with_scale](https://user-images.githubusercontent.com/100385953/179432676-2b861b1c-cbcb-4356-8c68-f9fd80591ddf.jpg)


Context :

Access to safe drinking water is essential to health, a basic human right, and a component of effective policy for health protection. This is important as a health and development issue at a national, regional, and local level. In some regions, it has been shown that investments in water supply and sanitation can yield a net economic benefit, since the reductions in adverse health effects and health care costs outweigh the costs of undertaking the interventions.

<img width="961" alt="Screenshot 2022-07-18 at 03 07 41" src="https://user-images.githubusercontent.com/100385953/179432828-c08c703a-8af7-4484-921d-993e9aab3440.png">



The drinkingwaterpotability.csv file contains water quality metrics for 3276 different water bodies

We will use different ML models and H2O Auto ML library in this project


<img width="213" alt="Screenshot 2022-07-18 at 03 10 20" src="https://user-images.githubusercontent.com/100385953/179432809-a817e943-129f-448d-b968-e6c20de50ec9.png">


AutoML: Automatic Machine Learning

In recent years, the demand for machine learning experts has outpaced the supply, despite the surge of people entering the field. To address this gap, there have been big strides in the development of user-friendly machine learning software that can be used by non-experts. The first steps toward simplifying machine learning involved developing simple, unified interfaces to a variety of machine learning algorithms (e.g. H2O).

Although H2O has made it easy for non-experts to experiment with machine learning, there is still a fair bit of knowledge and background in data science that is required to produce high-performing machine learning models. Deep Neural Networks in particular are notoriously difficult for a non-expert to tune properly. In order for machine learning software to truly be accessible to non-experts, we have designed an easy-to-use interface which automates the process of training a large selection of candidate models. H2O’s AutoML can also be a helpful tool for the advanced user, by providing a simple wrapper function that performs a large number of modeling-related tasks that would typically require many lines of code, and by freeing up their time to focus on other aspects of the data science pipeline tasks such as data-preprocessing, feature engineering and model deployment.

H2O’s AutoML can be used for automating the machine learning workflow, which includes automatic training and tuning of many models within a user-specified time-limit.

H2O offers a number of model explainability methods that apply to AutoML objects (groups of models), as well as individual models (e.g. leader model). Explanations can be generated automatically with a single function call, providing a simple interface to exploring and explaining the AutoML models.

AutoML Interface

The H2O AutoML interface is designed to have as few parameters as possible so that all the user needs to do is point to their dataset, identify the response column and optionally specify a time constraint or limit on the number of total models trained.

In both the R and Python API, AutoML uses the same data-related arguments, x, y, training_frame, validation_frame, as the other H2O algorithms. Most of the time, all you’ll need to do is specify the data arguments. You can then configure values for max_runtime_secs and/or max_models to set explicit time or number-of-model limits on your run.

Required Parameters

Required Data Parameters

y: This argument is the name (or index) of the response column.
training_frame: Specifies the training set.
Required Stopping Parameters

One of the following stopping strategies (time or number-of-model based) must be specified. When both options are set, then the AutoML run will stop as soon as it hits one of either When both options are set, then the AutoML run will stop as soon as it hits either of these limits.

max_runtime_secs: This argument specifies the maximum time that the AutoML process will run for. The default is 0 (no limit), but dynamically sets to 1 hour if none of max_runtime_secs and max_models are specified by the user.
max_models: Specify the maximum number of models to build in an AutoML run, excluding the Stacked Ensemble models. Defaults to NULL/None. Always set this parameter to ensure AutoML reproducibility: all models are then trained until convergence and none is constrained by a time budget.
Optional Parameters

Optional Data Parameters

x: A list/vector of predictor column names or indexes. This argument only needs to be specified if the user wants to exclude columns from the set of predictors. If all columns (other than the response) should be used in prediction, then this does not need to be set.
validation_frame: This argument is ignored unless nfolds == 0, in which a validation frame can be specified and used for early stopping of individual models and early stopping of the grid searches (unless max_models or max_runtime_secs overrides metric-based early stopping). By default and when nfolds > 1, cross-validation metrics will be used for early stopping and thus validation_frame will be ignored.
leaderboard_frame: This argument allows the user to specify a particular data frame to use to score and rank models on the leaderboard. This frame will not be used for anything besides leaderboard scoring. If a leaderboard frame is not specified by the user, then the leaderboard will use cross-validation metrics instead, or if cross-validation is turned off by setting nfolds = 0, then a leaderboard frame will be generated automatically from the training frame.
blending_frame: Specifies a frame to be used for computing the predictions that serve as the training frame for the Stacked Ensemble models metalearner. If provided, all Stacked Ensembles produced by AutoML will be trained using Blending (a.k.a. Holdout Stacking) instead of the default Stacking method based on cross-validation.
fold_column: Specifies a column with cross-validation fold index assignment per observation. This is used to override the default, randomized, 5-fold cross-validation scheme for individual models in the AutoML run.
weights_column: Specifies a column with observation weights. Giving some observation a weight of zero is equivalent to excluding it from the dataset; giving an observation a relative weight of 2 is equivalent to repeating that row twice. Negative weights are not allowed.
Optional Miscellaneous Parameters

nfolds: Specify a value >= 2 for the number of folds for k-fold cross-validation of the models in the AutoML run or specify “-1” to let AutoML choose if k-fold cross-validation or blending mode should be used. Blending mode will use part of training_frame (if no blending_frame is provided) to train Stacked Ensembles. Use 0 to disable cross-validation; this will also disable Stacked Ensembles (thus decreasing the overall best model performance). This value defaults to “-1”.
balance_classes: Specify whether to oversample the minority classes to balance the class distribution. This option is not enabled by default and can increase the data frame size. This option is only applicable for classification. If the oversampled size of the dataset exceeds the maximum size calculated using the max_after_balance_size parameter, then the majority classes will be undersampled to satisfy the size limit.
class_sampling_factors: Specify the per-class (in lexicographical order) over/under-sampling ratios. By default, these ratios are automatically computed during training to obtain the class balance. Note that this requires balance_classes set to True.
max_after_balance_size: Specify the maximum relative size of the training data after balancing class counts (balance_classes must be enabled). Defaults to 5.0. (The value can be less than 1.0).
max_runtime_secs_per_model: Specify the max amount of time dedicated to the training of each individual model in the AutoML run. Defaults to 0 (disabled). Note that models constrained by a time budget are not guaranteed reproducible.
stopping_metric: Specify the metric to use for early stopping. Defaults to AUTO. The available options are:

AUTO: This defaults to logloss for classification and deviance for regression.
deviance (mean residual deviance)
logloss
MSE
RMSE
MAE
RMSLE
AUC (area under the ROC curve)
AUCPR (area under the Precision-Recall curve)
lift_top_group
misclassification
mean_per_class_error
stopping_tolerance: This option specifies the relative tolerance for the metric-based stopping criterion to stop a grid search and the training of individual models within the AutoML run. This value defaults to 0.001 if the dataset is at least 1 million rows; otherwise it defaults to a bigger value determined by the size of the dataset and the non-NA-rate. In that case, the value is computed as 1/sqrt(nrows * non-NA-rate).
stopping_rounds: This argument is used to stop model training when the stopping metric (e.g. AUC) doesn’t improve for this specified number of training rounds, based on a simple moving average. In the context of AutoML, this controls early stopping both within the random grid searches as well as the individual models. Defaults to 3 and must be an non-negative integer. To disable early stopping altogether, set this to 0.
sort_metric: Specifies the metric used to sort the Leaderboard by at the end of an AutoML run. Available options include:

AUTO: This defaults to AUC for binary classification, mean_per_class_error for multinomial classification, and deviance for regression.
deviance (mean residual deviance)
logloss
MSE
RMSE
MAE
RMSLE
AUC (area under the ROC curve)
AUCPR (area under the Precision-Recall curve)
mean_per_class_error
seed: Integer. Set a seed for reproducibility. AutoML can only guarantee reproducibility under certain conditions. H2O Deep Learning models are not reproducible by default for performance reasons, so if the user requires reproducibility, then exclude_algos must contain "DeepLearning". In addition max_models must be used because max_runtime_secs is resource limited, meaning that if the available compute resources are not the same between runs, AutoML may be able to train more models on one run vs another. Defaults to NULL/None.
project_name: Character string to identify an AutoML project. Defaults to NULL/None, which means a project name will be auto-generated based on the training frame ID. More models can be trained and added to an existing AutoML project by specifying the same project name in multiple calls to the AutoML function (as long as the same training frame is used in subsequent runs).
exclude_algos: A list/vector of character strings naming the algorithms to skip during the model-building phase. An example use is exclude_algos = ["GLM", "DeepLearning", "DRF"] in Python or exclude_algos = c("GLM", "DeepLearning", "DRF") in R. Defaults to None/NULL, which means that all appropriate H2O algorithms will be used if the search stopping criteria allows and if the include_algos option is not specified. This option is mutually exclusive with include_algos. See include_algos below for the list of available options.
include_algos: A list/vector of character strings naming the algorithms to include during the model-building phase. An example use is include_algos = ["GLM", "DeepLearning", "DRF"] in Python or include_algos = c("GLM", "DeepLearning", "DRF") in R. Defaults to None/NULL, which means that all appropriate H2O algorithms will be used if the search stopping criteria allows and if no algorithms are specified in exclude_algos. This option is mutually exclusive with exclude_algos. The available algorithms are:

DRF (This includes both the Distributed Random Forest (DRF) and Extremely Randomized Trees (XRT) models. Refer to the Extremely Randomized Trees section in the DRF chapter and the histogram_type parameter description for more information.)
GLM (Generalized Linear Model with regularization)
XGBoost (XGBoost GBM)
GBM (H2O GBM)
DeepLearning (Fully-connected multi-layer artificial neural network)
StackedEnsemble (Stacked Ensembles, includes an ensemble of all the base models and ensembles using subsets of the base models)
modeling_plan: The list of modeling steps to be used by the AutoML engine. (They may not all get executed, depending on other constraints.)
preprocessing: The list of preprocessing steps to run. Only ["target_encoding"] is currently supported. There is more information about how Target Encoding is automatically applied here. Experimental.
exploitation_ratio: Specify the budget ratio (between 0 and 1) dedicated to the exploitation (vs exploration) phase. By default, the exploitation phase is disabled (exploitation_ratio=0) as this is still experimental; to activate it, it is recommended to try a ratio around 0.1. Note that the current exploitation phase only tries to fine-tune the best XGBoost and the best GBM found during exploration. Experimental.
monotone_constraints: A mapping that represents monotonic constraints. Use +1 to enforce an increasing constraint and -1 to specify a decreasing constraint.
keep_cross_validation_predictions: Specify whether to keep the predictions of the cross-validation predictions. This needs to be set to TRUE if running the same AutoML object for repeated runs because CV predictions are required to build additional Stacked Ensemble models in AutoML. This option defaults to FALSE.
keep_cross_validation_models: Specify whether to keep the cross-validated models. Keeping cross-validation models may consume significantly more memory in the H2O cluster. This option defaults to FALSE.
keep_cross_validation_fold_assignment: Enable this option to preserve the cross-validation fold assignment. Defaults to FALSE.
verbosity: (Optional: Python and R only) The verbosity of the backend messages printed during training. Must be one of "debug", "info", "warn". Defaults to NULL/None (client logging disabled).
export_checkpoints_dir: Specify a directory to which generated models will automatically be exported.
