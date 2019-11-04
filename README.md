# Description
In this project I played with a case of [imbalanced data](https://www.kaggle.com/mlg-ulb/creditcardfraud). The `0.0 - literature_research.md` contains my notes on the topic (methods of handling it, best models, etc.,).

## Notes:
* I've had to remove a couple of files from the `data` folder due to their size. Here are the links to find them:
    * `random_search_results.joblib` - dictionary used in `0.3-models_exploration.ipynb`. Contained the results from RandomSearchCV on the models which used sampling to rebalance the dataset: [Link to GDrive file](https://drive.google.com/open?id=1pQYIaV0OEmp4yuFOAmv_x_9ZMBJmIIJ-)
    * `creditcard.csv` - original csv of the data. Can be found here: [Kaggle dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
    * `processed_dataset.csv` - dictionary containing the training and test dataset properly split and processed for model evaluation. [Link to GDrive file](https://drive.google.com/file/d/1-oVIbqWWOuxb9-PPKOSjTnrrPSNAnn5H/view?usp=sharing)
    
* I've made a mistake calling average precision score the same as AUC of the PR curve. It summarizes the PR curve, but it's a different implementation. In case of AUC of PR it might be too optimistic, the correct way (which I used) is the AP score. In this case, the AUC and AP were equal, but, as mentioned previously, AUC might yield a more optimistic score. [Source](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
   * The notebooks haven't been touched tho'.
