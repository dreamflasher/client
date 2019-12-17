import wandb
import sklearn
import scikitplot
import matplotlib.pyplot as plt


def watch(estimator, X_test=None, y_test=None, labels=None):
    if sklearn.base.is_classifier(estimator):
        if X_test is not None and y_test is not None:
            y_pred = estimator.predict(X_test)
            y_probas = estimator.predict_proba(X_test)

            scikitplot.metrics.plot_roc(y_test, y_probas)
            wandb.log({"roc": plt})
            plt.clf()

            scikitplot.metrics.plot_confusion_matrix(y_test, y_pred)
            wandb.log({"confusion_matrix": plt})

        if labels is not None:
            plt.clf()
            scikitplot.estimators.plot_feature_importances(estimator, feature_names=labels)
            wandb.log({"feature_importances": plt})
