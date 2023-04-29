from sklearn import metrics, model_selection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
import numpy as np

def eval(y_pred, y_test):
    model = {}
    cm = metrics.confusion_matrix(y_test, y_pred, normalize='true')
    acc = metrics.accuracy_score(y_test, y_pred)
    balanced_acc = metrics.balanced_accuracy_score(y_test, y_pred)
    prfs = metrics.precision_recall_fscore_support(y_test, y_pred)

    display(cm)
    print(f'acc: {acc}\nbalanced_acc: {balanced_acc}\nprfs: {prfs}\n')

    model['y_pred'] = y_pred

    model['cm'] = cm
    model['acc'] = acc
    model['balanced_acc'] = balanced_acc
    model['prfs'] = prfs

    return model


def perform_gridsearch(X_train, y_train, X_test, y_test, pipe, params):
    model = {}
    gr_search = model_selection.GridSearchCV(pipe, param_grid=params, scoring='f1_macro', cv=5, verbose=0)
    gr_search.fit(X_train, y_train)
    best_params = gr_search.best_params_
    best_score = gr_search.best_score_

    best_model = gr_search.best_estimator_
    y_pred = best_model.predict(X_test)
    display(best_params)
    display(best_score)

    model = eval(y_pred, y_test)
    model['best_model'] = best_model
    model['best_params'] = best_params
    model['best_score'] = best_score

    return model


def define_pipelines(methods, scaler=None):
    pipelines = {}
    for method in methods:
        pipeline_steps = []

        if scaler:
            pipeline_steps.append(('scaler', scaler))

        pipeline_steps.append(method)
        pipeline = Pipeline(steps=pipeline_steps)
        name, cl = method
        pipelines[name] = pipeline
    return pipelines