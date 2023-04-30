from sklearn import metrics, model_selection
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning

import warnings

def eval(y_pred, y_test):
    model = {}
    cm = metrics.confusion_matrix(y_test, y_pred, normalize='true')
    acc = metrics.accuracy_score(y_test, y_pred)
    balanced_acc = metrics.balanced_accuracy_score(y_test, y_pred)
    p, r, f, s = metrics.precision_recall_fscore_support(y_test, y_pred, average='macro')

    print('Evaluation metrics:')
    display(cm)
    print(f'acc: {acc}\nbalanced_acc: {balanced_acc}\n')
    print(f'Macro-averaged precision: {p}')
    print(f'Macro-averaged recall: {r}')
    print(f'Macro-averaged f-score: {f}')
    print(f'Macro-averaged support: {s}')


    model['y_pred'] = y_pred

    model['cm'] = cm
    model['acc'] = acc
    model['balanced_acc'] = balanced_acc
    model['precision'] = p
    model['recall'] = r
    model['f-score'] = f
    model['support'] = s

    return model


def perform_gridsearch(X_train, y_train, X_test, y_test, pipe, params):
    model = {}
    gr_search = model_selection.GridSearchCV(pipe, param_grid=params, scoring='f1_macro', cv=5, verbose=0)
    gr_search.fit(X_train, y_train)
    best_params = gr_search.best_params_
    best_score = gr_search.best_score_
    refit_time = gr_search.refit_time_

    best_model = gr_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print('Best params:', best_params)
    print('Best score:', best_score)
    print('Refit time:', refit_time)

    model = eval(y_pred, y_test)
    model['best_model'] = best_model
    model['best_params'] = best_params
    model['best_score'] = best_score
    model['refit_time'] = refit_time

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


def compare_models(pipelines, params, X_train, y_train, X_test, y_test):
    models_cv = {}
    for model_name, pipeline in pipelines.items():
        print(model_name)

        # Disable MLP's convergence and runtime warnings to get a cleaner output for the hand-in;
        # During our experiments, we also considered whether MLP training convered in the given 1000 iterations
        # The analysis of this behaviour is mentioned in the report
        # warnings.simplefilter("ignore", category=ConvergenceWarning)
        # warnings.simplefilter("ignore", category=RuntimeWarning)

        models_cv[model_name] = perform_gridsearch(X_train, y_train, X_test, y_test, pipeline, params[model_name])
        print(f"Report line: {models_cv[model_name]['acc']:.3f} " +
                             f"{models_cv[model_name]['balanced_acc']:.3f} " +
                             f"{models_cv[model_name]['precision']:.3f} " +
                             f"{models_cv[model_name]['recall']:.3f} " +
                             f"{models_cv[model_name]['f-score']:.3f} " +
                             f"{models_cv[model_name]['refit_time']:.3f} ")
        print('----------------------------------------------------------------------------------------------------')
    return models_cv
    