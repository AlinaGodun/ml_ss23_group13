import numpy as np
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline


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


def define_pipelines(methods, scaler=None, oversampling=False, random_seed=1038):
    pipelines = {}
    for method in methods:
        pipeline_steps = []

        if scaler:
            pipeline_steps.append(('scaler', scaler))

        if oversampling:
            pipeline_steps.append(('oversampler', RandomOverSampler(random_state=random_seed)))

        pipeline_steps.append(method)

        if oversampling:
            pipeline = ImbPipeline(steps=pipeline_steps)
        else:
            pipeline = Pipeline(steps=pipeline_steps)

        name, _ = method
        pipelines[name] = pipeline
    return pipelines


def run_cv_experiments(pipelines, X, y, cv_num, scoring='f1_macro', n_jobs=10, print_output=False):
    model_params = {}
    model_lists = {}
    models = {}

    for model_name, pipeline in pipelines.items():
        cv_results = cross_validate(pipeline, X, y, cv=cv_num, scoring=scoring, return_estimator=True, n_jobs=n_jobs)

        models[model_name] = cv_results['estimator']
        model_params[model_name] = {}
        model_lists[model_name] = {}

        num_cols = ['test_score', 'fit_time', 'score_time']

        for num_col in num_cols:
            model_lists[model_name][num_col] = cv_results[num_col]
            model_params[model_name][f'{num_col}_mean'] = cv_results[num_col].mean()
            model_params[model_name][f'{num_col}_std'] = cv_results[num_col].std()
        
        model_params[model_name]['parameter_num'] = cv_results['estimator'][0][model_name].number_of_params_
        model_params[model_name]['hidden_layer_sizes'] = cv_results['estimator'][0][model_name].hidden_layer_sizes
        model_params[model_name]['activation_function'] = cv_results['estimator'][0][model_name].activation_function
        model_params[model_name]['learning_rate'] = cv_results['estimator'][0][model_name].learning_rate
        model_lists[model_name]['converged'] = [e[model_name].converged_ for e in cv_results['estimator']]
        model_lists[model_name]['validation_losses'] = [e[model_name].validation_losses_ for e in cv_results['estimator']]
        model_lists[model_name]['training_losses'] = [e[model_name].training_losses_ for e in cv_results['estimator']]
        model_lists[model_name]['gradients'] = [e[model_name].gradients_ for e in cv_results['estimator']]
        model_params[model_name]['num_iter'] = np.array(list([len(e[model_name].training_losses_) for e in cv_results['estimator']])).mean()

        if print_output:
            print(model_name)
            print(
                f"f1 scores: {model_lists[model_name]['test_score']}\n" +
                f"f1 mean: {model_params[model_name]['test_score_mean']:.3f}\n" +
                f"f1 std: {model_params[model_name]['test_score_std']:.3f}\n"
            )
            print('----------------------------------------------------------------------------------------------------')
    return models, model_params, model_lists
    
    