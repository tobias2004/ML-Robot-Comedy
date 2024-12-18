import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

plt.style.use('ggplot')
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 1000)

df = pd.read_csv('comedy_data.csv')


#Feature Engineering: Created features that could potentially help understand the data.

#Create range features using min and max.
feature_headers = {'intensity','pitch','1','2','3','4','5','6','7','8','9','10','11','12'}
for i in feature_headers:
    df[i+'_range'] = df[i+'_max'] - df[i+'_min']

#Create total intensity.
df['intensity_total'] = df['length'] * df['intensity_mean']

#Metrics used to reduce bias.
df['JokeMean'] = df['JokeId'].map(df.groupby('JokeId')['HumanScore'].mean())
df['PerfMean'] = df['PerformanceId'].map(df.groupby('PerformanceId')['HumanScore'].mean())
df['WeightedHumanScore_j'] = df['HumanScore'] * df['JokeMean']
df['WeightedHumanScore_p'] = df['HumanScore'] * df['PerfMean']
weighted_scores_p = df.groupby('PerformanceId')['WeightedHumanScore_j'].sum()
weighted_scores_j = df.groupby('JokeId')['WeightedHumanScore_p'].sum()


#Functions to visualize data

#Prints 3 Box plots displaying the distribution of HumanScore across a metric
def plot_humanscore(col):
    df.boxplot(column=col, by='HumanScore', grid=False, vert=False, showmeans=True, boxprops=dict(color="blue"))
    plt.title(f"Box Plot of {col} Grouped by HumanScore")
    plt.suptitle("")
    plt.xlabel(col)
    plt.ylabel("HumanScore")
    plt.grid(True)
    plt.show()

#Compares 2 features grouped by HumanScore
def plot_vs(col1,col2):
    plt.scatter(df[df['HumanScore'] == -1][col1], 
        df[df['HumanScore'] == -1][col2], 
        color='red', 
        label='Negative', 
        alpha=0.8, 
        edgecolor='k')
    
    plt.scatter(df[df['HumanScore'] == 0][col1], 
        df[df['HumanScore'] == 0][col2], 
        color='blue', 
        label='Neutral', 
        alpha=0.8, 
        edgecolor='k')
    
    plt.scatter(df[df['HumanScore'] == 1][col1], 
        df[df['HumanScore'] == 1][col2], 
        color='green', 
        label='Positive', 
        alpha=0.8, 
        edgecolor='k')
    plt.title(f"Scatter Plot of {col1} vs {col2}")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.legend(title="HumanScore", loc="upper left")
    plt.grid(True)
    plt.show()

#Boxplots comparing feature across performances.
def compare_performance(col):
    plt.figure(figsize=(12, 6))
    df.boxplot(column=col, by='PerformanceId', grid=False,vert=False ,showmeans=True)
    plt.title(f"Box Plot of {col} by PerformanceID")
    plt.suptitle("")
    plt.ylabel("PerformanceID")
    plt.xlabel(col)
    plt.grid(True)
    plt.show()

#Box plots showing average HumanScore per JokeId
def avg_human_score_per_joke():
    # Calculate mean HumanScore for each JokeId
    joke_means = df.groupby('JokeId')['HumanScore'].mean()

    # Plot the bar graph
    joke_means.plot(kind='bar', figsize=(12, 6), color='skyblue', edgecolor='black')
    plt.title("Mean HumanScore per Joke")
    plt.xlabel("JokeId")
    plt.ylabel("Mean HumanScore")
    plt.grid(True)
    plt.show()

#Box plots showing average HumanScore per performance.
def avg_human_score_per_performance():
    # Calculate mean HumanScore for each JokeId
    performance_means = df.groupby('PerformanceId')['HumanScore'].mean()

    performance_means.plot(kind='bar', figsize=(12, 6), color='skyblue', edgecolor='black')
    plt.title("Mean HumanScore per Performance")
    plt.xlabel("PerformanceId")
    plt.ylabel("Mean HumanScore")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

#Boxplots showing average weighted HumanScore per performance
def avg_weighted_human_score_per_performance():
    weighted_scores_p.plot(kind='bar', figsize=(12, 6), color='skyblue', edgecolor='black')
    plt.title("Mean Weighted HumanScore per Performance")
    plt.xlabel("PerformanceId")
    plt.ylabel("Mean Weighted HumanScore")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

#Boxplots showing average weighted HumanScore per joke.
def avg_weighted_human_score_per_joke():
    plt.figure(figsize=(15, 6))
    weighted_scores_j.sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Mean Weighted HumanScore per Joke")
    plt.xlabel("JokeId")
    plt.ylabel("Mean Weighted HumanScore")
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.show()

#Boxplots showing how different performances responded to each joke.
def performance_means_per_joke():
    joke_PerfMean = df.groupby('JokeId')['PerfMean'].mean()
    plt.figure(figsize=(12, 6))
    joke_PerfMean.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Average Performance HumanScore per JokeId")
    plt.xlabel("JokeId")
    plt.ylabel("Average Performance HumanScore")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

#Performs random forrest and permutation importance to find the most significant features.
def rand_forrest_permutation_importance(features):
    x = df[features]
    y = df['HumanScore']
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(x, y)

    perm_importance = permutation_importance(rf, x, y, n_repeats=10, random_state=42, scoring='accuracy')

    return rf.feature_importances_,perm_importance

#Fits the data to a desicion tree model, returning the accuracy metrics.
def decision_tree(features,Verbose=False):
    y = df['HumanScore']
    x = df[features]
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    dt = DecisionTreeClassifier(random_state=42)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    cumulative_conf_matrix = np.zeros((3, 3))
    # Perform Stratified K-Fold Cross-Validation
    for train_index, test_index in skfold.split(x, y):
        #Split the data
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        #Train the model
        dt.fit(x_train, y_train)
        y_pred = dt.predict(x_test)
        #Measure accuracy
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        fold_conf_matrix = confusion_matrix(y_test, y_pred, labels=[-1,0,1])
        cumulative_conf_matrix += fold_conf_matrix
    if Verbose:
        print(f"  Confusion Matrix:\n{cumulative_conf_matrix}\n")
    return np.mean(accuracy_scores),np.mean(precision_scores),np.mean(recall_scores),np.mean(f1_scores),cumulative_conf_matrix

#Fits the data to a KNN model, returning the accuracy metrics.
def k_nearest_neighbor(features,folds,Verbose=False):
    x = df[features]
    y = df['HumanScore']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=folds)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    cumulative_conf_matrix = np.zeros((3, 3))

    # Perform Stratified K-Fold Cross-Validation
    for train_index, test_index in skfold.split(X_scaled, y):
        # Split the data
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Train the model
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        fold_conf_matrix = confusion_matrix(y_test, y_pred, labels=[-1,0,1])
        cumulative_conf_matrix += fold_conf_matrix
    if Verbose:
        print(f"  Confusion Matrix:\n{cumulative_conf_matrix}\n")
    return np.mean(accuracy_scores),np.mean(precision_scores),np.mean(recall_scores),np.mean(f1_scores),cumulative_conf_matrix

#Fits the data to a SVM model, returning the accuracy metrics.
def support_vector_machine(features,kernel,Verbose=False):
    y = df['HumanScore']
    x = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)
    
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm = SVC(kernel=kernel, random_state=42)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    cumulative_conf_matrix = np.zeros((3, 3))

    # Perform Stratified K-Fold Cross-Validation
    for train_index, test_index in skfold.split(X_scaled, y):
        # Split the data
        x_train, x_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the model
        svm.fit(x_train, y_train)
        y_pred = svm.predict(x_test)
        
        # Measure performance
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        fold_conf_matrix = confusion_matrix(y_test, y_pred, labels=[-1,0,1])
        cumulative_conf_matrix += fold_conf_matrix
    if Verbose:
        print(f"  Confusion Matrix:\n{cumulative_conf_matrix}\n")
    return np.mean(accuracy_scores),np.mean(precision_scores),np.mean(recall_scores),np.mean(f1_scores),cumulative_conf_matrix


#Composite Analytic Functions
def analyze_performances():
    #Identify potential outperformers with features that directly influence HumanScore
    feature_list = {'intensity_mean','pitch_range','length','2_mean'}
    for i in feature_list:
        compare_performance(i)

    avg_human_score_per_performance()
    
    performance_means = df.groupby('PerformanceId')['HumanScore'].mean()
    highest_perf = performance_means.sort_values(ascending=False).head(5).index.tolist()
    lowest_perf = performance_means.sort_values(ascending=True).head(5).index.tolist()
    print('5 Best Performances')
    print(highest_perf)
    print('5 Worst Performances')
    print(lowest_perf)

    print("As we can see, not all jokes are equal. We can normalize HumanScore by relating it the joke's average HumanScore")
    avg_human_score_per_joke()
    avg_weighted_human_score_per_performance()

    weighted_highest_perf = weighted_scores_p.sort_values(ascending=False).head(5).index.tolist()
    weighted_lowest_perf = weighted_scores_p.sort_values(ascending=True).head(5).index.tolist()
    print('5 Best Performances after weighing')
    print(weighted_highest_perf)
    print('5 Worst Performances after weighing')
    print(weighted_lowest_perf)


def analyze_jokes():
    avg_weighted_human_score_per_joke()

    joke_means = df.groupby('JokeId')['HumanScore'].mean()
    highest_jokes = joke_means[joke_means > 0.75]
    lowest_jokes = joke_means[joke_means < -0.75]
    print('5 Best Jokes')
    print(highest_jokes)
    print('5 Worst Jokes')
    print(lowest_jokes)

    print("However, these averages don't account for biases (Times the joke was told and overall receptivity of the audience)")
    performance_means_per_joke()
    avg_weighted_human_score_per_joke()

    weighted_highest_jokes = weighted_scores_j.sort_values(ascending=False).head(5).index.tolist()
    weighted_lowest_jokes = weighted_scores_j.sort_values(ascending=True).head(5).index.tolist()
    print('5 Best Jokes after weighing')
    print(weighted_highest_jokes)
    print('5 Worst Jokes after weighing')
    print(weighted_lowest_jokes)


def analyze_features(verbose=False):
    #Show features with correlation to HumanScore
    if verbose:
        plot_list = {'length','intensity_mean','intensity_max','pitch_range','2_mean','7_mean','intensity_total'}
        for col in plot_list:
            plot_humanscore(col)
    
    #Use Random Forrests and Permutation Importance to find most significant features
    features = df.columns.tolist()
    features.remove('PerformanceId')
    features.remove('JokeId')
    features.remove('HumanScore')
    features.remove('WeightedHumanScore_p')
    features.remove('WeightedHumanScore_j')
    features.remove('JokeMean')
    rf_importance,perm_importance = rand_forrest_permutation_importance(features)

    rf_importances = pd.DataFrame(
        {'Feature': features,'Importance': rf_importance}
        ).sort_values(by='Importance', ascending=False)
    perm_importances = pd.DataFrame({
        'Feature': features,
        'Importance': perm_importance.importances_mean
    }).sort_values(by='Importance', ascending=False)
    if verbose:
        print('Feature Importances by RF\n',rf_importances)
        print('Feature Importance by Permutation Importance\n',perm_importances)

    highest_features_rf = rf_importances.sort_values(by='Importance',ascending=False).head(5).Feature.tolist()
    highest_features_perm = perm_importances.sort_values(by='Importance',ascending=False).head(5).Feature.tolist()
    print('5 Most Significant Features by RF')
    print(highest_features_rf)
    print('5 Most Significant Features by Permutation Importance')
    print(highest_features_perm)

    rf_features = rf_importances.sort_values(by='Importance',ascending=False).Feature.tolist()
    perm_features = perm_importances.sort_values(by='Importance',ascending=False).Feature.tolist()
    return rf_features,perm_features


def analyze_desicion_tree(rf_features,perm_features,verbose=False):
    num_cols = (df.columns.tolist().__len__()) / 2
    non_mfcc_features = []
    for feature in rf_features:
        if not feature[0].isdigit():
            non_mfcc_features.append(feature)
    non_mfcc_cols = (non_mfcc_features.__len__()) / 2
    features_list = [rf_features[:10],rf_features[:int(num_cols)],rf_features,perm_features[:6],non_mfcc_features[:int(non_mfcc_cols)],non_mfcc_features]
    results = []

    for features in features_list:

        accuracy_scores,precision_scores,recall_scores,f1_scores,conf_matrices = decision_tree(features,Verbose=verbose)
        result = {
            "features": features,
            "mean_accuracy": accuracy_scores,
            "mean_precision": precision_scores,
            "mean_recall": recall_scores,
            "mean_f1": f1_scores,
            "confusion_matrices": conf_matrices
        }
        results.append(result)

    if verbose:
        for result in results:
            print(f"Features: {result['features']}")
            print(f"  Mean Accuracy: {result['mean_accuracy']:.4f}")
            print(f"  Mean Precision: {result['mean_precision']:.4f}")
            print(f"  Mean Recall: {result['mean_recall']:.4f}")
            print(f"  Mean F1-Score: {result['mean_f1']:.4f}\n")
    return results[3]

        
def analyze_knn(rf_features,perm_features,verbose=False):
    num_cols = (df.columns.tolist().__len__()) / 2
    non_mfcc_features = []
    for feature in rf_features:
        if not feature[0].isdigit():
            non_mfcc_features.append(feature)
    non_mfcc_cols = (non_mfcc_features.__len__()) / 2
    features_list = [rf_features[:10],rf_features[:int(num_cols)],rf_features,perm_features[:6],non_mfcc_features[:int(non_mfcc_cols)],non_mfcc_features]
    results = []

    for features in features_list:

        accuracy_scores,precision_scores,recall_scores,f1_scores,conf_matrices = k_nearest_neighbor(features,3,Verbose=verbose)
        result = {
            "features": features,
            "mean_accuracy": accuracy_scores,
            "mean_precision": precision_scores,
            "mean_recall": recall_scores,
            "mean_f1": f1_scores,
            "confusion_matrices": conf_matrices
        }
        results.append(result)

    if verbose:
        for result in results:
            print(f"Features: {result['features']}")
            print(f"  Mean Accuracy: {result['mean_accuracy']:.4f}")
            print(f"  Mean Precision: {result['mean_precision']:.4f}")
            print(f"  Mean Recall: {result['mean_recall']:.4f}")
            print(f"  Mean F1-Score: {result['mean_f1']:.4f}\n")
    return results[0]


def analyze_svm(rf_features,perm_features,verbose=False):
    num_cols = (df.columns.tolist().__len__()) / 2
    non_mfcc_features = []
    for feature in rf_features:
        if not feature[0].isdigit():
            non_mfcc_features.append(feature)
    non_mfcc_cols = (non_mfcc_features.__len__()) / 2
    features_list = [rf_features[:10],rf_features[:int(num_cols)],rf_features,perm_features[:6],non_mfcc_features[:int(non_mfcc_cols)],non_mfcc_features]
    results = []

    for features in features_list:

        accuracy_scores,precision_scores,recall_scores,f1_scores,conf_matrices = support_vector_machine(features,'linear',Verbose=verbose)
        result = {
            "features": features,
            "mean_accuracy": accuracy_scores,
            "mean_precision": precision_scores,
            "mean_recall": recall_scores,
            "mean_f1": f1_scores,
            "confusion_matrices": conf_matrices
        }
        results.append(result)

    if verbose:
        for result in results:
            print(f"Features: {result['features']}")
            print(f"  Mean Accuracy: {result['mean_accuracy']:.4f}")
            print(f"  Mean Precision: {result['mean_precision']:.4f}")
            print(f"  Mean Recall: {result['mean_recall']:.4f}")
            print(f"  Mean F1-Score: {result['mean_f1']:.4f}\n")
    return results[0]

#Compare results between all 3 models.
def plot_results(dt,knn,svm):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    bar_width = 0.25

    # Plot bars
    plt.bar([0.2,1.2,2.2,3.2], dt, width=bar_width, label='Decision Tree', color='r', edgecolor='grey')
    plt.bar([0.4,1.4,2.4,3.4], knn, width=bar_width, label='k-NN', color='g', edgecolor='grey')
    plt.bar([0.6,1.6,2.6,3.6], svm, width=bar_width, label='SVM', color='b', edgecolor='grey')
    plt.ylabel('Mean Score')
    plt.title('Comparison of  Metrics')
    plt.legend(loc="lower right")
    plt.xticks([0.4,1.4,2.4,3.4],metrics)
    plt.grid(True)
    plt.show()

#Compare results between all 3 models.
def plot_results_totals(dt,knn,svm):
    models = ['Decision Tree', 'k-NN', 'SVM']
    x = [0.2,1.2,2.2]
    bar_width = 0.4
    
    y1 = [dt[0],knn[0],svm[0]]
    y2 = [dt[0]+dt[1],knn[0]+knn[1],svm[0]+svm[1]]
    y3 = [dt[0]+dt[1]+dt[2],knn[0]+knn[1]+knn[2],svm[0]+svm[1]+svm[2]]

    # Plot bar
    plt.bar(x, [dt[0],knn[0],svm[0]], bar_width, label='Accuracy', color='r', edgecolor='grey')
    plt.bar(x, [dt[1],knn[1],svm[1]], bar_width, bottom=y1, label='Precision', color='g', edgecolor='grey')
    plt.bar(x, [dt[2],knn[2],svm[2]], bar_width, bottom=y2, label='Recall', color='b', edgecolor='grey')
    plt.bar(x, [dt[3],knn[3],svm[3]], bar_width, bottom=y3, label='F1-Score', color='y', edgecolor='grey')
    plt.xlabel('Models')
    plt.ylabel('Total Metric Scores')
    plt.title('Comparison of Total Scores Across Models (Stacked)')
    plt.xticks(x, models)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

#Display confusion matrices
def confusion_matrices(dt_matrix,knn_matrix,svm_matrix):
    
    dt = ConfusionMatrixDisplay(dt_matrix,display_labels=[-1,0,1])
    print('   Confusion Matrix for Desicion Tree')
    dt.plot()
    knn = ConfusionMatrixDisplay(knn_matrix,display_labels=[-1,0,1])
    print('   Confusion Matrix for k_Nearest Neighbor')
    knn.plot()
    svm = ConfusionMatrixDisplay(svm_matrix,display_labels=[-1,0,1])
    print('   Confusion Matrix for Support Vector Machine')
    svm.plot()



#Execute functions
analyze_performances()
analyze_jokes()
rf_features,perm_features = analyze_features(verbose=True)
dt_results = analyze_desicion_tree(rf_features,perm_features,verbose=True)
knn_results = analyze_knn(rf_features,perm_features,verbose=True)
svm_results = analyze_svm(rf_features,perm_features,verbose=True)

dt_list = list(dt_results.values())[1:-1]
knn_list = list(knn_results.values())[1:-1]
svm_list = list(svm_results.values())[1:-1]
plot_results(dt_list,knn_list,svm_list)
plot_results_totals(dt_list,knn_list,svm_list)

dt_matrix = list(dt_results.values())[-1]
knn_matrix = list(knn_results.values())[-1]
svm_matrix = list(svm_results.values())[-1]
confusion_matrices(dt_matrix,knn_matrix,svm_matrix)

