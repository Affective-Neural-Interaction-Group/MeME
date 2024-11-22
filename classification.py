import mne
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from vectorizing_epochs import vectorize_epochs

# load the epoch data
epochs_dir = 'Enter the path to the folder where the epoch data is stored'
files = os.listdir(epochs_dir)

detected_markers = {}
for file in files:
    if file.endswith('.fif'):
        # get the filename
        filename = file.split('.')[0]
        epochs = mne.read_epochs(os.path.join(epochs_dir, file))

        # trial list
        t1010 = epochs[:255]
        t355 = epochs[255:510]
        t1172 = epochs[510:765]
        t1111 = epochs[765:1020]
        t1156 = epochs[1020:1275]
        t1223 = epochs[1275:1530]
        t193 = epochs[1530:1785]
        t209 = epochs[1785:2040]
        t439 = epochs[2040:2295]
        t908 = epochs[2295:2550]
        t587 = epochs[2550:2805]
        t2220 = epochs[2805:3060]
        t6251 = epochs[3060:3315]
        t269 = epochs[3315:3570]
        t1160 = epochs[3570:3825]

        epoch_dict = {
        't1010': t1010,
        't355':t355,
        't1172':t1172,
        't1111':t1111,
        't1156':t1156,
        't1223':t1223,
        't193':t193,
        't209':t209,
        't439':t439,
        't908':t908,
        't587':t587,
        't2220': t2220,
        't6251': t6251,
        't269':t269,
        't1160': t1160
        }
        trial_mark = {}

        for epoch_name, cur_epochs in epoch_dict.items(): 
            print(epoch_name)

            # Select epochs where event names start with 't'
            t_event_names = [event for event in cur_epochs.event_id if event.startswith('t')]
            tcur_epochs = cur_epochs[t_event_names]

            # prepare t labels
            t_labels = list(tcur_epochs.event_id.keys())

            # Vectorize the 't' epochs data
            Xt = vectorize_epochs(tcur_epochs)

            # Flatten the last two dimensions of Xt
            n_epochs, n_channels, n_windows = Xt.shape
            Xt_flattened = Xt.reshape((n_epochs, n_channels * n_windows))

            # Create a dictionary to store the signal data for each t label
            sig_dic = {}
            for i in range(len(t_labels)):
                sig_dic[t_labels[i]] = Xt_flattened[i, :]

            # Select epochs where event names start with 'nt'
            nt_event_names = [event for event in cur_epochs.event_id if not event.startswith('t')]
            ntcur_epochs = cur_epochs[nt_event_names]

            # Vectorize the 'nt' epochs data
            Xnt = vectorize_epochs(ntcur_epochs)      
           
            # Flatten the last two dimensions of Xnt
            n_epochs_nt, n_channels_nt, n_windows_nt = Xnt.shape

            # Shuffle and reduce Xnt while keeping track of indices
            indices = np.arange(n_epochs_nt)
            np.random.shuffle(indices)
            xt_range = Xt.shape[0] 
            reduced_indices = indices[:xt_range]
            Xnt_reduced = Xnt[reduced_indices, :, :]

            # Reshape the reduced data
            Xnt_flattened = Xnt_reduced.reshape((xt_range, n_channels_nt * n_windows_nt))
            
            nt_labels = list(ntcur_epochs.event_id.keys())
            # Ensure the number of tags matches the reduced number of epochs
            if len(nt_tags) > xt_range:
                nt_tags = nt_tags[:xt_range]
            # Use the reduced indices to map back to the original nt labels
            for i, idx in enumerate(reduced_indices):
                if idx < len(nt_labels):
                    sig_dic[nt_labels[idx]] = Xnt_flattened[i, :]

            # Combine the 't' and 'nt' tensors
            X_combined = np.concatenate((Xt, Xnt_reduced), axis=0)

            # Flatten the last two dimensions of X_combined
            n_epochs, n_channels, n_windows = X_combined.shape
            X_flattened = X_combined.reshape((n_epochs, n_channels * n_windows))

            # Create labels for the combined data
            y_labels = np.concatenate((np.ones(len(Xt)), np.zeros(len(Xnt_reduced))))

            # Now use X_flattened instead of X_combined for train_test_split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X_flattened, y_labels, test_size=0.2)

            # Create a dictionary to store classifier instances, where you can select the classifier you want to use, or add more classifiers
            clf_dic = {
                # 'svm' : sklearn.svm.SVC(),
                # 'logreg' : sklearn.linear_model.LogisticRegression(),
                # 'random_forest' : ensemble.RandomForestClassifier(n_estimators=100, random_state=42),
                # 'gradient_boosting' : ensemble.GradientBoostingClassifier(),
                # 'knn' : neighbors.KNeighborsClassifier(),
                # 'decision_tree' : tree.DecisionTreeClassifier(),
                # 'naive_bayes' : naive_bayes.GaussianNB(),
                # 'adaboost' : ensemble.AdaBoostClassifier(),
                # 'mlp' : neural_network.MLPClassifier(),
                'lda' : LDA() # Linear Discriminant Analysis is a good choice for this type of data
            }
            
            for clf_name, clf in clf_dic.items():
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                # Evaluate the classifier
                accuracy = accuracy_score(y_test, y_pred)
                print(f"{clf_name:<30s}Accuracy: {accuracy}")

                y_pred_train = clf.predict(X_train)
                
                instance_with_label = X_test[y_pred == 1]
                markers = [key for key, value in sig_dic.items() if any(i in value for i in instance_with_label)] 
            
                trial_mark[epoch_name] = markers

            # save the markers
            detected_markers[filename] = trial_mark
            print(detected_markers)

        import pandas as pd
        df = pd.DataFrame(detected_markers)
        # save as csv
        df.to_csv('detected_markers_test.csv', index=False)
