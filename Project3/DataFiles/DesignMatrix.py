import numpy as np
from trialinfo import Trialinfo

class DesignMatrix():
    def __init__(self):
        file_path = fr"\\hume.uio.no\student-u34\camilldb\pc\Desktop\masteroppgave\2022-11\imageIDs.npy"
        imageIDs = np.load(file_path)

        trials_dict = {}
        response = {}

        for imageID in imageIDs:
            if imageID == 0:
                "baseline"
            else:
                object = Trialinfo(imageID, "2022-11")
                trials, times, responses = object.get()
                trials_dict[imageID] = trials
                for i in range(0,3):
                    if responses[i] == "new":
                        responses[i] = 1
                    elif responses[i] == "old":
                        responses[i] = 0
                response[imageID] = responses

        t1_ids = []
        t2_ids = []
        t3_ids = []
        for id in imageIDs:
            if id == 0:
                "baseline"
            else:
                t = trials_dict[id]
                if len(t) > 1:
                    if len(t) > 2:
                        t3_ids.append(id)
                        t2_ids.append(id)
                        t1_ids.append(id)
                    else:
                        t2_ids.append(id)
                        t1_ids.append(id)
                elif len(t) == 1:
                    t1_ids.append(id)
        
        X_2 = [] #Design matrix where the images are shown two times
        y_2 = [] #target for this images
        X_3 = [] #Design matrix where the images are shown three times
        y_3 = [] #target for this images
        #finding the number of channels for this patient, using an arbitrary trial.
        n_channels = np.shape(trials_dict[72949][1])[1]
        #Making the Design Matrix
        for imageID in t2_ids:
            target_1 = response[imageID][0]
            target_2 = response[imageID][1] 
            if target_1 == ' ' or target_2 == ' ':
                continue
            trial_1 = trials_dict[imageID][0]
            trial_2 = trials_dict[imageID][1]
            for c in range(n_channels):
                X_2.append(trial_1[:, c])
                y_2.append(target_1)
                X_2.append(trial_2[:, c])
                y_2.append(target_2)

        for imageID in t3_ids:
            target_1 = response[imageID][0]
            target_2 = response[imageID][1]
            target_3 = response[imageID][2]
            if target_1 == ' ' or target_2 == ' ' or target_3 == ' ':
                continue
            trial_1 = trials_dict[imageID][0]
            trial_2 = trials_dict[imageID][1]
            trial_3 = trials_dict[imageID][2]
            for c in range(n_channels):
                X_3.append(trial_1[:, c])
                y_3.append(target_1)
                X_3.append(trial_2[:, c])
                y_3.append(target_2)
                X_3.append(trial_3[:, c])
                y_3.append(target_3)

        X_2 = np.asarray(X_2, dtype=int)
        X_3 = np.asarray(X_3, dtype=int)

        return X_2, X_3, y_2, y_3