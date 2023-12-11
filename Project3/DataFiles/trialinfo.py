import numpy as np
import os
import pandas as pd
import time

class Trialinfo:

  def __init__(self, imageID, patient):
    self.imageID = imageID
    self.patient = patient

  def get(self):
    file_path = fr"//hume.uio.no/student-u34/camilldb/pc/Desktop/masteroppgave/{self.patient}/{self.imageID}"
    #trial arrays saved in list
    trials = []
    path_0 = os.path.join(file_path,"trial_0.npy")
    if os.path.isfile(path_0):
      trial_0 = np.load(path_0)
      trials.append(trial_0)
    path_1 = os.path.join(file_path,"trial_1.npy")
    if os.path.isfile(path_1):
      trial_1 = np.load(path_1)
      trials.append(trial_1)
    path_2 = os.path.join(file_path,"trial_2.npy")
    if os.path.isfile(path_2):
      trial_2 = np.load(path_2)
      trials.append(trial_2)

    #response from patient from each trial saved in list:
    csv_path = os.path.join(file_path,"respons.csv")
    if os.path.isfile(csv_path):
      df = pd.read_csv(csv_path)
      responses = []
      if not pd.isnull(df.response[0]):
        responses.append(df.response[0])
        if len(df.response) > 1 and not pd.isnull(df.response[1]):
          responses.append(df.response[1])
          if len(df.response) > 2 and not pd.isnull(df.response[2]):
            responses.append(df.response[2])
          else:
            responses.append(None)
        else:
          responses.append(None)
          responses.append(None)
      else:
        responses = [None, None, None]
    else:
      responses = [None, None, None]
    #time between trials
    times = []
    if not pd.isnull(df.time[0]):
      t_0 = time.mktime(time.strptime(df.time[0]))
      times.append(0)
      if len(df.time) > 1 and not pd.isnull(df.time[1]):
        t_1 = time.mktime(time.strptime(df.time[1]))
        delta_1 = t_1-t_0
        min_1, sec_1 = divmod(delta_1, 60)
        times.append(min_1)
        if len(df.time) > 2 and not pd.isnull(df.time[2]):
          t_2 = time.mktime(time.strptime(df.time[2]))
          delta_2 = t_2-t_1
          min_2, sec_2 = divmod(delta_2, 60)
          times.append(min_2)
        else:
          times.append(None)
      else:
        times.append(None)
        times.append(None)
    else:
      times = [None, None, None]

    return trials, times, responses