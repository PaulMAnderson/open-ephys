# import sys

# sys.path.append('.')  # Add current directory to path
# from open_ephys.analysis.session import Session
# from open_ephys.analysis.recording import Recording
# from open_ephys.analysis.formats.BinaryRecording import BinaryRecording


# rec_path = '/mnt/g/To Process/PMA97/PMA97 2025-03-13 Session 1/PMA97 2025-03-13_10-50-54 Confidence Config 2'

# session = Session(rec_path)

# node = session.recordnodes[0]
# recs = node.recordings
# recording = recs[0]

# recording.load_barcode_data()

# continuous = recording.continuous

# events = recording.events

# recording.synchronize_timestamps()

# print(recording.continuous)


import sys
import os

from matplotlib import pyplot as plt

os.chdir('/home/paul/Documents/open-ephys/src/')

sys.path.append('.')  # Add current directory to path
from open_ephys.analysis.session import Session
from open_ephys.analysis.recording import Recording
from open_ephys.analysis.formats.BinaryRecording import BinaryRecording


rec_path = '/mnt/g/To Process/PMA97/PMA97 2025-03-13 Session 1/PMA97 2025-03-13_12-21-42 Opto Config 2'

session = Session(rec_path)


node = session.recordnodes[0]
recs = node.recordings
recording = recs[0]
continuous = recording.continuous
events = recording.events


daq_events = events[events['stream_name'] == 'PXIe-6341']
laser_events = daq_events[daq_events['line'].isin([8])]

chan = 1
laser_start_global_time = laser_events.iloc[0].global_timestamp

a = continuous[0].get_data(start_time=100)
