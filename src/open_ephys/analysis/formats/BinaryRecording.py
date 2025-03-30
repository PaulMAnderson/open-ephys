"""
MIT License

Copyright (c) 2020 Open Ephys

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import glob
import os
import numpy as np
import pandas as pd
import json

from open_ephys.analysis.recording import Recording
from open_ephys.analysis.utils import alphanum_key

class BinaryRecording(Recording):
    
    class Spikes:
        
        def __init__(self, info, base_directory, version):
        
            self.metadata = { }

            self.metadata['name'] = info['name']
            self.metadata['stream_name'] = info['stream_name']
            self.metadata['sample_rate'] = info['sample_rate']
            self.metadata['num_channels'] = info['num_channels']
            
            if version >= 0.6:
                directory = os.path.join(base_directory, 'spikes', info['folder'])
                self.sample_numbers = np.load(os.path.join(directory, 'sample_numbers.npy'), mmap_mode='r')
                self.timestamps = np.load(os.path.join(directory, 'timestamps.npy'), mmap_mode='r')
                self.electrodes = np.load(os.path.join(directory, 'electrode_indices.npy'), mmap_mode='r') - 1
                self.waveforms = np.load(os.path.join(directory, 'waveforms.npy')).astype('float64')
                self.clusters = np.load(os.path.join(directory, 'clusters.npy'), mmap_mode='r')

            else:
                directory = os.path.join(base_directory, 'spikes', info['folder_name'])
                self.sample_numbers = np.load(os.path.join(directory, 'spike_times.npy'), mmap_mode='r')
                self.electrodes = np.load(os.path.join(directory, 'spike_electrode_indices.npy'), mmap_mode='r') - 1
                self.waveforms = np.load(os.path.join(directory, 'spike_waveforms.npy')).astype('float64')
                self.clusters = np.load(os.path.join(directory, 'spike_clusters.npy'), mmap_mode='r')

            if self.waveforms.ndim == 2:
                self.waveforms = np.expand_dims(self.waveforms, 1)

            self.waveforms *= float(info['source_channels'][0]['bit_volts'])
    
    class Continuous:
        
        def __init__(self, info, base_directory, version, mmap_timestamps=True):
            
            directory = os.path.join(base_directory, 'continuous', info['folder_name'])
            self.directory = directory
            
            self.name = info['folder_name']

            self.metadata = {}

            if mmap_timestamps:
                self.mmap_mode = 'r'
            else:
                self.mmap_mode = None

            self.metadata['source_node_id'] = info['source_processor_id']
            self.metadata['source_node_name'] = info['source_processor_name']

            if version >= 0.6:
                self.metadata['stream_name'] = info['stream_name']
            else:
                self.metadata['stream_name'] = str(info['source_processor_sub_idx'])

            self.metadata['sample_rate'] = info['sample_rate']
            self.metadata['num_channels'] = info['num_channels']

            self.metadata['channel_names'] = [ch['channel_name'] for ch in info['channels']]
            self.metadata['channel_map'] = Recording.create_channel_map(info)

            self.metadata['bit_volts'] = [ch['bit_volts'] for ch in info['channels']]

            data = np.memmap(os.path.join(directory, 'continuous.dat'), mode='r', dtype='int16')
            self.samples = data.reshape((len(data) // self.metadata['num_channels'], 
                                         self.metadata['num_channels']))

            try:
                if version >= 0.6:
                    self.sample_numbers = np.load(os.path.join(directory, 'sample_numbers.npy'), mmap_mode=self.mmap_mode)
                    self.timestamps = np.load(os.path.join(directory, 'timestamps.npy'), mmap_mode=self.mmap_mode)
                    # Open ephys sometimes makes corrupt timestamps, we want to check that here and fix them
                    self._check_timestamps()
                else:
                    self.sample_numbers = np.load(os.path.join(directory, 'timestamps.npy'), mmap_mode=self.mmap_mode)
            except FileNotFoundError as e:
                if os.path.basename(e.filename) == 'sample_numbers.npy':
                    self.sample_numbers = np.arange(self.samples.shape[0])

            # Check for saved global timestamps
            if os.path.isfile(os.path.join(directory, 'global_timestamps.npy')):
                self.global_timestamps = np.load(os.path.join(directory, 'global_timestamps.npy'), 
                                                 mmap_mode=self.mmap_mode)
            else:
                self.global_timestamps = None


        def get_data(self, start_time, end_time=None, length=None, channels=0):
            """
            Returns samples scaled to microvolts. Converts sample values
            from 16-bit integers to 64-bit floats.
            Finds the sample to fetch by looking for global timestamps 
            or if not available standard timestamps
    
            """
            import numpy as np
            
            # Determine the end time if not provided
            if end_time is None:
                if length is None:
                    length = 2.0  # Default to 2 seconds
                end_time = start_time + length
            
            # Get the timestamps from the memory mapped data
            if self.global_timestamps is None:
                timestamps = self.timestamps
            else: 
                timestamps = self.global_timestamps

            # Find the indices that correspond to the start and end times
            # using binary search or numpy searchsorted for efficiency

            start_index = np.searchsorted(timestamps, start_time, side='left')
            end_index = np.searchsorted(timestamps, end_time, side='right')
            
            # Handle edge cases
            if start_index >= len(timestamps):
                raise ValueError(f"Start time {start_time} is beyond the available data range")
            
            if end_index > len(timestamps):
                end_index = len(timestamps)
            
            if start_index == end_index:
                end_index = start_index + 1  # Ensure we get at least one sample
            
            # Prepare the channels parameter as a numpy array
            if isinstance(channels, (int, np.integer)):
                selected_channels = np.array([channels], dtype=int)
            elif isinstance(channels, (list, tuple)):
                selected_channels = np.array(channels, dtype=int)
            elif isinstance(channels, np.ndarray):
                # Ensure the array is of integer type
                selected_channels = channels.astype(int)
            else:
                raise TypeError("Channels must be an int, list, tuple, or numpy array")
            
            # Validate channels
            num_channels = self.metadata['num_channels']
            for channel in selected_channels:
                if not 0 <= channel < num_channels:
                    raise ValueError(f"Channel {channel} is out of range. Valid range is 0 to {num_channels-1}")
            
            samples     = self.get_samples(start_index, end_index, selected_channels)
            sample_time = timestamps[start_index:end_index]

            return samples, sample_time



        def get_samples(self, start_sample_index, end_sample_index, selected_channels=None, *, channel_by_number = None):
            """
            Returns samples scaled to microvolts. Converts sample values
            from 16-bit integers to 64-bit floats.

            Parameters
            ----------
            start_sample_index : int
                Index of the first sample to return
            end_sample_index : int
                Index of the last sample to return
            selected_channels : numpy.ndarray, optional
                Array index of data to extract. The channel you will be returned is 
                the argument+1 as arrays are zero-indexed. Internally, the channel returned 
                will be looked up as described in ``channel_by_number``.
            channel_by_number : numpy.ndarray, optional 
                Channel number(s) that you request. The array index is looked-up from a 
                dict that translates the channel ID (an interger of version of channel
                name where ``'CH22' = 22``)  to the index of the storage array.
                Order is kept consisitent with the **Channel Map** plugin as recorded in the 
`               ``oebin`` file. By default, all channels are returned. If you board has 
                additional ``ADCn`` channels, they are sequentially numbered after reaching
                the last ``CHnn`` labeled channel. 

            Returns
            -------
            samples : numpy.ndarray (float64)

            """

            if selected_channels and channel_by_number:
                raise ValueError("Cannot use both ``selected_channels`` and ``channel_by_number`` channel selection methods") 
            elif selected_channels and not channel_by_number:
                print("WARNING: You are selecting channels by array index, not channel ID!\n"
                      "         Channel number will be the array index +1 by default\n"
                      "         Use ``channel_by_number`` keyword to select channels by ID\n"
                      "           This is important when channel ordering has changed due to\n"
                      "           the use of the channel selector plugin.")

                if type(selected_channels) is int:
                    selected_channels = np.array([selected_channels],dtype=np.uint32)
                    selected_channels += 1 
                elif isinstance(np.ndarray,type(selected_channels)):
                    selected_channels += 1 
                else:
                    selected_channels = np.array(selected_channels,dtype=np.uint32)
                    selected_channels += 1 
            elif not selected_channels and channel_by_number:
                if type(channel_by_number) is int:
                    selected_channels = np.array([channel_by_number],dtype=np.uint32)
                elif isinstance(np.ndarray,type(channel_by_number)):
                    pass
                else:
                    selected_channels = np.array(channel_by_number,dtype=np.uint32)
            else:
                selected_channels = np.arange(self.metadata['num_channels'],dtype=np.uint32)
                selected_channels += 1 #change index to match channel ID, not array index

            selected_ch = np.array([ self.metadata['channel_map'][ch] for ch in selected_channels ],dtype=np.uint32)

            samples = self.samples[start_sample_index:end_sample_index, selected_ch].astype('float64')

            for idx, ch in enumerate(selected_ch):
                samples[:,idx] = samples[:,idx] * self.metadata['bit_volts'][ch]

            return samples

        def _check_timestamps(self, sample_size=100000):
            """
            Checks for discontinuities in timestamps array.
            If discontinuities are found, regenerates timestamps from sample numbers,
            backs up the original timestamps file, and writes the new timestamps.
            
            Returns
            -------
            bool
                True if timestamps were corrupted and fixed, False otherwise
            """
            # Check if timestamps exist
            if not hasattr(self, 'timestamps') or self.timestamps is None:
                return False

           # Check if timestamps start at zero
            starts_at_zero = self.timestamps[0] == 0
            is_corrupted = False

            # Check for discontinuities in timestamps
            if starts_at_zero: # Check for discontinuities in timestamps
                if len(self.timestamps) <= 1:
                    return False  # Not enough timestamps to check for discontinuities
                    
                # if memory mapped just load a sub sample
                if self.mmap_mode == 'r':
                    # Calculate differences between a sample of timestamps
                    timestamp_diffs = np.diff(self.timestamps[0:min(sample_size,len(self.timestamps))])
                else: 
                    # Calculate differences between consecutive timestamps
                    timestamp_diffs = np.diff(self.timestamps)
                    
                # Calculate the median difference (expected time between samples)
                median_diff = np.median(timestamp_diffs)
                
                # Check for significant deviations from the expected difference
                # (allowing for small floating-point variations)
                tolerance = 0.1 * median_diff  # 10% tolerance
                is_corrupted = np.any(np.abs(timestamp_diffs - median_diff) > tolerance)
            
            if is_corrupted or  ~starts_at_zero:
                print(f"Discontinuities detected in timestamps for {self.name}. Regenerating timestamps...")
                
                # Generate new timestamps from sample numbers and sampling rate
                # new_timestamps = self.sample_numbers / self.metadata['sample_rate']
                sample_range = np.arange(len(self.timestamps))
                new_timestamps = sample_range / self.metadata['sample_rate']
                
                # Backup the original timestamps file
                timestamps_file = os.path.join(self.directory, 'timestamps.npy')
                backup_file = os.path.join(self.directory, 'timestamps.npy.bkup')
                
                # Check if backup already exists to avoid overwriting previous backups
                backup_index = 1
                while os.path.exists(backup_file):
                    backup_file = os.path.join(self.directory, f'timestamps.npy.bkup{backup_index}')
                    backup_index += 1
                
                # Copy the original file to backup
                import shutil
                shutil.copy2(timestamps_file, backup_file)
                print(f"Original timestamps backed up to {backup_file}")

                # Save the new timestamps
                np.save(timestamps_file, new_timestamps)
                print(f"New timestamps saved to {timestamps_file}")
                
                # Update the timestamps in memory
                if self.mmap_mode is None:
                    self.timestamps = new_timestamps
                else:
                    # Reload the timestamps with the specified mmap_mode
                    self.timestamps = np.load(timestamps_file, mmap_mode=self.mmap_mode)

                # Try and do the same for any related event files
                # This is a little hacky, should actually change the events 
                # to be a proper object like the continuos data streams
                event_path = self.directory.replace('continuous', 'events') + 'TTL'
                event_samples_path    = os.path.join(event_path,'sample_numbers.npy')
                if os.path.exists(event_samples_path):
                    start_sample = self.sample_numbers[0]
                    event_timestamps = np.load(event_samples_path) - start_sample

                    # Generate new timestamps from sample numbers and sampling rate
                    new_event_timestamps = event_timestamps / self.metadata['sample_rate']
                
                    # Backup the original timestamps file
                    event_timestamps_path = os.path.join(event_path,'timestamps.npy')
                    event_backup_file = os.path.join(event_path, 'timestamps.npy.bkup')
                
                    # Check if backup already exists to avoid overwriting previous backups
                    event_backup_index = 1
                    while os.path.exists(event_backup_file):
                        event_backup_file = os.path.join(event_path, f'timestamps.npy.bkup{event_backup_index}')
                        event_backup_index += 1
                
                    # Copy the original file to backup
                    import shutil
                    shutil.copy2(event_timestamps_path, event_backup_file)
                    print(f"Original event timestamps backed up to {event_backup_file}")

                    # Save the new timestamps
                    np.save(event_timestamps_path, new_event_timestamps)
                    print(f"New event timestamps saved to {event_timestamps_path}")

                return True
            
            return False

    
    def __init__(self, directory, experiment_index=0, recording_index=0, mmap_timestamps=True):
        
       Recording.__init__(self, directory, experiment_index, recording_index, mmap_timestamps)  
       
       with open(os.path.join(self.directory, 'structure.oebin'), 'r') as oebin_file:
            self.info = json.load(oebin_file)
       self._format = 'binary'
       self._version = float(".".join(self.info['GUI version'].split('.')[:2]))
       self.sort_events = True       
       
    def load_continuous(self):
        
        self._continuous = []

        for info in self.info['continuous']:
            
            try:
                c = self.Continuous(info, self.directory, self._version, self.mmap_timestamps)
            except FileNotFoundError as e:
                print(info["folder_name"] + " missing file: '" + os.path.basename(e.filename) + "'")
            else:
                self._continuous.append(c)
            
    def load_spikes(self):
        
        self._spikes = []
        
        self._spikes.extend([self.Spikes(info, self.directory, self._version) 
                             for info in self.info['spikes']])

    
    def load_events(self):

        import numpy as np

        search_string = os.path.join(self.directory,
                                    'events',
                                    '*',
                                    'TTL*')
        
        events_directories = glob.glob(search_string)
        
        df = []
        
        streamIdx = -1
        
        for events_directory in events_directories:
            
            node_name = os.path.basename(os.path.dirname(events_directory)).split('.')
            node = node_name[0]
            nodeId = int(node.split("-")[-1])
            stream = ''.join(node_name[1:])
            
            streamIdx += 1
            
            if self._version >= 0.6:
                channels = np.load(os.path.join(events_directory, 'states.npy'))
                sample_numbers = np.load(os.path.join(events_directory, 'sample_numbers.npy'))
                timestamps = np.load(os.path.join(events_directory, 'timestamps.npy'))
            else:
                channels = np.load(os.path.join(events_directory, 'channel_states.npy'))
                sample_numbers = np.load(os.path.join(events_directory, 'timestamps.npy'))
                timestamps = np.ones(sample_numbers.shape) * -1
        
            # Check for global samples
            global_samples_filepath = os.path.join(events_directory, 'global_samples.npy')
            if os.path.exists(global_samples_filepath):
                global_samples = np.load(global_samples_filepath)
            else:
                global_samples = np.ones_like(timestamps) * np.nan

            # Try to get global timestamps too
            global_timestamps_filepath = os.path.join(events_directory, 'global_timestamps.npy')
            if os.path.exists(global_timestamps_filepath):
                global_timestamps = np.load(global_timestamps_filepath)
            else:
                global_timestamps = np.ones_like(timestamps) * np.nan

            # Find the matching continuous data to check for stream length   
            cont_samples = False         
            for cont in self.continuous:
                if cont.metadata['source_node_id'] == nodeId:
                    cont_samples = cont.sample_numbers

            # Convert on off states to durations
            if channels.size > 0:        
                states = channels
                channels = np.unique(np.abs(channels))            
                if states.size > 0:                
                    
                    rising_indices = []
                    falling_indices = []

                    for channel in channels:
                        # Find rising and falling edges for each channel
                        rising = np.where(states == channel)[0]
                        falling = np.where(states == -channel)[0]

                        # Ensure each rising has a corresponding falling
                        if rising.size > 0 and falling.size > 0:
                            if rising[0] > falling[0]:
                                falling = falling[1:]
                            if rising.size > falling.size:
                                rising = rising[:-1]

                            # # ensure that the number of rising and falling edges are the same:
                            # if len(rising) != len(falling):
                            #     print(
                            #         f"Channel {channel} has {len(rising)} rising edges and "
                            #         f"{len(falling)} falling edges. The number of rising and "
                            #         f"falling edges should be equal. Skipping events from this channel."
                            #     )
                            #     continue


                        durations = sample_numbers[falling] - sample_numbers[rising]
                        durations = durations / self.continuous[0].metadata["sample_rate"]
                        channel_samples = sample_numbers[rising]
                        channel_global_samples = global_samples[rising]
                        channel_timestamps = timestamps[rising]        
                        channel_global_timestamps = global_timestamps[rising]            

                        if cont_samples is not False:
                            # Find indices
                            # indices = np.where(np.isin(cont_samples, channel_samples))[0]
                            # Faster method
                            indices = channel_samples - cont_samples[0]

                        df.append(pd.DataFrame(data = {'line' :  [channel] * len(durations),
                                        'sample_number' : channel_samples,  
                                        'sample_index'  : indices,             
                                        'global_sample_index': channel_global_samples,
                                        'timestamp' : channel_timestamps,
                                        'global_timestamp' : channel_global_timestamps,
                                        'duration' : durations,
                                        'processor_id' : [nodeId] * len(durations),
                                        'stream_index' : [streamIdx] * len(durations),
                                        'stream_name' : [stream] * len(durations)}
                                        ))



        if len(df) > 0:

            self._events = pd.concat(df)

            if self.sort_events:
                if self._version >= 0.6:                  
                    self._events.sort_values(by=['stream_index', 'sample_number'], 
                                             ignore_index=True,
                                             inplace=True)
                else:
                    self._events.sort_values(by=['stream_index', 'sample_number'], 
                                             ignore_index=True,
                                             inplace=True)
        
        else:
            
            self._events = None
    
    def load_barcode_data(self):
        self.barcode_data = {}
        loaded_barcodes = True
        for continuous in self.continuous:                
            if os.path.isfile(os.path.join(continuous.directory,'barcodes.npy')):
                barcodes = np.load(os.path.join(continuous.directory,'barcodes.npy'))
                # assign to self - get key first
                stream_name = continuous.metadata['stream_name']
                node_id = continuous.metadata['source_node_id']
                self.barcode_data[(node_id, stream_name)] = barcodes
            else: 
                loaded_barcodes = False
        return loaded_barcodes
        

    def load_messages(self):
        
        if self._version >= 0.6:
            search_string = os.path.join(self.directory,
                            'events',
                            'MessageCenter')
        else:
            search_string = os.path.join(self.directory,
                            'events',
                            'Message_Center-904.0', 'TEXT_group_1'
                            )

        msg_center_dir = glob.glob(search_string)

        df = []

        if len(msg_center_dir) == 1:

            msg_center_dir = msg_center_dir[0]

            if self._version >= 0.6:
                sample_numbers = np.load(os.path.join(msg_center_dir, 'sample_numbers.npy'))
                timestamps = np.load(os.path.join(msg_center_dir, 'timestamps.npy'))
            else:
                sample_numbers = np.load(os.path.join(msg_center_dir, 'timestamps.npy'))
                timestamps = np.zeros(sample_numbers.shape) * -1

            text = [msg.decode('utf-8') for msg in np.load(os.path.join(msg_center_dir, 'text.npy'))]

            df = pd.DataFrame(data = { 'sample_number' : sample_numbers,
                    'timestamp' : timestamps,
                    'message' : text} )

        if len(df) > 0:

            self._messages = df

        else:

            self._messages = None

    def __str__(self):
        """Returns a string with information about the Recording"""
        
        return "Open Ephys GUI Recording\n" + \
                "ID: " + hex(id(self)) + '\n' + \
                "Format: Binary\n" + \
                "Directory: " + self.directory + "\n" + \
                "Experiment Index: " + str(self.experiment_index) + "\n" + \
                "Recording Index: " + str(self.recording_index)
    

    
    
    
    
    
    
    
    
    
    
    
    #####################################################################
    
    @staticmethod
    def detect_format(directory):
        binary_files = glob.glob(os.path.join(directory, 'experiment*', 'recording*'))
        
        if len(binary_files) > 0:
            return True
        else:
            return False
    
    @staticmethod
    def detect_recordings(directory, mmap_timestamps=True):
        
        recordings = []
        
        experiment_directories = glob.glob(os.path.join(directory, 'experiment*'))
        experiment_directories.sort(key=alphanum_key)

        for experiment_index, experiment_directory in enumerate(experiment_directories):
             
            recording_directories = glob.glob(os.path.join(experiment_directory, 'recording*'))
            recording_directories.sort(key=alphanum_key)
            
            for recording_index, recording_directory in enumerate(recording_directories):
            
                recordings.append(BinaryRecording(recording_directory, 
                                                       experiment_index,
                                                       recording_index,
                                                       mmap_timestamps))
                
        return recordings

    @staticmethod
    def create_oebin_file(
        output_path, 
        stream_name="example_data",
        channel_count=16,
        sample_rate=30000.,
        bit_volts=0.195,
        source_processor_name=None,
        source_processor_id=100):

        """
        Generates structure.oebin (JSON) file for one data stream

        A minimal directory structure for the Binary format looks 
        like this:

        data-directory/
            continuous/
                stream_name/
                    continuous.dat
            structure.oebin

        To export a [samples x channels] numpy array, A (in microvolts), into 
        a .dat file, use the following code: 

        >> A_scaled = A / bit_volts # usually 0.195
        >> A_scaled.astype('int16').tofile('/path/to/continuous.dat')

        Parameters
        ----------
        output_path : string
            directory in which to write the file (structure.oebin will
            be added automatically)
        stream_name : string
            name of the sub-directory containing the .dat file
        channel_count : int
            number of channels stored in the .dat file
        sample_rate : float
            samples rate of the .dat file
        bit_volts : float
            scaling factor required to convert int16 values in to µV
        source_processor_name : string
            name of the processor that generated this data (optional)
        source_processor_id : string
            3-digit identifier of the processor that generated this data (optional)
        
        """

        output = dict()
        output["GUI version"] = "0.6.0"

        if source_processor_name is None:
            source_processor_name = stream_name
        
        output["continuous"] = [{
            "folder_name" : stream_name,
            "sample_rate" : sample_rate,
            "stream_name" : stream_name,
            "source_processor_id" : source_processor_id,
            "source_processor_name" : stream_name,
            "num_channels" : channel_count,
            "channels": [{
                    "channel_name" : "CH" + str(i+1),
                    "bit_volts" : bit_volts
                    } for i in range(channel_count)]
        }]

        with open(os.path.join(
            output_path, 
            'structure.oebin'), "w") as outfile:
            outfile.write(json.dumps(output, indent=4))
