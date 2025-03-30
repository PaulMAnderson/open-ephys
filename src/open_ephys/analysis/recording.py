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


from abc import ABC, abstractmethod
import warnings
import glob
import os

class Recording(ABC):
    """ Abstract class representing data from a single Recording
    
    Classes for different data formats should inherit from this class.
    
    Recording objects contain three properties:
        - continuous
        - events
        - spikes
        - messages
    
    which load the underlying data upon access.
    
    continuous is a list of data streams
        - samples (memory-mapped array of dimensions samples x channels)
        - sample_numbers (array of length samples)
        - timestamps (array of length samples)
        - metadata (contains information about the data source)
            - channel_names
            - bit_volts
            - source_node_id
            - stream_name
        
    spikes is a list of spike sources
        - waveforms (spikes x channels x samples)
        - sample_numbers (one per sample)
        - timestamps (one per sample)
        - electrodes (index of electrode from which each spike originated)
        - metadata (contains information about each electrode)
            - electrode_names
            - bit_volts
            - source_node_id
            - stream_name
        
    events is a pandas DataFrame containing six columns:
        - timestamp
        - sample_number
        - line
        - state (1 or 0)
        - processor_id
        - stream_index

    messages is a pandas DataFrame containing three columns:
        - timestamp
        - sample_number
        - message
    
    """
    
    @property
    def barcodes(self):
        if self._barcodes is None:
            self.load_barcode_data()

    @property
    def continuous(self):
        if self._continuous is None:
            self.load_continuous()
            
            # If auto-sync was requested, do it now that data is loaded
            if self._auto_sync_pending:
                self.synchronize_timestamps()
                self._auto_sync_pending = False
                
        return self._continuous

    @property
    def events(self):
        if self._events is None:
            self.load_events()
        return self._events

    @property
    def spikes(self):
        if self._spikes is None:
            self.load_spikes()
        return self._spikes

    @property
    def messages(self):
        if self._messages is None:
            self.load_messages()
        return self._messages
    
    @property
    def format(self):
        return self._format


    def __init__(self, directory, experiment_index=0, recording_index=0, mmap_timestamps=True, auto_sync=True):
        """ Construct a Recording object, which provides access to
        data from one recording (start/stop acquisition or start/stop recording)
        
        Parameters
        ----------
        directory: location of Recording directory
        experiment_index: index of the experiment this recording belongs to
        recording_index: index of this recording within the experiment
        mmap_timestamps: bool, optional
            If True, timestamps will be memory-mapped for faster access
        auto_sync: bool, optional
            If True, automatically attempt to synchronize timestamps when loading
        """
        
        self.directory = directory
        self.experiment_index = experiment_index
        self.recording_index = recording_index
        self.mmap_timestamps = mmap_timestamps
        
        self._continuous = None
        self._events = None
        self._spikes = None
        self._messages = None
        
        # New attributes for barcode synchronization
        self.barcode_data = None
        self.raw_barcodes = None
        
        # If auto-sync is enabled, try to synchronize timestamps after loading data
        if auto_sync:
            # We need to defer this until after data is loaded
            # This will be called the first time .continuous is accessed
            self._auto_sync_pending = True
        else:
            self._auto_sync_pending = False
        
    @abstractmethod
    def load_spikes(self):
        pass
    
    @abstractmethod
    def load_events(self):
        pass
    
    @abstractmethod
    def load_continuous(self):
        pass

    @abstractmethod
    def load_messages(self):
        pass
    
    @abstractmethod
    def detect_format(directory):
        """Return True if the format matches the Record Node directory contents"""
        pass
    
    @abstractmethod
    def detect_recordings(directory, mmap_timestamps=True):
        """Finds Recordings within a Record Node directory"""
        pass
    
    @abstractmethod
    def __str__(self):
        """Returns a string with information about the Recording"""
        pass

    @abstractmethod
    def load_barcode_data(self):
        """Loads saved barcode data"""
        pass
    

    
    def add_sync_line(self, line, processor_id, stream_name=None, main=False, ignore_intervals=[]):
        """Specifies an event channel to use for timestamp synchronization. Each 
        sync line in a recording should receive its input from the same 
        physical digital input line.
        
        For synchronization to work, there must be one (and only one) 'main' 
        sync line, to which all timestamps will be aligned.
        
        Parameters
        ----------
        line : int
            event line number (1-based indexing)
        processor_id : int
            ID for the processor receiving sync events (eg 101)
        stream_name : str
            name of the stream receiving sync events (eg 'Probe-A-AP')
            default = None
        main : bool
            if True, this stream's timestamps will be treated as the 
            main clock
        ignore_intervals : list of tuples
            intervals to ignore when checking for common events
            default = []
        
        """

        events_on_line = self.events[(self.events.line == line) &
                                     (self.events.processor_id == processor_id) &
                                     (self.events.stream_name == stream_name)]
        

        if len(events_on_line) == 0:
            raise Exception('No events found on this line. ' + 
                            'Check that the processor ID and stream name are correct.')
        
        if main:
            existing_main = [sync for sync in self.sync_lines 
                             if sync['main']]
            
            if len(existing_main) > 0:
                raise Exception('Another main sync line already exists. ' + 
                                'To override, add it again with main=False.')
                
        matching_node = [sync for sync in self.sync_lines 
                         if sync['processor_id'] == processor_id and
                            sync['stream_name'] == stream_name]
        
        if len(matching_node) == 1:
            self.sync_lines.remove(matching_node[0])
            warnings.warn('Another sync line exists for this processor/stream combination, overwriting.')

        self.sync_lines.append({'line' : line,
                                'processor_id' : processor_id,
                                'stream_name' : stream_name,
                                'main' : main,
                                'ignore_intervals' : ignore_intervals})

    @staticmethod    
    def create_channel_map(info):
        """ Create channel map to generate list of channel 
        names that map to indices. The `channels` key maps 
        to an (ordered) list of channels with tags of the 
        form CHn, or ADCn. Notes regarding conversion of 
        channel to array index map:
        
            The addition of +1 to channels that are not 
            have repeat indices (with different 
            leading charactars) is important as it 
            keeps the channel numbering consistennt 
            so that channels match their ID, not their 
            array index!
            
            If channel #64 is the last CHn channel
            followed by AC1, then the AC1 channel 
            should be 65 to always allow an integer 
            index to each channel and count on
            the user having to use array index
            for non CHn labeled channels.

        Parameters
        ----------
        info : dict

        Returns
        -------
        channel_map: dict

        """
        channel_names = [ch['channel_name'] for ch in info['channels']]

        channel_map = {}
        
        for i,c in enumerate(channel_names):
            if c.startswith('CH'):
                try:
                    ch_id = int(c.lstrip('CH'))
                except:
                    ch_id = i+1
            elif c.startswith('ADC'):
                ch_id = i+1
            else:
                ch_id = i+1
            channel_map[ch_id] = i

        return channel_map

    def compute_global_timestamps(self, overwrite=False):
        """After sync channels have been added, this function computes the
        global timestamps for all processors with a shared sync line.

        Parameters
        ----------
        overwrite : bool
            if True, overwrite existing timestamps
            if False, add an extra "global_timestamp" column
            default = False
        
        """
        
        if len(self.sync_lines) == 0:
            raise Exception('At least two sync lines must be specified ' + 
                            'using `add_sync_line` before global timestamps ' + 
                            'can be computed.')

        main_line = [sync for sync in self.sync_lines 
                             if sync['main']]
        
        aux_lines = [sync for sync in self.sync_lines 
                             if not sync['main']]
            
        if len(main_line) == 0:
            raise Exception('Computing global timestamps requires one ' + 
                            'main sync line to be specified.')
            
        main_line = main_line[0]

        main_events = self.events[(self.events.line == main_line['line']) & 
                   (self.events.processor_id == main_line['processor_id']) & 
                   (self.events.stream_name == main_line['stream_name']) &
                   (self.events.state == 1)]
        
        # sort by sample number, in case the original timestamps were incorrect
        main_events = main_events.sort_values(by='sample_number')

        # remove any events that fall within the ignore intervals
        for ignore_interval in main_line['ignore_intervals']:
            main_events = main_events[(main_events.sample_number < ignore_interval[0]) |
                                      (main_events.sample_number > ignore_interval[1])]
        
        main_start_sample = main_events.iloc[0].sample_number
        main_total_samples = main_events.iloc[-1].sample_number - main_start_sample
        main_line['start'] = main_start_sample
        main_line['scaling'] = 1
        main_line['offset'] = main_start_sample

        for continuous in self.continuous:

            if (continuous.metadata['source_node_id'] == main_line['processor_id']) and \
               (continuous.metadata['stream_name'] == main_line['stream_name']):
               main_line['sample_rate'] = continuous.metadata['sample_rate']
        
        print(f'Processor ID: {main_line["processor_id"]}, Stream Name: {main_line["stream_name"]}, Line: {main_line["line"]} (main sync line))')
        print(f'  First event sample number: {main_line["start"]}')
        print(f'  Last event sample number: {main_events.iloc[-1].sample_number}')
        print(f'  Total sync events: {len(main_events)}')
        print(f'  Sample rate: {main_line["sample_rate"]}')

        for aux in aux_lines:
            
            aux_events = self.events[(self.events.line == aux['line']) &
                   (self.events.processor_id == aux['processor_id']) &
                   (self.events.stream_name == aux['stream_name']) &
                   (self.events.state == 1)]
            
            # sort by sample number, in case the original timestamps were incorrect
            aux_events = aux_events.sort_values(by='sample_number')

            # remove any events that fall within the ignore intervals
            for ignore_interval in aux['ignore_intervals']:
                aux_events = aux_events[(aux_events.sample_number < ignore_interval[0]) |
                                      (aux_events.sample_number > ignore_interval[1])]
            
            aux_start_sample = aux_events.iloc[0].sample_number
            aux_total_samples = aux_events.iloc[-1].sample_number - aux_start_sample
            
            aux['start'] = aux_start_sample
            aux['scaling'] = main_total_samples / aux_total_samples
            aux['offset'] = main_start_sample
            aux['sample_rate'] = main_line['sample_rate']

            print(f'Processor ID: {aux["processor_id"]}, Stream Name: {aux["stream_name"]}, Line: {main_line["line"]} (aux sync line))')
            print(f'  First event sample number: {aux["start"]}')
            print(f'  Last event sample number: {aux_events.iloc[-1].sample_number}')
            print(f'  Total sync events: {len(aux_events)}')
            print(f'  Scale factor: {aux["scaling"]}')
            print(f'  Actual sample rate: {aux["sample_rate"] / aux["scaling"]}')

        for sync_line in self.sync_lines: # loop through all sync lines
            
            for continuous in self.continuous:

                if (continuous.metadata['source_node_id'] == sync_line['processor_id']) and \
                   (continuous.metadata['stream_name'] == sync_line['stream_name']):
                       
                    continuous.global_timestamps = \
                        ((continuous.sample_numbers - sync_line['start']) * sync_line['scaling'] \
                            + sync_line['offset']) 
                    
                    global_timestamps = continuous.global_timestamps / sync_line['sample_rate']
                            
                    if overwrite:
                        continuous.timestamps = global_timestamps
                    else:
                        continuous.global_timestamps = global_timestamps
                            
            event_inds = self.events[(self.events.processor_id == sync_line['processor_id']) & 
                   (self.events.stream_name == sync_line['stream_name'])].index.values

            global_timestamps = (self.events.loc[event_inds].sample_number - sync_line['start']) \
                                  * sync_line['scaling'] \
                                   + sync_line['offset']
                                   
            global_timestamps = global_timestamps / sync_line['sample_rate']
            
            if overwrite:
                self.events.loc[event_inds, 'timestamp'] = global_timestamps
            else:
                for ind, ts in zip(event_inds, global_timestamps):
                    self.events.at[ind, 'global_timestamp'] = ts
    
                                              
    # First, locate the Recording class definition in recording.py
    # Add these methods to the Recording class (not inside any other method)

    def detect_barcode_lines(self, short_duration = 10, long_duration = 30, tolerance = 1):
        """
        Identify event lines that contain synchronization barcodes across all streams.
        
        Barcodes are characterized by:
        - Starting with 1+ 10ms on-off pulses
        - Followed by 32-bit code of 30ms on-off pulses
        - Ending with 1+ 10ms on-off pulses
        
        Returns
        -------
        dict
            Dictionary mapping processor_id and stream_name to the line number 
            containing barcodes
        """
        import numpy as np

        barcode_lines = {}
        
        # Examine events dataframe
        if not hasattr(self, 'events') or len(self.events) == 0:
            print("No events data found in recording")
            return barcode_lines        
        else: 
            self.events['pulse_type'] = ''
            self.events['line_type'] = ''
            self.events['barcode_num'] = np.nan

        # Group events by processor_id, stream_name, and line
        grouped_events = self.events.groupby(['processor_id', 'stream_name', 'line'])
        
        for (processor_id, stream_name, line), events in grouped_events:
            # Sort events by timestamp
            events_sorted = events.sort_values('sample_number')
            
            # Calculate intervals between state changes
            if len(events_sorted) < 10:  # Need minimum number of events to detect pattern
                continue

            # Convert to ms                
            durations = events_sorted.duration.values * 1000
            intervals = np.append(0,np.diff(events_sorted.sample_number.values / self._continuous[0].metadata['sample_rate']) * 1000)
        
            # Start Generation Here
            # Identify if short_duration pulses
            short_durations = (durations >= short_duration - tolerance) & (durations <= short_duration + tolerance)
            events_sorted.loc[short_durations, 'pulse_type'] = 'short'
            self.events.loc[events_sorted[short_durations].index, 'pulse_type'] = 'short'

            # Identify long_duration (or multiples thereof) pulses
            closest_multiple = np.round(durations / long_duration) * long_duration
            multiples_of_long = np.isclose(durations - closest_multiple, 0, atol=tolerance)  # Allowing +/-2 ms tolerance

            # Exclude values where the closest multiple is 0 (which happens for very short durations)
            multiples_of_long = multiples_of_long & (closest_multiple >= long_duration)
            events_sorted.loc[multiples_of_long, 'pulse_type'] = 'long'
            self.events.loc[events_sorted[multiples_of_long].index, 'pulse_type'] = 'long'

            valid_durations = short_durations | multiples_of_long
            # invalid_durations = ~valid_durations
            # events_sorted[invalid_durations]
            proportion_valid = np.mean(valid_durations)
            if proportion_valid < 0.9:
                continue

            barcode_lines[(processor_id, stream_name)] = line
            # break

            barcode_count = 0
            first_short = False
            for i in events_sorted.index:
                if events_sorted.loc[i]['pulse_type'] == 'short' and not first_short:
                    first_short = True
                    barcode_count += 1
                    events_sorted.loc[i, 'barcode_num'] = barcode_count
                elif events_sorted.loc[i]['pulse_type'] == 'short' and first_short:
                    first_short = False
                    events_sorted.loc[i, 'barcode_num'] = barcode_count
                else:
                    events_sorted.loc[i, 'barcode_num'] = barcode_count
                
            self.events.loc[events_sorted.index, 'barcode_num'] = events_sorted['barcode_num']

        return barcode_lines

    def process_barcodes(self, short_duration = 10, long_duration = 30, tolerance = 1):
        
        import numpy as np
        import pandas as pd

        # First detect all barcode lines
        barcode_lines = self.detect_barcode_lines()
        
        if not barcode_lines:
            print("No barcode lines detected in any stream")
            return False
            
        # Decode barcodes from each stream
        self.barcode_data = {}
        self.raw_barcodes = []
        
        for (processor_id, stream_name), line in barcode_lines.items():
            # Get events for this line
            events = self.events[(self.events.processor_id == processor_id) & 
                                (self.events.stream_name == stream_name) &
                                (self.events.line == line)]
            
            # Sort events by timestamp
            events_sorted = events.sort_values('sample_number')
                    
            # Decode barcodes
            # Get durations of pulses in ms
            durations = events_sorted.duration.values * 1000
            durations[events_sorted.pulse_type == 'short'] = short_duration
            closest_multiple = np.round(durations / long_duration) * long_duration
            durations[events_sorted.pulse_type == 'long'] = closest_multiple[events_sorted.pulse_type == 'long'] 
            # convert sample numbers to timestamps
            timestamps = (events_sorted.sample_number.values / self._continuous[0].metadata['sample_rate'])
            events_sorted['timestamp'] = timestamps
            events_sorted['duration'] = durations

            # Get intervals between pulses in ms
            # intervals = np.diff(timestamps)
            # intervals_ms = intervals * 1000

            barcodes = self._decode_barcodes(events_sorted)
            
            barcode_values = [barcode['barcode_value'] for barcode in barcodes]
            barcode_timestamps = [barcode['start_time'] for barcode in barcodes]

            # Convert into a structured NumPy array with named fields
            barcode_data = np.array(
                list(zip(barcode_values, barcode_timestamps)),
                dtype=[("values", "i4"), ("timestamps", "f8")]  # "O" for generic object type, "f8" for float64
                )
                
            self.barcode_data[(processor_id, stream_name)] = barcode_data
            self.raw_barcodes.append(pd.DataFrame(barcodes))


        return True
        
    def synchronize_with_barcodes(self, main_processor_id=None, main_stream_name=None, short_duration = 10, long_duration = 30, tolerance = 2):
        """
        Decode barcodes in each stream and use them to synchronize timestamps
        across streams through interpolation.
        
        Parameters
        ----------
        main_processor_id : int, optional
            ID for the processor to use as the main clock reference
        main_stream_name : str, optional
            Name of the stream to use as the main clock reference
            
        Returns
        -------
        bool
            True if synchronization was successful, False otherwise
        """
        import numpy as np
        from scipy.interpolate import interp1d
                

        # If no main processor/stream specified, use the one with most barcodes
        if main_processor_id is None or main_stream_name is None:
            max_barcodes = 0
            for (proc_id, stream_name), data in self.barcode_data.items():
                if len(data['values']) > max_barcodes:
                    max_barcodes = len(data['values'])
                    main_processor_id = proc_id
                    main_stream_name = stream_name
        
        # Get reference barcodes
        main_key = (main_processor_id, main_stream_name)
        if main_key not in self.barcode_data:
            print(f"Main processor/stream {main_key} not found in barcode data")
            return False
            
        main_barcodes = self.barcode_data[main_key]
        
        # Now synchronize each continuous object using interpolation
        for stream_index, continuous in enumerate(self.continuous):
            proc_id = continuous.metadata['source_node_id']
            stream_name = continuous.metadata['stream_name']
            key = (proc_id, stream_name)
            
            if key not in self.barcode_data:
                continue  # Skip if no barcodes
                
            stream_barcodes = self.barcode_data[key]
            
            # Find common barcodes
            # common_barcodes = set(stream_barcodes['values']).intersection(main_barcodes['values'])
            stream_mask = np.isin(stream_barcodes["values"], main_barcodes["values"])
            main_mask   = np.isin(main_barcodes["values"], stream_barcodes["values"])

            if len(stream_mask) < 2:
                print(f"Not enough common barcodes for {key}")
                continue
                
            # Create mapping between timestamps
            stream_times = stream_barcodes["timestamps"][stream_mask]
            main_times   = main_barcodes["timestamps"][main_mask]

            # Sort by stream timestamps to ensure proper interpolation
            sorted_indices = np.argsort(stream_times)
            stream_times = np.array(stream_times)[sorted_indices]
            main_times = np.array(main_times)[sorted_indices]
                                
            # Create interpolation function
            interp_func = interp1d(
                stream_times, 
                main_times, 
                kind='linear', 
                bounds_error=False, 
                fill_value='extrapolate'
            )

            if key == main_key:
                # assign the raw timestamps as global timestamps if this is the main key
                continuous.global_timestamps = np.array(continuous.timestamps)
                self.synchronize_events(continuous, interp_func)
                continue

            # Use interpolation to align timestamps
            timestamps = continuous.timestamps
            continuous.global_timestamps = interp_func(timestamps)
            # # Calculate sample numbers for each timestamp
            # sample_rate = continuous.metadata['sample_rate']
            # # Can't use actual timestamps as they are often corrupted, using sample numbers instead
            # timestamps = continuous.sample_numbers / sample_rate      
            
            # Attempt to fix the event timestamps
            self.synchronize_events(continuous, interp_func)

        # Save the results for future use
        self.save_global_timestamps()
        # Reload the event data
        self.load_events()


        return True

    # def _decode_barcodes(self, events_sorted):
    #     """
    #     Decode barcodes from a dataframe of pulse data, including gaps as zeros.
        
    #     Parameters:
    #     events_sorted: pandas DataFrame with columns:
    #         - 'pulse_type': 'short' or 'long'
    #         - 'duration': duration in ms
    #         - 'timestamp': start time of the pulse in seconds
            
    #     Returns:
    #     A list of dictionaries, each containing:
    #         - 'barcode_value': Decoded integer value
    #         - 'binary_string': Full binary string representation
    #         - 'start_time': Start time of the barcode in seconds
    #         - 'end_time': End time of the barcode in seconds
    #     """
    #     import pandas as pd
    #     import numpy as np
        
    #     barcodes = []
    #     i = 0
        
    #     # Process all events in the sorted dataframe
    #     while i < len(events_sorted):
    #         # Find the starting short pulse
    #         if events_sorted.iloc[i]['pulse_type'] == 'short':
    #             start_time = events_sorted.iloc[i]['timestamp']
    #             current_time = start_time + (events_sorted.iloc[i]['duration'] / 1000)
    #             i += 1  # Move past the starting short pulse
                
    #             # Reset barcode data for this new detection
    #             binary_string = ""
                
    #             # Process all pulses and gaps until we find the ending short pulse
    #             found_ending_short = False
    #             while i < len(events_sorted):
    #                 pulse = events_sorted.iloc[i]
                    
    #                 # Check if we've found the ending short pulse
    #                 if pulse['pulse_type'] == 'short':
    #                     found_ending_short = True
    #                     # There is a 10 ms delay before a short pulse begins
    #                     end_time = pulse['timestamp'] - (pulse['duration'] / 1000)     
    #                     break
                        
    #                 # Check if there's a gap before this pulse
    #                 gap_duration = (pulse['timestamp'] - current_time) * 1000  # Convert seconds to ms
    #                 if gap_duration >= 25:  # Using 25ms consistently as threshold for gaps
    #                     # Count gap as zeros (every 30ms = one '0')
    #                     num_zeros = int(round(gap_duration / 30))
    #                     binary_string += "0" * num_zeros
                    
    #                 # Process the long pulse as ones
    #                 if pulse['pulse_type'] == 'long':
    #                     num_bits = round(pulse['duration'] / 30)
    #                     binary_string += "1" * num_bits
                    
    #                 # Update current time to the end of this pulse
    #                 current_time = pulse['timestamp'] + (pulse['duration'] / 1000)  # Convert ms back to seconds
    #                 i += 1
                    
    #             # If we found a complete barcode (with ending short pulse)
    #             if found_ending_short:
    #                 # Check if we need to add trailing zeros based on any gap before the ending pulse
    #                 gap_duration = (end_time - current_time) * 1000  # Convert seconds to ms
    #                 if gap_duration >= 25:
    #                     num_zeros = int(round(gap_duration / 30))
    #                     binary_string += "0" * num_zeros
                        
    #                 # Check if we have exactly 32 bits
    #                 if len(binary_string) == 32:
    #                     # Convert binary string to integer
    #                     barcode_value = int(binary_string, 2)
                        
    #                     barcodes.append({
    #                         'barcode_value': barcode_value,
    #                         'binary_string': binary_string,
    #                         'start_time': start_time,
    #                         'end_time': end_time
    #                     })
    #                 else:
    #                     print(f"Warning: Found a barcode with {len(binary_string)} bits instead of 32 bits: {binary_string}")
    #                     # Optional: add partial barcodes to the results with a flag
    #                     barcodes.append({
    #                         'barcode_value': None,  # or compute anyway
    #                         'binary_string': binary_string,
    #                         'start_time': start_time,
    #                         'end_time': end_time,
    #                         'is_valid': False,
    #                         'bit_length': len(binary_string)
    #                     })
                    
    #                 # Move to the next pulse after the ending short pulse
    #                 i += 1
    #             else:
    #                 # If we didn't find an ending short pulse, move to the next event
    #                 i += 1
    #         else:
    #             # If the current pulse isn't a starting short pulse, move to the next one
    #             i += 1
                    
    #     return barcodes

    def _decode_barcodes(self, events_sorted, n_bits=32, interval=30, init_duration=10, pulse_duration=30, tolerance=0.1):
        """
        Function to decode binary barcodes from digital TTL pulse events
        
        Parameters:
        -----------m
        n_bits : int, optional
            Number of bits in barcode
        interval : int, optional
            Inter-barcode interval in ms
        init_duration : int, optional
            Duration of initialization pulses
        pulse_duration : int, optional
            Duration of barcode pulses
        tolerance : float, optional
            Proportion of variation to accept (0.1 = 10%)
        
        Returns:
        --------
        barcode_data : list of dict
            List containing decoded barcode information
        """
        import numpy as np

        barcode_data = []
        barcodes = np.unique(events_sorted['barcode_num'])        

        for barcode in barcodes:
            # Find relevant events
            barcode_events = events_sorted[events_sorted['barcode_num'] == barcode]

            # Start Generation Here
            if (barcode_events.iloc[0]['pulse_type'] != 'short') or (barcode_events.iloc[-1]['pulse_type'] != 'short'):
                print(f'Barcode {barcode} does not start and end with a short event.')
                # continue
                a = 1
            else:
                start_time = barcode_events.iloc[0]['timestamp']
                start_latency = barcode_events.iloc[0]['sample_number']
                pulse_end = barcode_events.iloc[0]['timestamp'] + 2 * (barcode_events.iloc[0]['duration'] / 1000)
                end_time = barcode_events.iloc[-1]['timestamp']
                end_latency = barcode_events.iloc[-1]['sample_number']
                last_pulse_time = barcode_events.iloc[-1]['timestamp'] - (barcode_events.iloc[-1]['duration'] / 1000)
            
            # Remove wrapper events
            barcode_events = barcode_events[barcode_events['pulse_type'] != 'short']
                
            barcode_string = ""

            for i in barcode_events.index:
                # Determine the number of empty bits at the start
                empty_bits = round(((barcode_events.loc[i]['timestamp'] - pulse_end) * 1000) / pulse_duration)
                barcode_string += "0" * empty_bits
                # Determine how many bits to add
                num_bits = round(barcode_events.loc[i]['duration'] / pulse_duration)
                barcode_string += "1" * num_bits
                # save the end time of the pulse
                pulse_end = barcode_events.loc[i]['timestamp'] + (barcode_events.loc[i]['duration'] / 1000)

            # Need to add trailing zeros 
            empty_bits = round((last_pulse_time - pulse_end) / pulse_duration)
            barcode_string += "0" * empty_bits
            
            # Calculate barcode value
            # barcode_value = int(barcode_string, 2)
            barcode_value = int(barcode_string[::-1], 2)
            
            barcode_data.append({
                'barcode_num': barcode,
                'start_time': start_time,
                'start_latency': start_latency,
                'barcode_value': barcode_value,
                'barcode_string': barcode_string
            })
            
        return barcode_data

    def save_global_timestamps(self):
        """
        Write global timestamps to disk for each continuous data stream.
        
        Saves as 'global_timestamps.npy' in the same location as the original timestamp files.
        Also saves barcode information for later retrieval.
        """
        import os
        import numpy as np
        
        # Check if global timestamps exist
        for stream_index, continuous in enumerate(self.continuous):
            if not hasattr(continuous, 'global_timestamps'):
                print(f"Global timestamps not found for stream {continuous.metadata['stream_name']}")
                continue
            
            # Save global timestamps
            global_file = os.path.join(continuous.directory, 'global_timestamps.npy')
            np.save(global_file, continuous.global_timestamps)
            print(f"Saved global timestamps for stream {continuous.metadata['stream_name']} to {global_file}")
        
        # Save barcode information if available
        if hasattr(self, 'barcode_data'):
            for (processor_id, stream_name), data in self.barcode_data.items():
                # Find the corresponding continuous object to get its directory
                for continuous in self.continuous:
                    if (continuous.metadata['source_node_id'] == processor_id and 
                        continuous.metadata['stream_name'] == stream_name):
                                                
                        # Save barcode values and timestamps
                        barcode_file = os.path.join(continuous.directory, 'barcodes.npy')
                        np.save(barcode_file, data)                                            
                        print(f"Saved barcode data for stream {stream_name} to {barcode_file}")
                        break

    def load_global_timestamps(self):
        """
        Load global timestamps from disk for each continuous data stream if available.
        Also loads barcode information if it exists.
        
        Returns
        -------
        bool
            True if any global timestamps were loaded, False otherwise
        """
        import os
        import numpy as np
        
        any_loaded = False
        self.barcode_data = {}
        
        for stream_index, continuous in enumerate(self.continuous):
            # Get the directory where timestamps are stored
            continuous_directory = continuous.directory
            event_parent_directory = continuous_directory.replace('/continuous/', '/events/')
            event_directories = os.listdir(event_parent_directory)
            if len(event_directories) == 0:
                print(f"No event directories found for stream {continuous.metadata['stream_name']}")
                continue
            for event_directory in event_directories:
                timestamp_dir = os.path.join(event_parent_directory, event_directory)
                global_timestamp_file = os.path.join(event_parent_directory, event_directory, 'global_timestamps.npy')
                if os.path.exists(global_timestamp_file):
                    continuous.global_timestamps = np.load(global_timestamp_file)
                    print(f"Loaded global timestamps for stream {continuous.metadata['stream_name']}")
                    any_loaded = True
            
                # Try to load barcode data
                barcode_file = os.path.join(timestamp_dir, 'barcodes.npz')
            
                if os.path.exists(barcode_file):
                    barcode_npz = np.load(barcode_file)
                    
                    proc_id = continuous.metadata['source_node_id']
                    stream_name = continuous.metadata['stream_name']
                    
                    self.barcode_data[(proc_id, stream_name)] = {
                        'values': barcode_npz['values'].tolist(),
                        'timestamps': barcode_npz['timestamps'].tolist()
                    }
                    
                    print(f"Loaded barcode data for stream {stream_name}")
        
        return any_loaded

    def synchronize_timestamps(self, main_processor_id=None, main_stream_name=None):
        """
        Synchronize timestamps across streams. First tries to load saved global timestamps,
        then tries to use saved barcode data, and finally attempts to detect and use barcodes
        from raw data.
        
        Parameters
        ----------
        main_processor_id : int, optional
            ID for the processor to use as the main clock reference
        main_stream_name : str, optional
            Name of the stream to use as the main clock reference
            
        Returns
        -------
        bool
            True if synchronization was successful, False otherwise
        """
        # Check if all streams have global_timestamps
        all_has_global = True
        for continous in self.continuous:
            if continous.global_timestamps is None:
                all_has_global = False
        if all_has_global:
            return True
        
        # Check for saved barcodes
        if not self.barcode_data or not self.barcode_data is None:
            if self.load_barcode_data():
                return self.synchronize_with_barcodes()
            else:
                if self.process_barcodes():
                    return self.synchronize_with_barcodes()
        else:
            return self.synchronize_with_barcodes()        
        
        print("Failed to synchronize timestamps")
        return False
    
    def synchronize_events(self,continuous, interp_func):
        """
        Attempts to synchronise event data using the barcode approach. 
        Relies on global timestamps having been calculated.
        Inputs are a continuous data stream and the interpolation function that syncs its timestamps
        """

        import numpy as np
        processor_id = continuous.metadata['source_node_id']
        stream_name  = continuous.metadata['stream_name']

        # Annoying we dont have the events directories saved...
        # Maybe make a full class analgous to the continuous one?
        event_path = continuous.directory.replace('continuous', 'events') + 'TTL'
        if os.path.exists(event_path):
            timestamps = np.load(os.path.join(event_path, 'timestamps.npy'))
            # Alternative method where we regenerate timestamps
            # not needed as we already do this now
            # samples    = np.load(os.path.join(event_path,'sample_numbers.npy'))
            # new_timestamps = samples / continuous.metadata['sample_rate']

            # Convert timestamps to global timestamps
            global_timestamps = interp_func(timestamps)

            # Save the global timestamps
            global_file = os.path.join(event_path, 'global_timestamps.npy')
            np.save(global_file, global_timestamps)
            print(f"New global event timestamps saved to {global_file}")

    def concatenate_events(self, other, save_data=True, output_dir=None):
        import pandas as pd
        """
        Attempts to concatenate event data between two recordings
        """

        stream_data = []
        for stream in self.continuous:
            stream_name = stream.metadata['stream_name']
            node_id     = stream.metadata['source_node_id']
            node_name   = stream.metadata['source_node_name']
            num_samples = len(stream.sample_numbers)
            stream_data.append({
                'stream_name':stream_name,
                'node_id':node_id,
                'node_name':node_name,
                'num_sample':num_samples,
                'last_timestamp':stream.timestamps[-1],
                'last_global_timestamp':stream.global_timestamps[-1]
            })

        other_events = other.events

        # Initialize an empty list to collect processed event
        processed_events = []
            
        # Iterate through unique stream names
        for stream_info in stream_data:
            stream_name = stream_info['stream_name']
            num_samples = stream_info['num_sample']
            last_timestamp = stream_info['last_timestamp']
            last_global_timestamp = stream_info['last_global_timestamp']

            # Subset the DataFrame for the current stream
            stream_events = other_events[other_events['stream_name'] == stream_name].copy()
            
            # Add samples tom sampl_index column
            stream_events['sample_index'] += num_samples
            # Add time offset to timestamp column
            stream_events['timestamp'] += last_timestamp
            # Add time offset to global_timestamp column
            stream_events['global_timestamp'] += last_global_timestamp
            
            # Append the modified subset to the list
            processed_events.append(stream_events)

        # Combine all processed DataFrames
        other_events  = pd.concat(processed_events, ignore_index=True)
        all_events = pd.concat([self.events, other_events], ignore_index=True)
        all_events = all_events.sort_values(by=['stream_index','sample_index']).reset_index(drop=True)

        if save_data:
            # Check output directory
            if output_dir is None:
                import open_ephys.analysis as oe
                # Find the parent directory of this recording and write the data there
                # This is going to be a bit hacky...
                sesh_dir = os.path.dirname(os.path.dirname(os.path.dirname(self.directory)))
                # Check it by running Session
                try:
                    if oe.Session(sesh_dir):
                        output_dir = os.path.dirname(sesh_dir)
                except:                
                    output_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(self.directory))))
                    print('Not sure about path accuracy...')


            # Save events to csv
            filename_csv = os.path.join(output_dir, 'all_events.csv')
            all_events.to_csv(filename_csv, index=False)
            print(f"saved all events to .csv at {filename_csv}")

            # 2. Parquet - High performance, columnar storage
            filename_parquet = os.path.join(output_dir, 'all_events.parquet')
            all_events.to_parquet(filename_parquet, engine='pyarrow')
            print(f"saved all events to .parquet at {filename_parquet}")

        return all_events
