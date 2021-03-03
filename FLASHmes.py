import numpy as np
#import matplotlib.pyplot as plt
#import scipy.ndimage
#from scipy.signal import butter, filtfilt, freqz
#import pickle
import deepdish as dd
#import h5py


class FLASHmes:
    """
    # FLASHmes is a python class for written for any event-based analysis of FLASH DAQ runs.
    # The class serves at the same time as an API to BeamtimeDAQaccess to simplify the read-in process and 
    # as a data structure that  bundles the relevant experimental data into numpy arrays.
    # The first dimension of any data handled by the class must be equal for all data channels and the index
    # is directly relatable to the FLASH bunch ID.
    # The class can also serve as a container for calculation results. The method concatenate_FLASHrun works for
    # all numpy arrays of the correct length (i.e. number of events loaded)
    # Autor: Robin Engel, 2019, robin.engel@desy.de
    #
    # Attributes:
    #    self._ids                      Flash event-ID vector
    #    self._loaded_runs              List of DAQ runs currently loaded by the class
    #    self._loaded_runs_intervals    List of event-ID intervals (tuples) of the loaded runs
    #    self.good                      boolean numpy array indicating if each event is valid. Initialized to True
    #    self.daqAccess                 beamtimedaqaccess object for data readin
    #    self.<channel key>             Loaded experimental data
    # Constructor:
    #     __init__(self, daqAccess, runNo, channels):
    #     <daqAccess> must be a beamtimedaqaccess object initialized for the respective beamtime.
    #     <runNo> is the number of a DAQ run. This run will be read in during the class initialization
    #     <channels> is a dictionray that contains valid channels for DeamtimeDAQaccess, in which the key must be a string
    #     that will be the name of the FLASHmes class attribute by which the data can later be acessed.
    #     Example:
    #        channels = {
    #        "delay" : "/FL2/Experiment/Pump probe laser/laser delay readback", # Not recognized by BeamtimeDaqAccess
    #        "mono"  : "/FL2/Photon Diagnostic/GMD/Expert stuff/XGM.PHOTONFLUX/FL2.HALL/WAVELENGTH.USED",
    #        "images" :   "/FL1/Experiment/MUSIX/Great Eyes/image",
    #        "attenuator_pos" : "/FL2/Experiment/Pump probe laser/FL24/attenuator position"
    #        } 
    # Methods:
    #     append_run(self, runNo, skipchannels = []):
    #          This method allows to read in an additional DAQ run and concatenate all data arrays with
    #          that are already loaded with the newly loaded ones.
    #
    #     concatenate_FLASHruns(self, newrun, skipchannels = []):
    #          This method allows to concatenate all data arrays of two already loaded instances of FLASHmes
    #
    #     save(self,filename, skipchannels = []):
    #          uses deepdish library to save a FLASHmes object using 'blosc' compression. Also for large datasets.
    #          Load using the inheriting "LoadedFLASHmes" class
    #
    #     eventfilter(self,propname, condition):
    #          sets all entries of <propname> (usually "good") to False if <condition> does not apply.
    #
    #     remove_run(self,runNo):
    #          removes all data associated with the run <runNo> from the loaded data
    #
    #     def update_run(self, runNo):
    #          self.remove_run(runNo)
    #          self.append_run(runNo)
    #
    #     eval_diode_matrix(diode_matrix, N_peaks, ADC_params, plot = False)
    #          An indexing-based method to extract peak pulse energies from a 
    #          Matrix of diode-traces. Runs much faster than the loop-based alternative.
    #          The diode_matrix should have the shape (No_Events,len(ADC_trace)).
    #          ADC_params should be a dictionary with the entries (all int):
    #          "first_peak_start","peak_distance","peak_width","pre_bkg_width"
    #           Returns a matrix "peak_energies" of the shape (No_events, No_pulses)

    """
    
    
    def __init__(self, daqAccess, runNo, channels, alternative_readin = False, altern_time = 1800):

        # Define Attributes
        self._ids = []
        self._loaded_runs = []
        self._loaded_runs_intervals = {}
        self.good = []
        self.daqAccess = daqAccess
        
        # Read data
        print('Reading run {}'.format(runNo))
        old_interval = None
        new_interval = None
        for chan in channels.keys(): # Read all the channels
            print('Reading channel: ' + chan)
            if alternative_readin == False:
                value, new_interval = daqAccess.allValuesOfRun(channelName=channels[chan],runNumber=runNo)
            elif alternative_readin == True:
                value, new_interval = daqAccess.firstValuesOfRun(channelName=channels[chan],runNumber=runNo, timeLengthSecs=altern_time)
                print(f'Warning: Backup-readin function used. Only considering {altern_time/60} min of scan-time for run '+str(runNo))

            assert new_interval == old_interval or old_interval == None
            old_interval = new_interval
            if value.shape[1]==1:
                value = value[:,0]
            setattr(self,chan, value ) 
        self.channels = channels
        self._ids = np.arange(new_interval[0], new_interval[1])
        self._loaded_runs =[runNo]
        self._loaded_runs_intervals[runNo] = new_interval
        self.good = np.ones(self._ids.shape[0],dtype=bool)
        self.update_length()

        print('Successfully loaded {} Events'.format(new_interval[1]-new_interval[0]))
            
    def append_run(self, runNo, skipchannels = []):
        # Concatenating another run to the loaded one
        print('Appending run {}'.format(runNo))

        if runNo in self._loaded_runs:
            raise ValueError('Run {} already loaded. Use update_run to update.'.format(runNo))
        # Read data
        old_interval = None
        new_interval = None
        for chan in self.channels.keys(): # Read all the channels
            #print('Appending channel: ' + chan)
            if not chan in skipchannels:
                value, new_interval = self.daqAccess.allValuesOfRun(channelName=self.channels[chan],runNumber=runNo)

                if not (new_interval == old_interval or old_interval == None):
                    raise ValueError('Run {}, Interval of channel {} not matching to expected interval {}'.format(runNo, chan, old_interval))
                old_interval = new_interval

                old_value = getattr(self, chan)
                if value.shape[1]==1:
                    value = value[:,0]
                new_events = np.concatenate((old_value, value))
                setattr(self,chan, new_events )
        self._loaded_runs.append(runNo)
        self._loaded_runs_intervals[runNo] = new_interval
        self._ids = np.concatenate((self._ids,np.arange(new_interval[0], new_interval[1])), axis  = 0)
        self.update_length()
        print('Appended {} values to all channels from run {}'.format(new_interval[1]-new_interval[0], runNo))
        
    def __add__(self, other):
        return concatenate_FLASHruns(self, other)
    
#    @property
#    def length(self):
#        return self._length #Changed it back to protected so that loadedFLASHmes can set the attribute in its init method
    def update_length(self):
        self._length = len(self._ids)
        
    
    def concatenate_FLASHruns(self, newrun, skipchannels = [], include_attributes= []):
        # Concatenating another run to the loaded one
        print('Starting concatenate...')
        
        for adrun in newrun._loaded_runs:
            if adrun in self._loaded_runs:
                raise ValueError('Run {} already loaded in present FLASHrun.')
            # Read data
            old_interval = None
            copy_interval = None
            for key in dir(self):
                copy_interval = newrun._loaded_runs_intervals[adrun]
                index_low = np.argwhere(newrun._ids==copy_interval[0])[0][0]#this is really ugly .. how better?
                index_high = np.argwhere(newrun._ids==copy_interval[1]-1)[0][0]+1
                copy_slice = np.s_[index_low:index_high]
                append_length = index_high-index_low
                #selfdata_length
                self.update_length()
                if not (key in skipchannels):
                    data = getattr(self,key)
                    if type(data) is np.ndarray and not (key[0]=='_' or type(data) is type(self.save)) :
                        #print(data.shape, selfdata_length)
                        if data.shape[0] == self._length or key in include_attributes:
                #for chan in self.channels.keys(): # Read all the channels
                            print('Appending: {} of run {}'.format(key,adrun))
                            if not key in skipchannels:

                                #copy_slice = np.s_[copy_interval[0]:copy_interval[1]]
                                #print(copy_slice)
                                value = getattr(newrun, key)
                                #print(value.shape)
                                value = value[copy_slice]
                                old_value = getattr(self, key)

                                new_events = np.concatenate((old_value, value))
                                #print(old_value.shape,value.shape,new_events.shape)

                                setattr(self,key, new_events )
            self._loaded_runs.append(adrun)
            self._loaded_runs_intervals[adrun] = copy_interval
            self._ids = np.concatenate((self._ids,np.arange(copy_interval[0], copy_interval[1])), axis  = 0)

        print('Appended {} runs to all channels except'.format(newrun._loaded_runs, skipchannels))

    def save(self,filename, skipchannels = []):
        """save(filename, skipchannels = [])
        Saves all class attributes into a dictionary wich is then saved using deepdish,
        for later re-loading with the class "loadedFLASHmes" """
        thelib = {}

        for key in dir(self):
            if not key in skipchannels:
                data = getattr(self,key)
                if not ((key[0]=='_' and key[1]=='_') or type(data) is type(self.save)) :
                    #print(key)
                    thelib[key] = data
                
        dd.io.save(filename,thelib, compression=('blosc', 9))
        print('Successfully saved run {} as "{}"'.format(self._loaded_runs, filename))#

                
        
    def eventfilter(self,propname, condition):
        if propname in dir(self):
            print('Property {} found. Combining...'.format(propname))
            setattr(self, propname, np.logical_and(getattr(self, propname), condition))
        else:
            print('Property {} not found. Creating new.'.format(propname))
            setattr(self, propname, condition)
        print('Done.')
            
    def remove_run(self,runNo):
        remove_interval = self._loaded_runs_intervals[runNo]
        remove_slice = np.s_[remove_interval[0]:remove_interval[1]]
        
        for chan in self.channels.keys(): # Delete Data
            setattr(self,chan, np.delete(getattr(self,chan), remove_slice, axis = 0))
        self._ids = np.delete(self._ids, remove_slice, axis = 0)
        self._loaded_runs.remove(runNo) # Delete run from loaded
        del self._loaded_runs_intervals[runNo]
        
        print('Removal of run {} complete'.format(runNo))
        
    def update_run(self, runNo):
        self.remove_run(runNo)
        self.append_run(runNo)
    
    def check_datalength(self):
        #returns False if any channel has a differnt length

        successlist = []
        for chan in self.channels.keys(): # Read all the channels
            succ = len(getattr(self, chan)) == self.length
            successlist.append(succ)
            if not succ:
                print('Channel {} had a different length of {} than the _ids vector.'.format(chan,len(getattr(self, chan)),self.length))
            
        print('Check complete')
        return not np.sum(np.logical_not(successlist))
    
    def run_of_id(self,ID):
        run = [rn for rn in self._loaded_runs_intervals \
                   if within(ID, self._loaded_runs_intervals[rn])]
        if len(run)>1:
            raise ValueError('More than one matching run found!')
        return run[0]
    
    
    def eval_diode_matrix(self, diode_matrix, N_peaks, ADC_params, plot = False):
        """eval_diode_matrix(diode_matrix, N_peaks, ADC_params, plot = False)
        An indexing-based function to extract peak pulse energies from a 
        Matrix of diode-traces. Runs much faster than the loop-based alternative.
        The diode_matrix should have the shape (No_Events,len(ADC_trace)).
        ADC_params should be a dictionary with the entries (all int):
        "first_peak_start","peak_distance","peak_width","pre_bkg_width"

        Returns a matrix "peak_energies" of the shape (No_events, No_pulses)
        
        This function is implemented as a method to FLASHmes although it does not use
        the object by itself, simply because it is more practical to import it this way.
        """
        first_peak_start = ADC_params['first_peak_start'] 
        peak_distance = ADC_params["peak_distance"]
        peak_width = ADC_params["peak_width"]
        pre_bkg_width = ADC_params["pre_bkg_width"]

        # Boolean Index Matrix locating the relevant signals
        peak_indexmat = np.zeros(diode_matrix.shape,dtype = bool)
        for i in range(peak_width):
            peak_indexmat[:,first_peak_start+i:i+first_peak_start+N_peaks*peak_distance:peak_distance] = True

        # This is the just the relevant signal
        condensed_diode_signal = np.reshape(diode_matrix[peak_indexmat],(diode_matrix.shape[0],-1))
        if peak_width*N_peaks != condensed_diode_signal.shape[1]:
            print("Warning: Could not extract as many peaks as given - check number of peaks!")

        # Integrating the relevant signal to the peak energy matrix
        peak_energies = np.zeros((diode_matrix.shape[0],N_peaks))
        for i in range(N_peaks):
            peak_energies[:,i] = np.nansum(condensed_diode_signal[:,i*peak_width:peak_width+i*peak_width],1)
        peak_energies[peak_energies==0]=np.nan
        if plot:
            plt.figure()
            plt.imshow(condensed_diode_signal, aspect = "auto")

            plt.figure()
            plt.imshow( peak_energies, aspect = "auto")

        return peak_energies

class loadedFLASHmes(FLASHmes):
    def __init__(self, loadfilename, skipchannels =[]):
        print('Loading from file: '.format(loadfilename))
        # Define Attributes
        thelib = dd.io.load(loadfilename)
        
        for key in thelib.keys():
           # print(f"Loading key: {key}.")
            if not key in skipchannels:
                setattr(self, key, thelib[key])
        
        print('Successfully loaded Runs {}'.format(self._loaded_runs))  
