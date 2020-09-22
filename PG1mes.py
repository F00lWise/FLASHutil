## Actually for this beamtime: /asap3/flash/gpfs/pg1/2020/data/11008872/raw/hdf/online-0/
#hdf_path = "/asap3/flash/gpfs/pg1/2019/data/11006938/raw/hdf/online-0/"
hdf_path = "/asap3/flash/gpfs/pg1/2020/data/11008872/raw/hdf/online-0/"

daqAccess= BeamtimeDaqAccess.create(hdf_path)


save_spectra_path = '../spectra/'

channels = {
    'images' : '/FL1/Experiment/Camera/Multiplex/Pixis 1/image',
    #'diode1' : "/FL1/Experiment/PG/SIS8300 100MHz ADC/CH6/TD",
    #'diode1_energies' : "/FL1/Experiment/PG/SIS8300 100MHz ADC/CH6/pulse energy/TD",
    #'diode2' : "/FL1/Experiment/PG/SIS8300 100MHz ADC/CH7/pulse energy/TD ",
    'diode' :  "/FL1/Experiment/PG/SIS8300 100MHz ADC/CH7/TD",

    #'shutter' : "/FL1/Beamlines/PG/Fast shutter/shutter",
    'mono' : "/FL1/Beamlines/PG/Monochromator/monochromator photon energy",
    'delay_raw' : '/FL1/Experiment/Pump probe laser/delay line IK220.0/ENC'
    }

class PG1mes(FLASHmes):
    def __init__(self,runNo, alternative_readin = False, altern_time = 1800):
        FLASHmes.__init__(self, daqAccess, runNo, channels, alternative_readin, altern_time)
        
        
class PG1evaluator():
    """
    This class contains the partly Beamtime-specific evaluation code, but no data.
    All data ae contained in an instance of  PG1mes, which is passed to this class upon initialization.
    
    """
    def __init__(self, mes, delmargin = 0.08,t0=0,bkg = None,
                 daqAccess = daqAccess,channels= channels, alternative_readin=False, 
                 N_peaks = None, ADC_params=ADC_params):

        # Split into data class and eval class.
        self.mes = mes
        
        # Load background data if given as run number
        self.mes.delay_margin = delmargin
        
        self.mes.bkg_subtracted = bkg is not None
        if bkg is None:
            bkg = np.zeros((self.mes.images.shape[1],self.mes.images.shape[2]))
            
        self.mes.bkgim = bkg
        self.mes.delay = self.mes.delay_raw[:,1]    
        
        #self.mes.good = np.ones(self.mes._ids.shape[0],dtype=bool)
        self.mes.good = np.any(self.mes.images,(1,2))
        def exclude_images_after_moving(self):
            # Looking for camera images that might have averaged over a moving delay stage
            first = True
            last_delay = np.nan
            excluded_counter = 0
            
            
            for i in range(self.mes._length):
                if self.mes.good[i]:
                    if first:
                        first = False
                        continue
                    else:
                        this_delay = self.mes.delay[i]
                        moved = np.abs(this_delay - last_delay)>self.mes.delay_margin
                        if moved:
                            self.mes.good[i] = False
                            excluded_counter +=1
                        last_delay = this_delay
            #print(f'Excluded {excluded_counter} images because the delay stage had moved.')
        exclude_images_after_moving(self)
        
        self.mes.t0 = t0

        #self.mes.delay = self.mes.delay_raw[:,1]

        self.mes.artifacts_removed = False
    # section dedicated to find the right delays (used to be 2)
        delay_vector = lh.interp_nans(self.mes.delay)
        
        diffdelay = np.diff(delay_vector)
        self.mes.good[1:][np.abs(diffdelay)>0.01] = False  # Exclude Images where delay stage was moving
        delay_vector= delay_vector[self.mes.good]#
        
        
        self.mes.static_in_delayhopping = False
        if N_peaks is not None: # If laser off images are included
            print("Extraction diode data.")
            self.mes.static_in_delayhopping = True

            # Find Out where the laser was on
            self.mes.diode_energies = self.mes.eval_diode_matrix(self.mes.diode, N_peaks, ADC_params)
            diode_integral = lh.interp_nans(np.sum(self.mes.diode_energies,1))
            low_diode = np.min(diode_integral)
            high_diode = np.max(diode_integral)
            self.mes.lason = diode_integral > (low_diode+high_diode)/2
                
        """ #Original find delays function:        
        def find_delays(delay_vector, rec_counter, min_distance  = delmargin):
            # This function recursively finds all peaks in the histogram of
            # a delay trace that are further than min_distance apart
            totmean = np.mean(delay_vector)

            subset_low = delay_vector[delay_vector<totmean]
            subset_high = delay_vector[delay_vector>totmean]

            low_delay = np.mean(subset_low)
            high_delay = np.mean(subset_high)
            if np.abs(low_delay-high_delay)<min_distance:
                delay_list = [np.mean([low_delay,high_delay])]
            else:
                low_delaylist = find_delays(subset_low, min_distance=min_distance,rec_counter=rec_counter+1)
                high_delaylist = find_delays(subset_high, min_distance=min_distance,rec_counter=rec_counter+1)
                delay_list = np.concatenate((low_delaylist,high_delaylist))

            return delay_list
        
         
        self.mes.delayvals = find_delays(delay_vector,min_distance=delmargin, rec_counter =1)

        
        self.mes.belongs_to_delay = np.zeros((len(self.mes.delayvals), len(self.mes._ids)))
        for i, delval in enumerate(self.mes.delayvals):
            self.mes.belongs_to_delay[i,:] = lh.simeq(self.mes.delay,self.mes.delayvals[i],self.mes.delay_margin)
        
        self.update_parameters()
        """
        
        
        def find_delays(delay_vector,delmargin=delmargin):
            counts, delay_edges = np.histogram(delay_vector, np.arange(np.nanmin(delay_vector),np.nanmax(delay_vector),delmargin/2))
            delvals = []
            for i in range(len(counts)):
                if counts[i]>5:
                    delvals.append(lh.midpoints(delay_edges)[i])
            return np.sort(np.array(delvals))
        
        self.mes.delayvals = find_delays(delay_vector)#,min_distance=delmargin, rec_counter =1)

        #self.mes.delay_margin = delmargin
        
        self.mes.belongs_to_delay = np.zeros((len(self.mes.delayvals), len(self.mes._ids)))
        for i, delval in enumerate(self.mes.delayvals):
            self.mes.belongs_to_delay[i,:] = lh.simeq(self.mes.delay,self.mes.delayvals[i],self.mes.delay_margin)
        
        self.update_parameters()
        
    
    def update_parameters(self):
        # Require valid data
        #self.mes.eventfilter('good' ,self.mes.images[:,5,5]>0) #ugly way
        print(f'Updating Parameters.\n Loaded Runs: {self.mes._loaded_runs}')
        self.mes.eventfilter('good' ,np.any(self.mes.images,(1,2))) #better
        
        
        print(f'Found {np.sum(self.mes.good)} good images')
        #self.mes.eventfilter('good',np.isfinite(self.mes.mono))
        #self.eventfilter('good',np.isfinite(self.delay))
        if self.mes.static_in_delayhopping:
            self.mes.lasongood = self.mes.lason &self.mes.good
            self.mes.lasoffgood = (~ self.mes.lason) &self.mes.good
            print(f'Of the Good images {np.sum(self.mes.lasongood)} were taken with laser on, {np.sum(self.mes.lasoffgood)} were taken with laser off,')

        print(f'Found {np.sum(self.mes.good)} good events overall')
        
        
        # Subtract bkg
        self.mes.avgim = np.subtract(np.mean(self.mes.images[self.mes.good],0),self.mes.bkgim)
        
        self.mes.avgim_at_del = np.zeros((len(self.mes.delayvals),self.mes.images.shape[1],self.mes.images.shape[2]))
        self.mes.images_per_delay = np.zeros(len(self.mes.delayvals))
        
        self.mes.events_at_delay = np.zeros((self.mes._length,len(self.mes.delayvals)), dtype = bool)
        for i, delval in enumerate(self.mes.delayvals):
            associated_events = np.array(self.mes.belongs_to_delay[i,:], dtype=bool)
            if self.mes.static_in_delayhopping:
                self.mes.events_at_delay[:,i] = self.mes.lasongood&associated_events
                self.mes.avgim_at_del[i,...] = np.mean(self.mes.images[self.mes.lasongood&associated_events],0)-self.mes.bkgim      
                self.mes.images_per_delay[i] = np.sum(self.mes.lasongood&associated_events)
            else:
                self.mes.events_at_delay[:,i] = self.mes.good&associated_events
                self.mes.avgim_at_del[i,...] = np.mean(self.mes.images[self.mes.good&associated_events],0)-self.mes.bkgim      
                self.mes.images_per_delay[i] = np.sum(self.mes.good&associated_events)

        if self.mes.static_in_delayhopping:
            self.mes.avgim_lasoff = np.mean(self.mes.images[self.mes.lasoffgood],0)

        self.mes.gimages = self.mes.images[self.mes.good]


    def make_spectrum(self, slope = 14.0e-3, correct_jitter = False, jitter_precision = 5, energy_calibration = True, save = False, plot = False):
        self.mes.spectrum_raw =  np.mean(self.mes.avgim[:,:],1)
        self.mes.spectrum_bkg =  np.mean(self.mes.bkgim[:,:],1)
        if self.mes.static_in_delayhopping:
            if not correct_jitter:
                self.mes.spectrum_lasoff = np.mean(self.mes.avgim_lasoff,1)
            else:
                self.mes.spectrum_lasoff,_ =spec_correcting_pointing_jitter(self.mes.images[self.mes.lasoffgood]-self.mes.bkgim, precision = jitter_precision)
                print('Laser off spectrum calculated with jitter correction.')
        self.mes.spectrum_raw_at_del = np.zeros((len(self.mes.delayvals),self.mes.avgim.shape[0]))
        for i, delval in enumerate(self.mes.delayvals):
            if not correct_jitter:
                self.mes.spectrum_raw_at_del[i,...] = np.mean(self.mes.avgim_at_del[i,:,:],1)
            else:
                if not self.mes.static_in_delayhopping:
                    raise ValueError('Alignment not implemented without static in delayhopping')
                else:
                    self.mes.spectrum_raw_at_del[i,...],_ = \
                        spec_correcting_pointing_jitter_aligned(self.mes.images[self.mes.events_at_delay[:,i]]-self.mes.bkgim,\
                                                                self.mes.spectrum_lasoff, precision = jitter_precision)
                    print(f'Delay {delval:.2f} spectrum calculated with jitter correction.')

        self.mes.E = np.nanmean(self.mes.mono)
        self.mes.energy_calibration = slope
          
        self.mes.peakindex = np.argmax(self.mes.spectrum_raw[::-1])
        self.mes.enax_around_max = (np.arange(self.mes.spectrum_raw.shape[0],0,-1)-self.mes.peakindex)*self.mes.energy_calibration

        
        # Calculate spectral weight
        elastic_specrange = (self.mes.enax_around_max<0.2)&(self.mes.enax_around_max>-0.7)
        specweight = lh.spectral_weight(self.mes.enax_around_max[elastic_specrange], self.mes.spectrum_raw[elastic_specrange])
        self.mes.enax = self.mes.enax_around_max - specweight
        if plot:
            plt.figure(figsize = (8,5))
            plt.semilogy(self.mes.enax,self.mes.spectrum_raw,"-.", label = "run 781, old 500 um sample")
            plt.axvline(self.mes.enax_around_max[2048-self.mes.peakindex], c='C0', label = 'Old definition of zero energy loss')
            plt.axvspan(self.mes.enax_around_max[np.where(elastic_specrange==1)[0][0]],self.mes.enax_around_max[np.where(elastic_specrange==1)[0][-1]],\
                        label = 'area in which spectral weight is evaluated', color='g', alpha = 0.5)
            plt.axvline(0,  c= 'C3',label = 'new definition of zero energy loss')
            plt.legend()
        
        if save:
            fname = save_spectra_path+f'Run_{self.mes._loaded_runs}_Raw_Avg_Spec_{datetime.now().strftime("%d-%m-%Y_%H.%M")}.txt'
            comment = f'Background subtracted:'+str(self.mes.bkg_subtracted)
            dicti = {'E': self.mes.enax, 'I': self.mes.spectrum_raw}
            #print(dicti)
            lh.save_dict_to_txt(fname,dicti, header = comment)
            print('Saved as ',fname)

        
        return self.mes.enax, self.mes.spectrum_raw
    
    def save_spectra(self):
            fname = save_spectra_path+f'Run_{self.mes._loaded_runs}_Raw_Avg_Spec_{datetime.now().strftime("%d-%m-%Y_%H.%M")}.txt'
            comment = f'Background subtracted:'+str(self.mes.bkg_subtracted)
            dicti = {'E': self.mes.enax, 'I': self.mes.spectrum_raw}
            #print(dicti)
            lh.save_dict_to_txt(fname,dicti, header = comment)
            
            for i, delay in enumerate(self.mes.delayvals):
                fname = save_spectra_path+'/time_resolved/'+f'Run_{self.mes._loaded_runs}_Delay{delay:.2f}ps_Day-{datetime.now().strftime("%d-%m")}.txt'
                comment = f'Background subtracted:'+str(self.mes.bkg_subtracted)+' Delay: '+ str(delay)
                dicti = {'E': self.mes.enax, 'I': self.mes.spectrum_raw_at_del[i]}
                #print(dicti)
                lh.save_dict_to_txt(fname,dicti, header = comment)

            
            print('Saved as ',fname)
    
    def check_sample_damage(self, pixel_interval = (800,1000)):
        plt.figure()
        plt.title("Average Spectrum")
        plt.xlabel("Pixel")
        plt.ylabel("Raw Intensity (summed)")
        plt.plot(np.sum(self.mes.avgim,1))
        plt.axvspan(pixel_interval[0],pixel_interval[1], color = 'r', alpha = 0.4)

        plt.figure()
        plt.title("Intensity over Time")
        
        trace = np.mean(self.mes.images[self.mes.good][:,pixel_interval[0]:pixel_interval[1],:],axis=(1,2))
        plt.plot(self.mes._ids[self.mes.good], trace, '.-')
        
        return trace
    
    def axislabels(self, legendfontsize = 12):
        plt.xlabel('Energy loss / eV')
        plt.ylabel('Intensity')
        plt.legend()
        plt.xlim(10,-5)
        plt.tight_layout()
        
    def normalize(self, enax, spectrum, norminterval = [2,-1],bkginterval = [-20,-10]):
        """
        Norminterval and bkginterval should each be tuples of values in eV, relative to the peak position
        The spectrum will be normalized such that
        bkginterval will be normalized to the value zero, norminterval to one.
        """
        norminterval = np.sort(norminterval)
        bkginterval = np.sort(bkginterval)

        normval = np.mean(spectrum[(enax>norminterval[0])&(enax<norminterval[1])])
        bkgval = np.mean(spectrum[(enax>bkginterval[0])&(enax<bkginterval[1])])
        
        scale = normval-bkgval
        
        spectrum_norm = spectrum - bkgval
        spectrum_norm = spectrum_norm / scale
        return spectrum_norm
    
    def plot_delhopping(self, smooth = None,\
                        return_spectra = False, offset = 0,remove_offset=True,\
                        norminterval = None ,bkginterval = [-20,-10], **plotkwargs):
        if not 'spectrum_raw_del1' in dir(self):
            self.make_spectrum()
            
        
        plotkwargs_original = deepcopy(plotkwargs)
        n_delays = len(self.mes.delayvals)+1
        color_idx = np.linspace(0,1,n_delays)
        
        for i, delval in enumerate(self.mes.delayvals):
            # Define what to plot
            plotkwargs = plotkwargs_original.copy()
            enax = self.mes.enax
            spectrum = self.mes.spectrum_raw_at_del[i]
            effdelay = delval-self.mes.t0
            N_images = self.mes.images_per_delay[i]
            
            # Normalize
            if norminterval is not None:
                spectrum = self.normalize(enax, spectrum, norminterval,bkginterval)
            elif remove_offset:
                bkginterval = np.sort(bkginterval)
                bkg =    np.mean(spectrum[(enax>bkginterval[0])&(enax<bkginterval[1])])
                #print(f'From Spectrum at delay {delval:.2f} Removeing offset of {bkg}.',)
                spectrum = spectrum - bkg
            # Smooth
            if smooth is not None:
                spectrum = scipy.ndimage.gaussian_filter(spectrum, sigma = smooth)

            # Plot options
            if not 'color' in plotkwargs.keys():
                #plotkwargs['color'] = f'C{i}'
                plotkwargs['color'] = plt.cm.jet(color_idx[i])
            if not 'label' in plotkwargs.keys():
                plotkwargs['label'] = f'Runs {self.mes._loaded_runs}, E={self.mes.E:.1f}eV, t={effdelay:.2f}ps, {N_images:.0f} images'     
            else:
                plotkwargs['label'] = plotkwargs['label'] +f't={effdelay:.2f}ps, {N_images:.0f} images'         
            
            if not 'linestyle' in plotkwargs.keys():
                if effdelay < 0:
                    plotkwargs['linestyle'] = '--'
                else:
                    plotkwargs['linestyle'] = '-'
            
            # Finally Plot
            plt.plot(enax,offset+spectrum,  **plotkwargs)
            
            
        ## And the same for the Laser Off spectrum
        
        if self.mes.static_in_delayhopping:
            # Define what to plot
            plotkwargs = plotkwargs_original.copy()
            enax = self.mes.enax
            spectrum = self.mes.spectrum_lasoff
            N_images = np.sum(self.mes.lasoffgood)
            # Normalize
            if norminterval is not None:
                spectrum = self.normalize(enax, spectrum, norminterval,bkginterval)
            elif remove_offset:
                bkg = np.mean(spectrum[(enax>bkginterval[0])&(enax<bkginterval[1])])
                spectrum = spectrum - bkg
            # Smooth
            if smooth is not None:
                spectrum = scipy.ndimage.gaussian_filter(spectrum, sigma = smooth)

            # Plot options
            if not 'color' in plotkwargs.keys():
                #plotkwargs['color'] = f'C{i+1}'
                plotkwargs['color'] = 'k'
            if not 'label' in plotkwargs.keys():
                plotkwargs['label'] = f'Runs {self.mes._loaded_runs}, E={self.mes.E:.1f}eV, Static (Laser Off), {N_images:.0f} images'     
            else:
                plotkwargs['label'] = plotkwargs['label'] +f'static, {N_images:.0f} images'         

            if not 'linestyle' in plotkwargs.keys():
                plotkwargs['linestyle'] = ':'

            # Finally Plot
            plt.plot(enax,offset+spectrum,  **plotkwargs)

        self.axislabels()
        
    def timeseries(self,interval,remove_offset=True, norminterval = None,bkginterval=[-5,-10], remove_DC = True,**plotkwargs):
        interval= np.sort(interval)
        bkginterval= np.sort(bkginterval)
        time_axis  = self.mes.delayvals - self.mes.t0
        delay_trace = np.zeros(time_axis.shape)
        enax  = self.mes.enax
        
        for i, delay in enumerate(delay_trace):
            spectrum = lh.interp_nans(self.mes.spectrum_raw_at_del[i])
            # Normalize
            if norminterval is not None:
                spectrum = self.normalize(enax, spectrum, norminterval,bkginterval)
            elif remove_offset:
                bkg = np.mean(spectrum[(enax>bkginterval[0])&(enax<bkginterval[1])])
                spectrum = spectrum - bkg
            delay_trace[i] = np.mean(spectrum[(enax>interval[0])&(enax<interval[1])])
        
        if remove_DC:
            delay_trace /=np.mean(delay_trace[self.mes.delayvals>0.1])
            plt.axhline(1,c='k',ls=':')
            
        plt.plot(time_axis, delay_trace, label = f'DE = {interval}',**plotkwargs)
        plt.xlabel('Delay (negative=pumped)/ ps')
        plt.ylabel('Spectral intensity / arb. u.')
        return time_axis,delay_trace
    
    def remove_artifacts(self, artifact_threshold = 100,artifact_min_width = 30, ignore_region = None, bkglevel=600, force= False):
        self.make_spectrum(slope = 31e-3, save = False)
        
        if self.mes.artifacts_removed and not force:
            print("Artifacts of this run were already subtracted - aborting")
        else:
            self.mes.images = np.array(self.mes.images,dtype=np.float32)

            if ignore_region is None:
                mn = 2048-(self.mes.peakindex) -10
                mx = 2048-self.mes.peakindex +40
                ignore_region = np.s_[mn:mx]  
            def find_image_artifacts(im, mes, ignore_region, bkglevel=bkglevel, artifact_threshold=artifact_threshold, artifact_min_width=artifact_min_width):
                imSM = scipy.ndimage.convolve(im, np.array([[1,1,1],[0,0,0],[1,1,1]]), mode='wrap')
                imSM[:,ignore_region] = np.nan
                artifact_locations = ((im-bkglevel)>artifact_threshold) & (((imSM/6)-bkglevel)<(artifact_threshold)) 
                located = scipy.ndimage.convolve(artifact_locations, np.ones((1,artifact_min_width)), mode='wrap')
                im[located>0]= np.nan
                return im
            

            for i, im in enumerate(self.mes.images):
                if self.mes.good[i]:
                    self.mes.images[i,...] = find_image_artifacts(np.array(self.mes.images[i,...] ,dtype=float), self.mes, ignore_region)

            self.update_parameters()
            print("Artifacts removed successfully.")
            self.mes.artifacts_removed = True
            
    def print_runinfo(self):
        print(f'Loaded Runs: {self.mes._loaded_runs}')
        print(f'Loaded Events: {self.mes._length}')
        print(f'Good images: {self.mes.gimages.shape[0]}')
        
        
    
def spec_correcting_pointing_jitter(gimages, precision = 5, plot = False):
    """
    arguments: 
        gimages - 3D set of good images
        precision - oversampling precision with whichthe cross correlation is calculated
        plot - option to plot each shift. Careful! Opens a plot for every image.
    
    Extracts the average spectrum from a stack of images while correcting for any possible pointing jitter.
    It does so by computing the cross-correlation function of the spectrum from each image with the average
    spectrum, which is iteratively calculated, beginning with the first.
    It returns the average spectrum and the calculated shifts for each spectrum."""
    all_spectra = np.array(np.mean(gimages,2),dtype = np.float32)
    avg_spectrum = np.array(all_spectra[0,:],dtype = np.float32)
    N=1
    shifts = np.zeros(all_spectra.shape[0])
    for i,spec in enumerate(all_spectra):
        shifts[i] , _, spec_shifted = lh.match_spectra(avg_spectrum,spec,\
                                 oversample = precision, return_shifted = True, return_oversampled = False, plot = plot)
        avg_spectrum *=N
        avg_spectrum +=spec_shifted
        avg_spectrum /=N+1
        N+=1
    print(shifts)
    return avg_spectrum, shifts

def spec_correcting_pointing_jitter_aligned(gimages, alignment_spectrum, precision = 5, plot = False):
    """
    arguments: 
        gimages - 3D set of good images
        precision - oversampling precision with whichthe cross correlation is calculated
        plot - option to plot each shift. Careful! Opens a plot for every image.
    
    Extracts the average spectrum from a stack of images while correcting for any possible pointing jitter.
    It does so by computing the cross-correlation function of the spectrum from each image with the average
    spectrum, which is iteratively calculated, beginning with the first.
    It returns the average spectrum and the calculated shifts for each spectrum."""
    
    all_spectra = np.array(np.mean(gimages,2),dtype = np.float32)
    avg_spectrum = np.zeros(all_spectra[0,:].shape, dtype = np.float32)
    N=0
    shifts = np.zeros(all_spectra.shape[0])
    for i,spec in enumerate(all_spectra):
        shifts[i] , _, spec_shifted = lh.match_spectra(alignment_spectrum,spec,\
                                 oversample = precision, return_shifted = True, return_oversampled = False, plot = plot)
        avg_spectrum *=N
        avg_spectrum +=spec_shifted
        avg_spectrum /=N+1
        N+=1

    print(shifts)
    return avg_spectrum, shifts



class loadedPG1evaluator:
    def __init__(self,mesList, delmargin = 0.05,static_in_delayhopping = True, t0=0):
        #self.delayvals = []
        self.spectrum_raw_at_del_dict = {} # These are dictionaries and that have the delayvals as keys
        self.images_per_delay_dict = {}    # are converted to arrays later
        self.spectrum_lasoff = None
        self.num_images_lasoff = None
        self._loaded_runs = []
        self.static_in_delayhopping = static_in_delayhopping
        self.enax = None
        self.t0 =0
        #self.E = None
        
        for mes in mesList: # For each laoded measurement
            print(f'Adding mes with runs {mes._loaded_runs}')
            for mesDelIndex, mesDelVal in enumerate(mes.delayvals): # For each delay in the loaded measurement
                # Combine Spectra at each delay
                #print(f'Delay {mesDelVal}')
                correspondingFound = False
                for specDelay in self.spectrum_raw_at_del_dict.keys(): # Compare if it fits to any existent delay
                    if np.abs(specDelay-mesDelVal)<delmargin: # if found, average them
                        correspondingFound = True
                        #print(f'Found corresponding delay: {mesDelVal} fits into {specDelay}')
                        
                        old_spectrum = self.spectrum_raw_at_del_dict[specDelay]
                        old_num_images = self.images_per_delay_dict[specDelay].copy()
                        
                        new_spectrum = mes.spectrum_raw_at_del[mesDelIndex]
                        num_additional_images = mes.images_per_delay[mesDelIndex]
                        
                        self.images_per_delay_dict[specDelay] = old_num_images+num_additional_images
                        self.spectrum_raw_at_del_dict[specDelay] = (old_spectrum*old_num_images/self.images_per_delay_dict[specDelay] )+ (new_spectrum*num_additional_images/self.images_per_delay_dict[specDelay])
                        break # only add it to one of course (this break should not be necessary but save a little time)
                if not correspondingFound:
                    specDelay = np.round(mesDelVal,2)
                    print(f'Adding Delay: {specDelay}')
                    self.images_per_delay_dict[specDelay] = mes.images_per_delay[mesDelIndex]
                    self.spectrum_raw_at_del_dict[specDelay] =mes.spectrum_raw_at_del[mesDelIndex]
                #print(f'Corresponding found: {correspondingFound}')
                
                # Combine Laser off spectra
                if self.spectrum_lasoff is None:
                    self.spectrum_lasoff = mes.spectrum_lasoff
                    self.num_images_lasoff = np.sum(mes.lasoffgood)
                else:
                    new_spectrum = mes.spectrum_lasoff
                    new_num_images = np.sum(mes.lasoffgood)

                    old_num_images = self.num_images_lasoff.copy()
                    
                    self.num_images_lasoff = np.sum(mes.lasoffgood) + old_num_images
                    self.spectrum_lasoff = (self.spectrum_lasoff*old_num_images/self.num_images_lasoff) +(new_spectrum*new_num_images/self.num_images_lasoff)
                
                # Combine loaded run numbers
                for runNo in mes._loaded_runs:
                    self._loaded_runs.append(runNo)
                #self.loaded_runs =self._loaded_runs + mes._loaded_runs
                
                # Combine Energy Axes
                if self.enax is None:
                    self.enax = mes.enax
                    #self.E = mes.E
                else:
                    difference = np.mean(self.enax-mes.enax)
                    if np.abs(difference)>0.1:
                        print(f"warning! Energy axis mismatch by {difference} (average)")
                    #if np.abs(self.E-mes.E)>0.1:
                    #    print(f"warning! FEL Energy  mismatch by {self.E-mes.E} (average)")
                
            self.delayvals = np.array(list(self.spectrum_raw_at_del_dict.keys()))
            self.spectrum_raw_at_del = np.empty((len(self.delayvals),2048))
            self.images_per_delay = np.empty(len(self.delayvals))
            for i, delayval in enumerate(self.delayvals):
                self.spectrum_raw_at_del[i] = self.spectrum_raw_at_del_dict[delayval]
                self.images_per_delay[i] = self.images_per_delay_dict[delayval]

            
    def timeseries(self,interval,remove_offset=True, norminterval = None,bkginterval=[-5,-10], remove_DC = True,**plotkwargs):
        interval= np.sort(interval)
        bkginterval= np.sort(bkginterval)
        time_axis  = self.delayvals - self.t0
        delay_trace = np.zeros(time_axis.shape)
        enax  = self.enax

        for i, delay in enumerate(delay_trace):
            spectrum = lh.interp_nans(self.spectrum_raw_at_del[i])
            # Normalize
            if norminterval is not None:
                spectrum = normalize(enax, spectrum, norminterval,bkginterval)
            elif remove_offset:
                bkg = np.mean(spectrum[(enax>bkginterval[0])&(enax<bkginterval[1])])
                spectrum = spectrum - bkg
            delay_trace[i] = np.mean(spectrum[(enax>interval[0])&(enax<interval[1])])

        if remove_DC:
            delay_trace /=np.mean(delay_trace[self.delayvals>0.1])
            plt.axhline(1,c='k',ls=':')

        plt.plot(time_axis, delay_trace, label = f'DE = {interval}',**plotkwargs)
        plt.xlabel('Delay (negative=pumped)/ ps')
        plt.ylabel('Spectral intensity / arb. u.')
        return time_axis,delay_trace

    def plot_delhopping(self, smooth = None,\
                        return_spectra = False, offset = 0,remove_offset=True,\
                        norminterval = None ,bkginterval = [-20,-10], **plotkwargs):
        #remind robin to change the following line in the PG1evaluator
        if not 'spectrum_raw_at_del' in dir(self):
            self.make_spectrum()


        plotkwargs_original = deepcopy(plotkwargs)
        n_delays = len(self.delayvals)+1
        color_idx = np.linspace(0,1,n_delays)

        for i, delval in enumerate(self.delayvals):
            # Define what to plot
            plotkwargs = plotkwargs_original.copy()
            enax = self.enax
            spectrum = self.spectrum_raw_at_del[i]
            effdelay = delval-self.t0
            N_images = self.images_per_delay[i]

            # Normalize
            if norminterval is not None:
                spectrum = normalize(enax, spectrum, norminterval,bkginterval)
            elif remove_offset:
                bkginterval = np.sort(bkginterval)
                bkg =    np.mean(spectrum[(enax>bkginterval[0])&(enax<bkginterval[1])])
                spectrum = spectrum - bkg
            # Smooth
            if smooth is not None:
                spectrum = scipy.ndimage.gaussian_filter(spectrum, sigma = smooth)

            # Plot options
            if not 'color' in plotkwargs.keys():
                #plotkwargs['color'] = f'C{i}'
                plotkwargs['color'] = plt.cm.jet(color_idx[i])
            if not 'label' in plotkwargs.keys():
                plotkwargs['label'] = f'Runs {self._loaded_runs}, t={effdelay:.2f}ps, {N_images:.0f} images'     
            else:
                plotkwargs['label'] = plotkwargs['label'] +f't={effdelay:.2f}ps, {N_images:.0f} images'         

            if not 'linestyle' in plotkwargs.keys():
                if effdelay < 0:
                    plotkwargs['linestyle'] = '--'
                else:
                    plotkwargs['linestyle'] = '-'

            # Finally Plot
            plt.plot(enax,offset+spectrum,  **plotkwargs)


        ## And the same for the Laser Off spectrum

        if self.static_in_delayhopping:
            # Define what to plot
            plotkwargs = plotkwargs_original.copy()
            enax = self.enax
            spectrum = self.spectrum_lasoff
            N_images = self.num_images_lasoff
            # Normalize
            if norminterval is not None:
                spectrum = normalize(enax, spectrum, norminterval,bkginterval)
            elif remove_offset:
                bkg = np.mean(spectrum[(enax>bkginterval[0])&(enax<bkginterval[1])])
                spectrum = spectrum - bkg
            # Smooth
            if smooth is not None:
                spectrum = scipy.ndimage.gaussian_filter(spectrum, sigma = smooth)

            # Plot options
            if not 'color' in plotkwargs.keys():
                #plotkwargs['color'] = f'C{i+1}'
                plotkwargs['color'] = 'k'
            if not 'label' in plotkwargs.keys():
                plotkwargs['label'] = f'Runs {self._loaded_runs}, Static (Laser Off), {N_images:.0f} images'     
            else:
                plotkwargs['label'] = plotkwargs['label'] +f'static, {N_images:.0f} images'         

            if not 'linestyle' in plotkwargs.keys():
                plotkwargs['linestyle'] = ':'

            # Finally Plot
            plt.plot(enax,offset+spectrum,  **plotkwargs)
            plt.xlabel('Energy loss / eV')
            plt.ylabel('Intensity')
            plt.legend()
            plt.xlim(10,-5)
            plt.tight_layout()

def normalize(enax, spectrum, norminterval = [2,-1],bkginterval = [-20,-10]):
    """
    Norminterval and bkginterval should each be tuples of values in eV, relative to the peak position
    The spectrum will be normalized such that
    bkginterval will be normalized to the value zero, norminterval to one.
    """
    norminterval = np.sort(norminterval)
    bkginterval = np.sort(bkginterval)

    normval = np.mean(spectrum[(enax>norminterval[0])&(enax<norminterval[1])])
    bkgval = np.mean(spectrum[(enax>bkginterval[0])&(enax<bkginterval[1])])

    scale = normval-bkgval

    spectrum_norm = spectrum - bkgval
    spectrum_norm = spectrum_norm / scale
    return spectrum_norm
