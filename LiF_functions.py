from FLASHmes import FLASHmes, loadedFLASHmes
import numpy as np
import little_helpers as lh
from matplotlib import pyplot as plt
import warnings
def getmes(run, Npulses, channels,daqAccess):
    if type(run) is int:
        runNo = run
        mes = FLASHmes(daqAccess = daqAccess, runNo = runNo, channels = channels)
    elif str(type(run)) == "<class 'small_lib.FLASHmes'>":
        mes = run
    else:
        raise ValueError('Run must be a FLASHmes or run number integer.')
    
    
    
    added_channels = [key for key in channels.keys()]
    if 'diode' in added_channels:
        print('Performing trace peak integration...')
        ADC_params ={
        "first_peak_start": 857,
        "peak_distance": 1080,
        "peak_width" : 11,
        "pre_bkg_width": 200
        }

        mes.energies = mes.eval_diode_matrix(mes.diode, N_peaks=40, ADC_params=ADC_params)
        mes.eventfilter('good', np.isfinite(np.mean(mes.energies,1)))
        
    if 'mono' in added_channels:
        mes.eventfilter('good', np.isfinite(mes.mono))
    if 'delay' in added_channels:
        mes.eventfilter('good', np.isfinite(mes.delay))
    return mes

def get_meslist_nokwargs(runlist, bkg_run_no,calib, t0, beamblocked_pix, Npulses, channels,\
                binning,first_pix,last_pix, image_cutoff , zone_of_disbelief,daqAccess):

    
    #import runs from meslist
    mes = getmes(31000+runlist[0], Npulses, channels,daqAccess)
    for meas in runlist[1:]:
        mes.append_run(31000+meas)
    
    mes.good = np.any(mes.images,(1,2))

    print(f"All Runs read succesfully. Reading Background...")
    ## Define energy axis
    CCDbin=binning
    mes.enax = np.polyval(calib,np.arange(2048/CCDbin))
    
    # Set values
    mes.t0 = t0
    mes.beamblocked_pix = beamblocked_pix
    photenergies = lh.nm2eV(mes.mono)
    if np.nanstd(photenergies) > 0.05:
        raise ValueError(f'Sure that these runs belong together? Std(photon energies) = {np.nanstd(photenergies)}')
    mes.E_cent = np.nanmean(photenergies)
    mes.first_pix = first_pix
    mes.last_pix= last_pix
    mes.image_cutoff = image_cutoff
    mes.bkgRunNo = bkg_run_no

    mes.binning = binning
    if zone_of_disbelief is None:
        warnings.warn("Specify Zone of disbelief")
        zone_of_disbelief = [mes.E_cent-1,mes.E_cent+1]
    mes.zone_of_disbelief =  zone_of_disbelief
    
    mes.laser_waveplate_position = np.nanmean(mes.waveplate)
    #mes.laser_pulse_uJ    
    
    #Background
    bkgmes = getmes(bkg_run_no,Npulses,channels,daqAccess)
    print('Mask cosmics in Background Run')
    bkg_images, _, _ = lh.cosmics_masking(bkgmes.images, kernel_size = (4,2), Nsigma = 10,roi = np.s_[2:,:], plot = False)
    mes.bkgim = np.mean(bkg_images[bkgmes.good],0)
    
    print('Mask cosmics of actual_data Run')
    mes.images, mes.cosmic_excluded_region, mes.images_with_masked_cosmics =\
        lh.cosmics_masking(mes.images, kernel_size = (4,2), Nsigma = 10,roi = np.s_[3:,:], plot = True)
    mes.images = mes.images-mes.bkgim

    return mes


def set_delbins(mes, Ddel = 0.06, t0_margin = 0.25, outputfile = None):
    mes.Ddel = Ddel
    mes.delbin_edg = np.arange(np.nanmin(mes.delay[mes.good])-Ddel/2,np.nanmax(mes.delay[mes.good])+1.01*Ddel,Ddel)
    mes.delbin_mids = lh.midpoints(mes.delbin_edg)
    mes.unpumped_events = (mes.delay>(mes.t0+t0_margin))|(mes.delay<(mes.t0-t0_margin))
    mes.delay = lh.interp_nans(mes.delay)

    plt.figure(figsize =(6,5))
    plt.plot(mes._ids[mes.good]-mes._ids[0],mes.delay[mes.good],'C1.-', lw = 0.2)
    plt.plot(mes._ids[mes.good&mes.unpumped_events]-mes._ids[0],mes.delay[mes.good&mes.unpumped_events],'C0.')

    for dd in mes.delbin_edg:
        plt.axhline(dd)
    plt.ylabel('Delay / ps')
    plt.xlabel('Event ID of Spectra')
    plt.tight_layout()
    
    if outputfile is not None:
        plt.savefig(outputfile, format='pdf')
    
def normalize_spectra(mes, beamblocked_pix= None, first_pix = None,last_pix= None,  image_cutoff = None, outputfile = None):
    """
    returns event-based normalized spectra
    """
    if beamblocked_pix is None:
        beamblocked_pix = mes.beamblocked_pix
    else:
        mes.beamblocked_pix = beamblocked_pix
    if first_pix is None:
        first_pix = mes.first_pix
    else:
        mes.first_pix= first_pix
    if last_pix is None:
        last_pix = mes.last_pix
    else:
        mes.last_pix=last_pix
    if image_cutoff is not None:
        mes.image_cutoff = image_cutoff
        
    mes.De = np.mean(mes.enax[1:]-mes.enax[:-1]) #needed later

    mes.avgim_pumped = np.mean(mes.images[mes.good&np.logical_not(mes.unpumped_events)],0)
    B = mes.avgim_pumped.shape[1]
    plt.figure()
    plt.plot(range(B),np.sum(mes.avgim_pumped,0),'-', label = 'all')
    plt.plot(np.arange(B)[mes.image_cutoff],np.sum(mes.avgim_pumped,0)[mes.image_cutoff],'.-',label = 'used')
    plt.legend()

        
    mes.Nrows = mes.avgim_pumped[:,mes.image_cutoff].shape[1] 
    
    mes.image_offsets = np.mean(mes.images[:,beamblocked_pix[0]:beamblocked_pix[1],mes.image_cutoff],\
                                (1,2))*mes.Nrows# In the beam block shadow

    mes.all_spectra_raw = (np.sum(mes.images[:,:,mes.image_cutoff],2).T-mes.image_offsets.T).T /mes.binning
    mes.all_spectra_raw[:,0] = 0 # Setting that buggy first line of pixels to 0
    mes.all_spectra_raw = (mes.all_spectra_raw.T/lh.interp_nans(mes.gmd_hall)).T # Normalize by hall gmd
    
    mes.stray_high = np.sum(mes.all_spectra_raw[:,last_pix[0]:last_pix[1]],1) # First pixels
    mes.stray_low = np.sum(mes.all_spectra_raw[:,first_pix[0]:first_pix[1]],1) # Last Pixels
    mes.straylight = np.mean((mes.stray_low.T,mes.stray_high.T),0) # Average first and last
    mes.straylight = mes.straylight/np.mean(mes.straylight[mes.good])
    
    if outputfile is not None:
        plt.savefig(outputfile, format='pdf')

    ############## Event filtering here
    
    plt.figure()
    #plt.title('Event Filtering')
    ### High stry light region
    plt.plot(np.arange(len(mes._ids))[mes.good],mes.stray_high[mes.good],'C0.', label ='stry high')

    average_stray_high = lh.smooth(mes.stray_high[mes.good],50)
    plt.plot(np.arange(len(mes._ids))[mes.good],average_stray_high,'-',c='blue')

    sigma_stray_high = np.nanstd(mes.stray_high[mes.good]- average_stray_high)
    stray_high_too_low = mes.stray_high[mes.good]< average_stray_high-2*sigma_stray_high#0.7*average_stray_high
    stray_high_too_high = mes.stray_high[mes.good]> average_stray_high+2*sigma_stray_high#1.3*average_stray_high
    stray_high_bad = stray_high_too_low | stray_high_too_high
    plt.plot(np.arange(len(mes._ids))[mes.good][stray_high_bad],mes.stray_high[mes.good][stray_high_bad],'x',c='blue')


    ### Low stary light region
    plt.plot(np.arange(len(mes._ids))[mes.good],mes.stray_low[mes.good],'C1.', label ='stry low')

    average_stray_low = lh.smooth(mes.stray_low[mes.good],50)
    plt.plot(np.arange(len(mes._ids))[mes.good],average_stray_low,'-',c='red')

    sigma_stray_low = np.nanstd(mes.stray_low[mes.good]- average_stray_low)
    stray_low_too_low = mes.stray_low[mes.good]< average_stray_low-2*sigma_stray_low#0.7*average_stray_low
    stray_low_too_high = mes.stray_low[mes.good]> average_stray_low+2*sigma_stray_low#1.3*average_stray_low
    stray_low_bad = stray_low_too_high | stray_low_too_low
    plt.plot(np.arange(len(mes._ids))[mes.good][stray_low_bad],mes.stray_low[mes.good][stray_low_bad],'x',c='red')


    ### Offsets region
    offsets_variation = mes.image_offsets-np.nanmean(mes.image_offsets[mes.good])
    plt.plot(np.arange(len(mes._ids))[mes.good],offsets_variation[mes.good]-10,'C2.', label ='stray offset')

    average_offset_variation = lh.smooth(offsets_variation[mes.good],50)
    plt.plot(np.arange(len(mes._ids))[mes.good],average_offset_variation-10,'-',c='green')
    
    offset_sigma = np.nanstd(offsets_variation[mes.good]-average_offset_variation)
    offset_variation_too_low = offsets_variation[mes.good]< average_offset_variation-2*offset_sigma#3.7*mes.binning
    offset_variation_too_high = offsets_variation[mes.good]> average_offset_variation+2*offset_sigma#3.7*mes.binning
    offset_variation_bad = offset_variation_too_low | offset_variation_too_high

    plt.plot(np.arange(len(mes._ids))[mes.good][offset_variation_bad],offsets_variation[mes.good][offset_variation_bad]-10,'x',c='green')
    
    temp = mes.good.copy()

    temp[np.arange(len(mes._ids))[mes.good][offset_variation_bad]] = False
    temp[np.arange(len(mes._ids))[mes.good][stray_high_bad]] = False
    temp[np.arange(len(mes._ids))[mes.good][stray_low_bad]] = False

    mes.good = temp
    
    print(f'{200* np.sum(mes.good)/len(mes.good)}% of camera on events were classified as good.')
    plt.title('Event Filtering:'+f'{200* np.sum(mes.good)/len(mes.good):.2f}% of camera on events were classified as good.')
    plt.legend()
    #### Event filtering complete
    
    
    mes.ccd_noise_pix = np.std(mes.images[mes.good][:,beamblocked_pix[0]:beamblocked_pix[1],mes.image_cutoff])# RMS Noise each pixel
    mes.ccd_noise_spec_pix = mes.ccd_noise_pix/np.sqrt(mes.images[0,0,mes.image_cutoff].shape[0])# RMS Noise per pixel in spectrum
    
    #avgspec_raw = np.mean(mes.all_spectra_raw[mes.good&mes.unpumped_events],0)

    mes.all_spectra = ((mes.all_spectra_raw).T/mes.straylight.T).T # Normalize by stray light
        

    # Error derived from the average CCD noise and scaled as the signal was
    mes.all_spectra_err = mes.ccd_noise_spec_pix*(np.ones(mes.all_spectra.shape).T/mes.straylight.T).T
        
    mes.avgspec = np.mean(mes.all_spectra[mes.good&mes.unpumped_events],0)
    mes.avgspec_err = np.std(mes.all_spectra[mes.good&mes.unpumped_events],0)/np.sqrt(np.sum(mes.good&mes.unpumped_events))
    
    mes.avgspec_pumped = np.mean(mes.all_spectra[mes.good&np.logical_not(mes.unpumped_events)],0)
    mes.avgspec_pumped_err = np.std(mes.all_spectra[mes.good&np.logical_not(mes.unpumped_events)],0)\
                /np.sqrt(np.sum(mes.good&np.logical_not(mes.unpumped_events)))

    def difference_with_error(x,y,xerr,yerr):
        diff = x-y
        comberr = np.sqrt((xerr**2) + (yerr**2))
        return diff, comberr
    
    mes.all_spectra_ppdiff,mes.all_spectra_ppdiff_err =  \
            difference_with_error(mes.all_spectra,mes.avgspec,mes.all_spectra_err,mes.avgspec_err)
    
    if outputfile is not None:
        plt.savefig(outputfile, format='pdf')

    plt.figure()
    
    # Over Energy
    plt.plot(mes.enax,mes.avgspec)
    plt.axvline(mes.E_cent, ls='-')
    for e in [-3.1,-1.55,1.55,3.1]:
        plt.axvline(mes.E_cent+e, ls=':')
        
    # Over pixels
    ax2 = plt.gca().twiny()
    plt.sca(ax2)
    plt.plot(mes.avgspec,'-.')
    
    L = len(mes.avgspec)
    plt.axvspan(*beamblocked_pix, alpha = .5, color = 'grey', label = 'Offset')
    plt.axvspan(first_pix[0],first_pix[1], alpha = .5, color = 'black', label = 'scaling')
    plt.axvspan(last_pix[0],last_pix[1], alpha = .5, color = 'black')
    plt.legend()
    
    if outputfile is not None:
        plt.savefig(outputfile, format='pdf')