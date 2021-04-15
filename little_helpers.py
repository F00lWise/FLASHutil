import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy as sc
import scipy.ndimage
import scipy.signal
from scipy.signal import butter, filtfilt, freqz
import astropy.convolution
import copy


def strip(lst):
    """Takes a list or tuple with just one value.
     Returns the value, even if packed in multpiple single-tuples or lists.
     E.g.:
     lst = [([4,],),]
     strip(lst)
     >> 4"""
    if len(lst)< 1:
        return np.nan
    assert len(lst)==1, 'strip function requires single values in all sublists/tuples'
    content = lst[0]
    if type(content) in (list,tuple):
        content = strip(content)
    return content

def midpoints(x):
    """return (x[1:] + x[:-1])/ 2
    Returns the middle values between the input vector valus.
    This also means the langth of the output vector is one less."""
    return (x[1:] + x[:-1])/ 2
def edgepoints(middles):
    """ Opposite of midpoints """
    edges = np.empty(middles.shape[0]+1)
    edges[1:-1] = midpoints(middles)
    edges[0]= middles[0]-(middles[1]-middles[0])/2
    edges[-1]= middles[-1]+(middles[-1]-middles[-2])/2
    return edges

def normmax(X):
    """ Normalize X between Min and Max"""
    X = X-np.nanmin(X)
    return X/np.nanmax(X)
def normsum(X):
    """ Normalize X between Min and Sum"""
    X = X-np.nanmin(X)
    return X/np.nansum(X)

def gaussian_topnmorm(x, mu, sig):
    """ Gaussian, normalized to peak"""
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def gaussian(x, mu, sig):
    """ Gaussian, normalized to integral"""
    return (1/np.sqrt(2*np.pi*sig**2))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def lorentzian_topnorm(x,mu, gamma):
    """
    Loretzian peak function,
    centered around mu,
    half-with of gamma (FWHM = 2*gamma)
    Amplitude equals 1
    """
    denom = 1 + ((x-mu)/gamma)**2
    return 1 / denom

def lorentzian(x,mu, gamma):
    """
    Loretzian peak function,
    centered around mu,
    half-with of gamma (FWHM = 2*gamma)
    Integral equals 1
    """
    denom = 1 + ((x-mu)/gamma)**2
    return 1 / (denom*np.pi*gamma)


def erf(x,mu,sigma):
    """
    returns the error function scipy.special.erf scaled such that
    its derivative corresponds to a gaussian with the same sigma and my
    """
    return (1+scipy.special.erf((x-mu)/(np.sqrt(2)*sigma)))/2

def spectral_weight(x,y):
    """First moment of distribution y over x"""
    y = y/np.nansum(y)
    return np.nansum(x*y)
def spectral_variance(x,y):
    """Second moment of distribution y over x"""
    y = y/np.nansum(y)
    weight = spectral_weight(x,y)
    return np.nansum(y*(x-weight)**2)

import resource
def memory_used(point=""):
    """Memory used overall or by passed variable."""
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return '''%s: usertime=%s systime=%s mem=%s mb
           '''%(point,usage[0],usage[1],
                usage[2]/1024.0 )

def binn_vector(vec, factor):
    """Binns a vector by a given factor"""
    uneven = np.mod(len(vec),factor)>0
    Lout = int(np.floor(len(vec)/factor))+uneven
    binned = np.empty(Lout)
    for i in range(Lout-uneven):
        binned[i] = np.mean(vec[i*factor:(i+1)*factor])
    if uneven:
        binned[i+1] = np.mean(vec[(i+1)*factor:])
    return binned

def smooth(y, pts):
    """smooth(y, pts) 
    Semi-gaussian smooth function.
    Works by convoluting a box with 
    width of pts twice with itself and
    then the vector y"""
    box = np.ones(pts)/pts
    tri = np.convolve(box,box,'full')
    sgauss = np.convolve(tri,tri, 'same')
    sgauss /= np.sum(sgauss)
    
    padding = np.ones(pts*2)
    y_padded = np.concatenate((padding*y[0],y, padding*y[-1]))
    
    
    smoothed = np.convolve(y_padded, sgauss,'same')
    return smoothed[pts*2:-pts*2]

def shift_by_n(vec,n):
    """ Shift a vector by an even number of elements """
    res = np.zeros(vec.shape)*np.nan
    if n>0:
        res[:n] = np.nan
        res[n:] = vec[:-n]
    elif n<0:
        res[:n] = np.nan
        res[:n] = vec[-n:]
    elif n==0:
        res = vec
    else:
        raise ValueError('Invalid n.')
    return res

def shift_by_delta(y, sft,x = None, oversampling=10):
    """ Shift a vector by any distance, with a precidion of <oversampling>
     compared to the current sampling,
     optionally on an axis x."""
    L = len(y)
    nans = np.isnan(y)    
    if np.any(nans):
        y = interp_nans(y)
        
    if x is None:
        x = np.arange(L)
    yo, xo = sc.signal.resample(y,L*oversampling,t=x, window = ('gaussian',L/2))
    
    dx = np.mean(x[1:]-x[:-1])
    #print(sft,dx)
    shifto = int(np.round(oversampling*sft/dx))
    #print(sft/dx)
    yo_shifted = shift_by_n(yo,shifto)
    y_shifted = sc.signal.resample(interp_nans(yo_shifted),L)
    
    if any(nans):
        nans_shifted = interp_nans(shift_by_n(nans,int(np.round(shifto/oversampling)))==1)
        y_shifted[nans_shifted] = np.nan
        
    return y_shifted

def correlate(a,b):
    """Returns: correlation, shift"""
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    c = np.correlate(a, b, 'full')
    cshift = np.arange(2*len(a)-1)-len(a)+1
    return c, cshift

def fft_shifted(x, Fs):
    """freq_Hz, X = fft_shifted(x, Fs)
    Returns centered frequency axis and fft for given signal x,
    sapled with the frequency Fs.
    
    Test/example-Code:
        dt = .04
        t = np.arange(1000)*dt
        T = 2
        omega = 2*np.pi/T
        x= np.sin(omega*t)

        plt.figure()
        plt.plot(t,x)

        plt.figure()
        freq_Hz, X = lh.fft_shifted(x, 1/dt)
        plt.plot(freq_Hz,np.abs(X),'.-',ms = 1, lw=0.5)
    """
    L = len(x)
    if np.mod(L,2)==1: L-=1 # make sure L is even
    t = np.arange(L)*(1/Fs)
    X  =np.fft.fft(x,L)
    X  = np.fft.fftshift(X)
    freqHz_part = np.arange(0,L/2)*Fs/L
    freq_Hz = np.concatenate((-freqHz_part[::-1],freqHz_part ),axis=0)
    return freq_Hz, X

def addpatch(patch,axes):
    """adds patch to axes"""
    cp = copy(patch)
    cp.axes = None
    cp.figure = None
    cp.set_transform(axes.transData)
    axes.add_patch(cp)
    
def simeq(a,b,d):
    return np.abs(a-b)<d

def within(a, inter):
    """
    checks if <a> is between the tuple inter(left sided)
    """
    intersort = np.sort(inter)
    a = np.array(a)
    return (a>=intersort[0])&(a<intersort[1])
    
def nm2eV(nm):
    wl = nm/1e9
    c = 299792458
    nu = c/wl
    E = 4.13566769692386e-15*nu    
    return E
def eV2nm(eV):
    nu = eV/4.13566769692386e-15
    c = 299792458
    wl = c/nu
    return wl*1e9

def save_dict_to_txt(fname, lib, header = None):
    # Saves a dictionary of 1d arrays of the same size to a txt file
    keys = list(lib.keys())
    savearr1 = np.zeros((len(lib[keys[0]])+1,len(keys)))
    head = ''
    for i,k in enumerate(lib.keys()):
        head = head + ' ' + k
        savearr1[1:,i] = lib[k]
    if header is not None:
        np.savetxt(fname, savearr1, header=header+'\n'+head) 
    else:
        np.savetxt(fname, savearr1, header = head) 

def plot_fft(sig,f_s=0.1, xaxis='freq'):
    from scipy import fftpack

    sig_padded = sig#np.concatenate((sig, np.zeros(int(sig.shape[0]*.1))))
    sigfft = fftpack.fft(sig_padded)
    freqs = fftpack.fftfreq(len(sig_padded)) * f_s
    
    Lh = int(freqs.shape[0]/2)
    fig, ax = plt.subplots()
    
    if xaxis is 'freqs':
        ax.plot(freqs[freqs>0], np.abs(sigfft[freqs>0]),'-',markersize = 2)
        ax.set_xlabel('Frequency (Hz if f_s in 1/s)')
    elif xaxis is 'T':
        ax.plot(1/freqs[freqs>0], np.abs(sigfft[freqs>0]),'-',markersize = 2)
        ax.set_xlabel('Period')
        ax.set_xlim(np.max(1/freqs[freqs>0]), np.min(1/freqs[freqs>0]))
        ax.set_xscale('log')
    else: raise ValueError('xaxis must be freqs or T')
        
    ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
    ax.set_title('Fourier Transform of the reference signal (roi 0)')
    
def oversample2d(image, oversampling, kind = 'cubic'):
    if type(oversampling) in [int,float]:
        oversampling_x = oversampling
        oversampling_y = oversampling
    elif type(oversampling) is tuple:
        oversampling_x = oversampling[0]
        oversampling_y = oversampling[1]
        
    xax = np.arange(image.shape[1])# ingoing axes
    yax = np.arange(image.shape[0])
    x2 = np.linspace(0,image.shape[1],image.shape[1]*oversampling_x) #outgoing axes
    y2 = np.linspace(0,image.shape[0],image.shape[0]*oversampling_y)
    
    f = scipy.interpolate.interp2d(xax, yax, image, kind = kind)
    return  f(x2, y2)

def match_spectra(v1,v2, oversample = 10, return_shifted = False, return_oversampled = False, plot = False):
    ## Oversample both vectors. Pad in fourier domain to avoid overshoots
    # NaN handling only implemented
    L1 = len(v1)
    L2 = len(v2)
    nans1 = np.isnan(v1)
    nans2 = np.isnan(v2)
    
    if np.any(nans1) or np.any(nans2):
        v1 = interp_nans(v1)
        v2 = interp_nans(v2)

    vo1 = sc.signal.resample(v1,L1*oversample,window = ('gaussian',L1/3))
    vo2 = sc.signal.resample(v2,L2*oversample,window = ('gaussian',L1/3))

    if plot:
        plt.figure()
        plt.plot(np.arange(L1),v1,'.-', ms=2,label = 'v1 original')
        plt.plot(np.arange(L1),v2,'.-', ms=2,label = 'v2 original')
        plt.plot(np.arange(L1*oversample)/oversample,vo1,'C0.', ms=2,label = 'v1 oversampled')
        plt.plot(np.arange(L1*oversample)/oversample,vo2,'C1.', ms=2,label = 'v2 oversampled')
        ax1 = plt.gca()
        plt.legend()
    ## Determine exact shift
    corr,shifts = correlate(vo1,vo2) 
    shifto = shifts[np.argmax(corr)]


    ## shift v2 to match v1 and return
    if return_shifted:
        vo2_shifted = shift_by_n(vo2,shifto)
        
        if plot:
            ax1.plot(np.arange(L1*oversample)/oversample,vo2_shifted, label = 'vo2 shifted to match v1') 
            plt.legend()
            
        if return_oversampled:
            return shifto/oversample, vo1, vo2_shifted
        
                                   
        v2_shifted = sc.signal.resample(interp_nans(vo2_shifted),L2)
        
        if any(nans1):
            v1[nans1] = np.nan
        if any(nans2):
            nans2_shifted = shift_by_n(nans2,int(np.round(shifto/oversample)))
            nans2_shifted = interp_nans(nans2_shifted)==1 # yes, really - shift returns nans after all
            v2_shifted[nans2_shifted] = np.nan
                                   
        if plot:
            ax1.plot(v2_shifted, ms=2,label = 'v2 shifted to match v1')
            plt.legend()
        return shifto/oversample, v1, v2_shifted
    else:
        return shifto/oversample
    
def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an 8-connected neighborhood
    neighborhood = scipy.ndimage.morphology.generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = scipy.ndimage.filters.maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = scipy.ndimage.morphology.binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def interp_nans(y, modify_input = False):
    nans, x= nan_helper(y)
    if modify_input:
        y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        return
    else:
        Y = y.copy()
        Y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        return Y
    
def interp_nans2d(array, method = 'linear'):
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    #mask invalid values
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    GD1 = sc.interpolate.griddata((x1, y1), newarr.ravel(),
                          (xx, yy),
                             method=method)
    return GD1


def find_hotpixel(image, threshold, plot = False):
    # Function to find hot pixel and negative values
    blurred = scipy.ndimage.gaussian_filter(image, sigma=2)
    difference = image - blurred
    hot_pixels = np.logical_or(np.abs(difference>threshold),image<0)
    count = np.sum(hot_pixels)
    print('Detected %i hot/dead pixels out of %i.'%(count,int(image.shape[0]*image.shape[1])))
    retimage = image.copy()
    retimage[hot_pixels] = np.nan
    corrimage = retimage.copy()
    corrimage[hot_pixels] = blurred[hot_pixels]
    if plot:
        plt.figure(figsize=(8,4))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        ax1.set_title('Identified Hot pixels')
        ax1.imshow(hot_pixels,interpolation='none',origin='lower')
        ax2.set_title('Image after filtering')
        ax2.imshow(retimage,interpolation='nearest', origin='lower')
    return retimage, hot_pixels, corrimage
        
def binn(x,y,nbins):
	sums, edges = np.histogram(x,nbins, weights = y)
	edges = midpoints(edges)
	means = sums/np.histogram(x,nbins)[0]
	return [edges, means]

def binning(x, data, func, bins=100, bin_length=None):
    #Von Loic
    """ General purpose 1-dimension data binning

        Inputs:
            x: input vector of len N
            data: structured array of len N
            func: a function handle that takes data from a bin an return
                a structured array of statistics computed in that bin
            bins: array of bin-edges or number of desired bins
            bin_length: if not None, the bin width covering the whole range

        Outputs:
            bins: the bins edges
            res: a structured array of binned data
    """

    if bin_length is not None:
        bin_start = np.amin(x)
        bin_end = np.amax(x)
        bins = np.arange(bin_start, bin_end+bin_length, bin_length)
    elif np.size(bins) == 1:
        bin_start = np.amin(x)
        bin_end = np.amax(x)
        bins = np.linspace(bin_start, bin_end, bins)
    bin_centers = (bins[1:]+bins[:-1])/2
    nb_bins = len(bin_centers)

    bin_idx = np.digitize(x, bins)
    dummy = func([])
    res = np.empty((nb_bins), dtype=dummy.dtype)
    for k in range(nb_bins):
        res[k] = func(data[bin_idx == k])

    return bins, res

def binn_weighed(x,values,weigths,bins, range=None):
    # as scipy.stats.binned_statistic,
    # but calls np.average with weights on each bin
    # function as yet untested

    spec, monbins, binno = scipy.stats.binned_statistic(mono,\
                    ratios, statistic=np.average,bins=bins, range = range)

    spec_weighted = np.zeros(spec.shape)
    for i in range(len(monbins)):
        if i==0: continue #this is due to the empty bin 0 story
        spec_weighted[i-1] = np.average(ratios[binno==i],weights=weights[binno==i], range = range)
    return spec, monbins, spec_weighted

def argmax2d(im):
    a = im.shape[0]
    b = im.shape[1]
    
    fl = im.flatten()
    mx = np.argmax(fl)
    
    col = np.mod(mx, a)
    row = np.floor(mx/a)
    return row, col

def lowpass(x0,y0, DX, N = None, order = 2, AC = False, plot = False):
    # Filter the 1d data y over x with a cutoff frequency of 1/DX using a butterworth filter
    # Option to make a new axis with N points with interpolated x values
 
    if AC:
        y0-=np.nanmin(y0)

    if N is not None:
        xax = np.linspace(np.nanmin(x0), np.nanmax(x0),N)
        dx = xax[1]-xax[0]
        y = np.interp(xax, x0,y0)
    else:
        dx = x0[1]-x0[0]
        xax = x0
        y = y0
    
        
    fs = 1/dx
    nyq = 0.5 * fs
    cutoff = 1/(DX) # sample/eV
    normal_cutoff = cutoff / nyq
    
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    Y = filtfilt(b,a,y, padtype = 'even')    
    
    if plot:
        print('dx =',dx,'DX=',DX)
        print('fs=',fs,'fn=',nyq,'cutoff=',cutoff)
        fig, (ax1, ax2) = plt.subplots(2,1)
        w, h = freqz(b, a, worN=8000)
        ax1.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
        #ax1.plot(w, np.abs(h), 'b')
        ax1.plot(cutoff, 0.5*np.sqrt(2), 'ko')
        ax1.axvline(cutoff, color='k')
        ax1.set_xlim(0, 0.5*fs)
        ax1.set_title("Lowpass Filter Frequency Response")
        ax1.set_xlabel('Frequency [1/x axis unit]')
        ax1.grid()
        ax2.plot(xax, y, 'b.',label='input data')
        ax2.plot(xax, Y, 'g-', linewidth=2, label='filtered data')
        ax2.set_xlabel('X-Axis [Unit given]')
        ax2.grid()
        ax2.legend()

    if N is not None:
        return xax,Y
    else:
        return Y

def highpass(x0,y0, DX, N = None, order = 2, AC = False, plot = False):
    # Filter the 1d data y over x with a cutoff frequency of 1/DX using a butterworth filter
    # Option to make a new axis with N points with interpolated x values
 
    if AC:
        y0-=np.nanmin(y0)

    if N is not None:
        xax = np.linspace(np.nanmin(x0), np.nanmax(x0),N)
        dx = xax[1]-xax[0]
        y = np.interp(xax, x0,y0)
    else:
        dx = x0[1]-x0[0]
        xax = x0
        y = y0
    
        
    fs = 1/dx
    nyq = 0.5 * fs
    cutoff = 1/(DX) # sample/eV
    normal_cutoff = cutoff / nyq
    
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    Y = filtfilt(b,a,y, padtype = 'even')    
    
    if plot:
        print('dx =',dx,'DX=',DX)
        print('fs=',fs,'fn=',nyq,'cutoff=',cutoff)
        fig, (ax1, ax2) = plt.subplots(2,1)
        w, h = freqz(b, a, worN=8000)
        ax1.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
        #ax1.plot(w, np.abs(h), 'b')
        ax1.plot(cutoff, 0.5*np.sqrt(2), 'ko')
        ax1.axvline(cutoff, color='k')
        ax1.set_xlim(0, 0.5*fs)
        ax1.set_title("Lowpass Filter Frequency Response")
        ax1.set_xlabel('Frequency [1/x axis unit]')
        ax1.grid()
        ax2.plot(xax, y, 'b.',label='input data')
        ax2.plot(xax, Y, 'g-', linewidth=2, label='filtered data')
        ax2.set_xlabel('X-Axis [Unit given]')
        ax2.grid()
        ax2.legend()

    if N is not None:
        return xax,Y
    else:
        return Y
    
def fancy_errplot(axis, x, y, yerr, smootheness = 2, lab = None, empty = False, smooth_midline = False, **plotopts):
    if type(smootheness) is int:
        DX = smootheness
    elif type(smootheness) is np.float:
        dx = np.mean(x[1:]-x[:-1])
        DX = smootheness/dx/2
        print(dx,DX)
    
    good = np.isfinite(x)&np.isfinite(y)&np.isfinite(yerr)
    x = x[good]
    y = y[good]
    yerr = yerr[good]
    upper = scipy.ndimage.gaussian_filter1d(y+yerr, DX)
    lower = scipy.ndimage.gaussian_filter1d(y-yerr, DX)

    #upper = lowpass(x, y+yerr, smootheness)
    #lower = lowpass(x, y-yerr, smootheness)
    if not empty:
        if smooth_midline:
            #midl = lowpass(x,y,smootheness)
            midl = scipy.ndimage.gaussian_filter1d(y, DX)
            axis.plot(x,midl , label = lab, **plotopts)
        else:
            axis.plot(x,y, label = lab, **plotopts)
    # Filter plotopts for fill plot
    plotopts['linewidth'] = 0
    if 'marker' in plotopts: del plotopts['marker']
    if not 'alpha' in plotopts: plotopts['alpha'] = 0.5
    axis.fill_between(x, upper,lower, **plotopts)
    
def cosmics_masking(image_stack, kernel_size = (3,1), Nsigma = 10, roi = np.s_[:,:], plot = True, Nsigma_average = None, fill_by_average = False, exclude_no_peaks = False, average_image_by = 'median'):
    """
    image_stack, excluded_region, hitlist = cosmics_masking(image_stack, kernel_size = (3,1), Nsigma = 10, roi = np.s_[:,:], plot = True, Nsigma_average= None, fill_by_average = False, exclude_no_peaks = False, average_image_by = 'median')
    
    This function masks outliers in a given 3D array <image_stack> with images in dimensions (1,2).
    This is done by comparing each image with its smoothed counterpart.
    Smoothing is done by convoluting with a Gaussian function with a kernel of <kernel_size>
    Outliers that deviate more than <Nsigma> from the smoothed version are  masked.
    Regions outside of the <roi> (2D, valid for each image) are disregarded and never masked
    As this algorithm can fail around regions with high intensities (e.g. elastic peaks),\
    the same procedure is first done on an average_image with a lower sigma tolerance of <Nsigma_average>.
    Pixels that are found to deviate in the average image are disregarded and never masked
    
    Parameters:
        image_stack: AxBxC sized stack of A images (3D np.array or np.ma.array)
        kernel_size: size of the smoothing kernel (2D tuple of integers)
        Nsigma: max allowed deviation from smoothed images (float)
        roi: disregard anything outside this 2d slice (np.s_ 2D slice object)
        plot: show debug plots (bool)
        Nsigma_average: max allowed deviation for average image from smoothed images (float)
        fill_by_average: If set to true, a regular numpy array is returned instead of a masked one and cosmic pixels are assigned the average value of the image.
        exclude_no_peaks: If set to true, no pixels are excluded from the cosmic masking process.
        average_image_by: Can be 'mean' or 'median'. which is better depends on the dataset
    Returns:
        image_stack: the input image stack, but as a masked array with cosmics masked (3D np.ma.array)
        excluded_region: Mask of what the algorithm disregarded (2D boolean array)
        hitlist: indices of all images where cosmics were found (1D np.array)
    """
    print(f'Begin Cosmic masking.')
    
    non_empty = np.any(image_stack,(1,2))
    
    stack_mask = np.ones(image_stack.shape, dtype = bool)
    roi_excluded_region = np.ones((image_stack.shape[1],image_stack.shape[2]),dtype = bool)
    roi_excluded_region[roi] = False
    
    kernel = astropy.convolution.Gaussian2DKernel(*kernel_size)    
    
    if Nsigma_average is None:
        Nsigma_average = Nsigma/3

    def mask_image(image, kernel, Nsigma, excluded_region = None):
        nan_problem = False
        if np.any(np.isnan(image)):
            nan_problem = True
            nanmask = np.isnan(image)
            avg = np.nanmedian(image)
            image[nanmask] = avg
        #im_sm = sc.ndimage.uniform_filter(np.array(image,dtype=float),size=kernel_size, mode = 'nearest')
        im_sm = astropy.convolution.convolve(np.array(image,dtype=float), kernel, boundary='extend')

        diff = np.abs(image - im_sm)
        deviation = np.nanstd(diff)
        hits = diff>Nsigma*deviation
        if excluded_region is not None:
            hits[excluded_region] = False
        if nan_problem:
            image[nanmask] = np.nan
        return hits

    # First, I see where already the average image has outliers.
    # These are computed for half the sigmas to make sure
    if average_image_by is 'mean':
        avgim = np.nanmean(image_stack[non_empty],0)
    elif average_image_by is 'median':
        avgim = np.nanmedian(image_stack[non_empty],0)
    else:
        print('Average_image_by was neither mean nor median. I will try it as a function (like lambda x: mean(x,0) ).')
        avgim = average_image_by(image_stack[non_empty])
              
    
    if exclude_no_peaks:
        peak_excluded_region = np.zeros((image_stack.shape[1],image_stack.shape[2]),dtype = bool) # Just for having the variable (plotting error catch)
        excluded_region = roi_excluded_region
    else:
        peak_excluded_region_raw = mask_image(avgim,kernel = kernel, Nsigma = Nsigma_average, excluded_region= roi_excluded_region)

        ## Smooth the peak excluded region with the kernel
        #peak_excluded_region_sm = sc.ndimage.convolve(np.array(peak_excluded_region_raw,dtype=float), kernel, mode = 'nearest')
        peak_excluded_region_sm = astropy.convolution.convolve(np.array(peak_excluded_region_raw,dtype=float), kernel, boundary='extend')
        peak_excluded_region = peak_excluded_region_sm>(3/np.nansum(kernel)) # more than 3 pixel within the kernel triggered in the avgim
        excluded_region = peak_excluded_region | roi_excluded_region
    
    if plot:
        fig, axes = plt.subplots(2,1,constrained_layout = True)
        fig.suptitle('Cosmic Correction')
        plt.sca(axes[0])
        plt.title('Average Image (Blue) and where cosmics were found (Red)')
        plt.imshow(avgim.T, aspect='auto', alpha = 1, cmap = 'Blues')
        plt.sca(axes[1])
        plt.imshow(roi_excluded_region.T, aspect='auto', alpha = .7, cmap = 'Blues')
        plt.imshow(peak_excluded_region.T, aspect='auto', alpha = .3, cmap = 'Reds')
        plt.title(f'Blue: Excluded by ROI, Red: Excluded due to peaks')

    print(f'Computed peak region to exclude from algorithm, containing {np.nansum(peak_excluded_region)} pixel.\n \
          furthermore, {np.nansum(roi_excluded_region)} pixel are not in the given ROI')
    
    # Now we iterate through the stack and make masks,
    # keeping track in the hitlist of where cosmits were detected (which events)
    hitlist = []
    Nskipped = 0
    for i, im in enumerate(image_stack):
        if not np.any(image_stack[i]):# any point not equal origin makes sure that empty images are omitted
            Nskipped += 1
            continue
            
        stack_mask[i] = mask_image(im, kernel, Nsigma, excluded_region=excluded_region)
        if np.any(stack_mask[i]):
            hitlist.append(i)
        if fill_by_average:
            median = np.nanmedian(np.ma.array(data=im,mask= stack_mask[i]))
            image_stack[i][stack_mask[i]==True] = median
            #= np.ma.fix_invalid(im, mask = stack_mask[i], copy = False, fill_value = median)
            
    print(f'Skipped {Nskipped} empty images.')
    print(f'Found {np.sum(stack_mask[non_empty])} cosmics in {len(hitlist)} out of {len(image_stack[non_empty])} non-empty images.')
    if fill_by_average:
        print(f'Pixels with cosmics where set to the image mean value.')
    else:
        image_stack = np.ma.array(data= image_stack, mask = stack_mask)
    
    if plot:
        plt.sca(axes[0])
        found_image = np.nansum(np.array(stack_mask[non_empty],dtype=float),0).T
        found_image[found_image==0] = np.nan
        plt.imshow(found_image, aspect='auto', cmap = 'Reds',vmin = 0, vmax = 5)#, norm = LogNorm())
    return image_stack, excluded_region, hitlist
