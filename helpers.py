### Helper functions
from datetime import datetime,timedelta
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import sunpy.coordinates
import sunpy.map
import pfsspy

def gen_dt_arr(dt_init,dt_final,cadence_days=1) :
    """
    Get array of datetime.datetime from {dt_init} to {dt_final} every 
    {cadence_days} days
    """
    dt_list = []
    while dt_init < dt_final :
        dt_list.append(dt_init)
        dt_init += timedelta(days=cadence_days)
    return np.array(dt_list)

def carr2SkyCoord(lon,lat,radius,obstime) :
    """
    Given a set of heliographic coordinates and an observation time,  return
    an `astropy.coordinates.SkyCoord` encoding that information.
    Useful for annotating `sunpy.map.Map` with additional data.
    """
    return SkyCoord(lon=lon,lat=lat,radius=radius,
                    frame = sunpy.coordinates.HeliographicCarrington(
                        observer="Earth",obstime=obstime
                        ),
                    representation_type="spherical"
                   )

def get_pixel_lims(xlims_world,ylims_world,map_,ax):
    """
    Get pixel bounds of a given set of world coordinates for
    a `sunpy.map.Map` (speeds up changing the limits/zooming)
    """
    if "CRLN" in map_.coordinate_system[0] :
        world_coords = SkyCoord(lon=xlims_world, 
                                lat=ylims_world,
                                frame=map_.coordinate_frame)
    else : 
        world_coords = SkyCoord(Tx=xlims_world, 
                                Ty=ylims_world, 
                                frame=map_.coordinate_frame)
    pixel_coords = map_.world_to_pixel(world_coords)
    return pixel_coords.x.value, pixel_coords.y.value

def datetime2unix(dt_arr) :
    """Convert 1D array of `datetime.datetime` to unix timestamps"""
    return np.array([dt.timestamp() for dt in dt_arr])

def unix2datetime(ut_arr) : 
    """Convert 1D array of unix timestamps (float) to `datetime.datetime`"""
    return np.array([datetime.utcfromtimestamp(ut) for ut in ut_arr])

@u.quantity_input
def delta_long(r:u.R_sun,
               r_inner=2.5*u.R_sun,
               vsw=360.*u.km/u.s,
               omega_sun=14.713*u.deg/u.d,
               ):
    """ 
    Ballistic longitudinal shift of a Parker spiral connecting two
    points at radius r and r_inner, for a solar wind speed vsw. Solar
    rotation rate is also tunable
    """
    return (omega_sun * (r - r_inner) / vsw).to("deg")

def ballistically_project(skycoord,r_inner = 2.5*u.R_sun, vr_arr=None) :
    """
    Given a `SkyCoord` of a spacecraft trajectory in the Carrington frame,
    with `representation_type="spherical"`, and optionally an array of
    measured solar wind speeds at the same time intervals of the trajectory,
    return a SkyCoord for the trajectory ballistically projected down to 
    `r_inner` via a Parker spiral of the appropriate curvature. When `vr_arr`
    is not supplied, assumes wind speed is everywhere 360 km/s
    """
    if vr_arr is None : vr_arr = np.ones(len(skycoord))*360*u.km/u.s
    lons_shifted = skycoord.lon + delta_long(skycoord.radius,
                                             r_inner=r_inner,
                                             vsw=vr_arr
                                            )
    return SkyCoord(
        lon = lons_shifted, 
        lat = skycoord.lat,
        radius = r_inner * np.ones(len(skycoord)),
        representation_type="spherical",
        frame = skycoord.frame
    )
@u.quantity_input
def rollto180(arr:u.deg) : 
    """
    Cast an array of longitudes in the range [0,360] deg to the range
    [-180,180] deg. Useful when examining stuff that crosses through
    Carrington L0.
    """
    return (((arr + 180*u.deg).to("deg").value % 360) - 180)*u.deg

def adapt2pfsspy(filepath, #must already exist on your computer
                 rss=2.5, # Source surface height
                 nr=60, # number of radial gridpoints for model
                 realization="mean", #which slice of the adapt ensemble to choose
                 return_magnetogram = False # switch to true for function to return the input magnetogram
                ):
    """ read in an ADAPT fits file and use it to generate a PFSS model using pfsspy"""
    # Load the FITS file into memory
    # ADAPT includes 12 "realizations" - model ensembles
    # pfsspy.utils.load_adapt is a specific function that knows
    # how to handle adapt maps
    adaptMapSequence = pfsspy.utils.load_adapt(filepath)
    # If realization = mean, just average them all together
    if realization == "mean" : 
        br_adapt_ = np.mean([m.data for m in adaptMapSequence],axis=0)
        adapt_map = sunpy.map.Map(br_adapt_,adaptMapSequence[0].meta)
    # If you enter an integer between 0 and 11, the corresponding
    # realization is selected
    elif isinstance(realization,int) : adapt_map = adaptMapSequence[realization]
    else : raise ValueError("realization should either be 'mean' or type int ") 
    
    # pfsspy requires that the y-axis be in sin(degrees) not degrees
    # pfsspy.utils.car_to_cea does this conversion
    adapt_map_strumfric = pfsspy.utils.car_to_cea(adapt_map)

    # Option to return the magnetogram
    if return_magnetogram : 
        return adapt_map_strumfric
    # Otherwise run the PFSS Model and return
    else :
        # ADAPT maps input are in Gauss, multiply by 1e5 to units of nT
        adapt_map_input = sunpy.map.Map(adapt_map_strumfric.data*1e5,
                                        adapt_map_strumfric.meta)
        peri_input = pfsspy.Input(adapt_map_input, nr, rss)
        peri_output = pfsspy.pfss(peri_input)
        return peri_output


# Define function which does the field line tracing
def pfss2flines(pfsspy_output, # pfsspy output object
                nth=18,nph=36, # number of tracing grid points
                rect=[-90,90,0,360], #sub-region of sun to trace (default is whole sun)
                trace_from_SS=False, # if False : start trace from photosphere, 
                                     #if True, start tracing from source surface
                skycoord_in=None, # Use custom set of starting grid poitns
                max_steps = 1000 # max steps tracer should take before giving up
                ) :
    """
    For input `pfsspy.output` instance, use pfsspy tracing api to produce field lines.
    Default behavior is to trace a grid of points in heliographic coords bounded by 
    rect = [latmin,latmax,lonmin,lonmax], but tracing can also be done from a custom
    input SkyCoord, e.g. a set of spacecraft magnetic footpoints at 2.5Rs
    """
    
    # Tracing if grid
    if skycoord_in is None  :
        [latmin,latmax,lonmin,lonmax]=rect
        lons,lats = np.meshgrid(np.linspace(lonmin,lonmax,nph),
                                np.linspace(latmin,latmax,nth)
                                )
        if not trace_from_SS : alt = 1.0*u.R_sun # Trace up from photosphere
        else : alt = po.grid.rss*u.R_sun  # Trace down from ss
        alt = [alt]*len(lons.ravel())
        seeds = SkyCoord(lons.ravel()*u.deg,
                         lats.ravel()*u.deg,
                         alt,
                         frame = pfsspy_output.coordinate_frame
                         )
        
    # Tracing if custom set of points (SkyCoord)
    else : 
        skycoord_in.representation_type = "spherical"
        seeds = SkyCoord(skycoord_in.lon,
                         skycoord_in.lat,
                         skycoord_in.radius,
                         frame = pfsspy_output.coordinate_frame
                         )
        
    return pfsspy_output.trace(pfsspy.tracing.FortranTracer(max_steps=max_steps),seeds)

def smooth(time_series,box_size) :
    """ 
    Smooth an input `np.array` by taking the mean over some box/window size
    Window moves incrementally such that output array is same shape as input
    """
    time_series_smoothed = np.copy(time_series)
    if box_size == 0 : return time_series_smoothed
    for ii in np.arange(box_size,len(time_series)-1-box_size) :
        time_series_smoothed[ii] = np.nanmean(time_series[ii-box_size:ii+box_size+1])
    return time_series_smoothed