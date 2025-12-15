# Programming exercise with the radiative transfer model RRTMG
# METRAD module, University of Cologne

*Questions:* kerstin.ebell@uni-koeln.de, a.walbroel@uni-koeln.de

The energetic impact of atmospheric radiation will be analyzed using the broadband, one-dimensional radiative transfer model RRTMG of the Atmospheric and Environmental Research, Incorporated ([RRTM](http://rtweb.aer.com/rrtm_frame.html)). Here, we use RRTMG in the python framework pyRRTMG, which is used by [T-CARS](https://doi.org/10.5281/zenodo.11147087) (Tropospheric Research Cloud and Aerosol Radiative effect Simulator). RRTMG is implemented in many numerical weather prediction and climate models, e.g. in the ECMWF Integrated Forecasting System, the NCEP Global Forecast System, the climate model ECHAM5 and in the operational global model ICON of the Deutscher Wetterdienst.

RRTMG consists of 14 consecutive spectral bands in the solar spectral region (UV/VIS: 0.2-0.625 μm; NIR: 0.625-12.195 μm) and 16 bands in the longwave spectral region (3.077-1000 μm). It accounts for absorption of longwave radiation and extinction, i.e. absorption and scattering, of shortwave radiation by water vapor (H<sub>2</sub>O), carbon dioxide (CO<sub>2</sub>), ozone (O<sub>3</sub>), methane (CH<sub>4</sub>), molecular oxygen (O<sub>2</sub>), nitrous oxide (N<sub>2</sub>O), liquid water and ice clouds. The effect of clouds is taken into account by means of their liquid and ice water content, the corresponding effective radii and cloud fraction. In addition, assumptions on the overlapping of clouds at different heights need to be made. Usually, the so called maximum-random overlap is assumed. This means, that clouds of adjacent height levels maximally overlap and randomly, if they are separated by cloud-free layers.

Multiple scattering of radiation is allowed for in the RRTMG by the two-stream approach: instead of calculating radiances coming from each solid angle, only irradiances (fluxes) from two directions, i.e. a downward component from the upper hemisphere and a upward component of the lower hemisphere, are considered. This approach is usually applied in numerical weather prediction and climate models, since it enables a quite accurate calculation of irradiances and thus atmospheric heating rates

### Preparation

1. (If needed:) Copy some background data needed for T-CARS: subarctic_summer.txt to .../vmr_Anderson_1986/ and NOAA_Annual_Mean_MoleFractions_2023.csv to .../trace_gas_data/. Then, change `path_data` in the **metrad_rrtmg_ex.py** cell in such a way that the path points to the directory where the folders vmr_Anderson_1986 and trace_gas_data are located.
2. While answering the questions, you have to change some parameters in the **metrad_rrtmg_ex.py** cell. Also make sure to rename your output files and plots for each task. Otherwise, your previous results will be overwritten! For example, you can use and set the `suffix` and `subtask` variables in the **metrad_rrtmg_ex.py** cell.
3. Run all cells step by step. The first couple of cells only define functions and constants used for the RRTMG simulations and plotting.
4. You may save the simulation results using `tcars_client.save_output`. The command `ncdump -h <file>` shows which variables have been stored:
   | Name | Description | Unit |
   | ---- | ----------- | ---- |
   | time | not important here, undefined | / |
   | height | height of the 26 full level (center of layer) | in m |
   | height_h | height of the 27 half level (layer boundaries) | in m |
   | swuflx | Upward shortwave flux at the 27 half levels | in W m-2 |
   | swdflx | Downward shortwave flux | in W m-2 |
   | swdirflx | Direct shortwave flux | in W m-1 |
   | swhr | Shortwave radiative heating rates for layers | in K day-1 |
   | swuflxc | Clear sky upward shortwave flux | in W m-2 |
   | swdflxc | Clear sky downward shortwave flux | in W m-2 |
   | swdirflxc | Clear sky direct shortwave flux | in W m-2 |
   | swhrc | Clear sky shortwave radiative heating rates | in K day-1 |
   | lwuflx | Upward longwave flux | in W m-2 |
   | lwdflx | Downward longwave flux | in W m-2 |
   | lwhr | Longwave radiative heating rates | in K day-1 |
   | lwuflxc | Clear sky upward longwave flux | in W m-2 |
   | lwdflxc | Clear sky downward longwave flux | in W m-2 |
   | lwhrc | Clear sky longwave radiative heating rates | in K day-1 |

The radiative fluxes and heating rates are calculated for one atmospheric profile (mid-latitude summer, MLS), which consists of 26 vertical layers with a thickness of 1 km. The radiative fluxes are provided for the layer boundaries, the heating rates for the model layers. When the radiation budget is positive (more radiation goes into the layer than out of the layer), the layer is heated. In the opposite case, the atmospheric layer is cooled.

Below, necessary constants and functions used for this exercise are defined and loaded. For the exercise, only the **metrad_rrtmg_ex.py** cell must be modified.

### Changing the atmospheric profile

In order to analyze the effect of trace gases (CO<sub>2</sub>), surface albedo, temperature, humidity and clouds on the radiative fluxes, the input data is modified. This can be done directly in the **metrad_rrtmg_ex.py** cell. There are three possibilities to change the profile:
1. Scaling the whole profile: change the corresponding factor in the `scaling` dictionary.
2. Adding an offset in each layer: change the corresponding values in the `offset` dictionary.
3. Changing directly the profile values and special parameters: Modify variables in the xarray Dataset DS after loading it with `read_default_profiles` or modify `tcars_client.DS`.

If clouds shall be included in the profile, modify the variable `cloudtypes`, which is either an empty list, or a list containing the elements 'liquid', 'ice' or both. Optionally, you can change the effective radius of the liquid or ice cloud and define your own liquid and ice water paths by adjusting the following variables in the function `define_cloud`: `re_value`, `lwp_iwp_lim` (the latter is an argument of the `fill_cloud_data` function). The cloud fraction 'clc' is automatically set to 1 at the same height where liquid water path (LWP) or ice water path (IWP) is added.

Because of the chosen parameterization of the optical properties of liquid clouds (Hu und Stamnes, 1993), `re_value['liquid']` must have a value between 2.5 and 60 μm. For `re_value['ice']`, values between 13 and 130 μm have to be used (Ebert und Curry, 1992). By default, typical values are set for `re_value['liquid']` and `re_value['ice']`: 5 µm and 30 µm, respectively.

### Reading the output and plotting the results

The cell below the **metrad_rrtmg_ex.py** cell executes the RRTMG simulations, as well as some basic plotting and output saving routines. Theoretically, the lines saving the output can be commented out because the results can also be directly plotted using that cell. Default plots are:
- `plot_heating_rates`: Longwave and shortwave heating rates for clear and all sky conditions (note that clear and all sky heating rates are identical if no cloud has been defined)
- `plot_cloud_forcing_heating_rate`: Cloud forcing heating rates for the shortwave and longwave spectrum
- `plot_transmission_direct_diffuse_rad_lwp`: Transmission of shortwave radiative flux and the ratio of direct and diffuse shortwave radiative fluxes

### Tasks and questions

Start with the temperature-/humidity profile of the mid-latitude standard atmosphere in the summer (cosine of solar zenith angle=0.7). Set the solar surface albedo to 0.2. These values are readily set in `read_default_profiles`.

<img src="glob_energy_flows_overview.png" width="600">

### Clear-sky studies

1. Calculate the radiation budget (S↓ - S↑ + L↓ - L↑) at the surface, at the top of the atmosphere (TOA) and for the atmosphere. Note that the computation is already performed using the function `compute_radiative_flux_products`, written to the variable `OUT_DS` in the plotting cell.

For the following exercises, change the settings in the **metrad_rrtmg_ex.py** cell as described below. Make sure that you always undo the changes before you move on to the next exercise so that only one variable is changed for each exercise.

2. Calculate the radiation budget of the atmosphere for CO<sub>2</sub> concentration at pre-industrial times (280 ppm) and four times this value (note that 1. used CO<sub>2</sub> concentrations of 400 ppm). For this, set  
   `tcars_client.iflag_co2_vmr = 1`  
   For quadrupling the CO<sub>2</sub> concentration, additionally change the value of the key `co2_vmr` of the `scaling` dictionary to 4. Determine the difference of both cases and discuss implications for climate and possible feedback mechanisms.
3. What happens if the humidity is increased by 50 % in the whole atmosphere? Change the value of the key `h2o_vmr` of the `scaling` dictionary to 1.5.
4. How does a) an increase of the temperature profile by 10 K and b) an increase of the temperature in the lowest layer by 10 K affect the results?  
   For a), set the value of the key `temp_h` of the `offset` dictionary to 10.  
   For b), add 20 K to the first value in `temp_h`: For this, use  
   `tcars_client.DS['temp_h'][...,0] += 20.`  
   Note that adding 20 K to the lower boundary of the lowest layer (level index 0) while keeping its upper boundary (level index 1) unchanged results in an increase of the layer temperature by 10 K.
5. How does the radiation budget change, if the surface albedo has typical values for ocean (\~0.06), forest (\~0.15), grassland (\~0.2) and desert (\~0.3)? To keep it simple, use the same value for both diffuse and direct albedo. For this, change the value given to  
   `tcars_client.DS[var][:] = `  
   accordingly, where var takes the variable names 'alb_dir_uv', 'alb_dif_uv', 'alb_dir_nir' and 'alb_dif_nir' (see surrounding for loop).

### Cloudy-sky sensitivities

Change the albedo back to 0.2.
Add a liquid water cloud to the standard atmosphere between 2 and 3 km. Set  
`cloudtypes = ['liquid']`  
and make sure that  
`cloud_height = np.array([2000., 3000.])`

By default, these RRTMG simulations are set up to simulate liquid water path (LWP) values between 0 and 500 g m<sup>-2</sup> with a step of 10 g m<sup>-2</sup> (see function `fill_cloud_data`in `define_cloud`). Plots are only created for LWP 0, 30, 100 and 500 g m<sup>-2</sup>. The cloud cover `clc` and droplet effective radii `re_liq` are automatically set to 1 and typical values, respectively, where needed.

6. How does the radiation budget change?
7. Depict the transmission of the solar radiation as a function of the total liquid water path of the cloud (LWP). You can use `plot_transmission_direct_diffuse_rad_lwp` for this. This function is executed automatically by default.
8. How does the ratio direct/diffuse radiation change if the LWP is increased? For visualisation, you can also use `plot_transmission_direct_diffuse_rad_lwp`.
9. What happens if the cloud (LWP: 100 g m<sup>-2</sup>) is vertically moved and located between 5 and 6 km height? Adapt the variable `cloud_height`. You can use the variable `OUT_DS_all`, which contains all needed variables for all simulated LWP values (default: 0 to 500 g m<sup>-2</sup>in 10 g m<sup>-2</sup> steps).
10. What happens if the cloud from 9. is replaced by an ice cloud with the same water content? Set:  
    `cloudtypes = ['ice']`
11. And if both liquid and ice clouds occur between 5 and 6 km? Use:  
    `cloudtypes = ['liquid', 'ice']`

### Heating rate studies

12. Plot the shortwave and longwave heating rate profiles for the six atmospheric states in 6., 9., 10. and 11.. Always plot both profiles, with clouds and clear sky. For plotting, you can use `plot_heating_rates`.
13. Plot also the difference of cloud and clear sky heating rates, i.e. the cloud radiative forcing. Plot the cloud radiative forcing for the SW, the LW and the net value (SUM of SW and LW). You can use `plot_cloud_forcing_heating_rate`.
14. Which state leads particularly to a warming/cooling of the atmosphere?
