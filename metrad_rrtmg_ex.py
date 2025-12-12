import os
import sys
import pdb
from tqdm import tqdm

import numpy as np
import matplotlib as mpl
mpl.use("WebAgg")                               # uncomment if needed (if some tornado error pops up)
import matplotlib.pyplot as plt
import xarray as xr

from readers_writers import read_default_profiles
from tcars import tcars
import plot_styles as pls

def main():
    
    path_data = {'tcars_data': "/mnt/d/heavy_data/share_data_TCARS_example/",
                 }
    path_output = "/mnt/d/heavy_data/METRAD_WS2526/rrtmg_output/"
    path_plots = "/mnt/d/Studium_NIM/work/Plots/METRAD_WS2526/"
    
    set_dict = {'save_figures': True,       # if True: plot is only saved to file, not shown
                }
    
    # scaling: to scale the parameters in read_default_profiles
    scaling = dict(
          pres      = 1.,
          pres_h    = 1.,
          temp_h    = 1.,
          lwp       = 1.e+00,
          iwp       = 1.e+00,
          re_liq    = 1.,
          re_ice    = 1.,
          h2o_vmr   = 1.,           # modify for question 03
          co2_vmr   = 1.,           # modify for question 02
          o3_vmr    = 1.,
          n2o_vmr   = 1.,
          ch4_vmr   = 1.,
          o2_vmr    = 1.,
          height    = 1.,
          height_h  = 1.,
          )
    
    # offset: to apply additive offsets to the data in read_default_profiles
    offset = dict(
          pres       = 0.,
          pres_h     = 0.,
          temp_h     = 0.,          # modify for question 04
          lwp        = 0.,
          iwp        = 0.,
          re_liq     = 0.,
          re_ice     = 0.,
          h2o_vmr    = 0.,
          co2_vmr    = 0.,
          o3_vmr     = 0.,
          n2o_vmr    = 0.,
          ch4_vmr    = 0.,
          o2_vmr     = 0.,
          height     = 0.,
          height_h   = 0.,  
          )
    

    DS = read_default_profiles(scaling=scaling, offset=offset)    
    tcars_client = tcars(path_tcars_data=path_data['tcars_data'], DS=DS)


    suffix = "_q11"
    subtask = "a"
    
    # DATA MODIFICATIONS:
    tcars_client.iflag_co2_vmr = 0      # adjust for question 02; for valid values, see tcars.py.__init__
    # tcars_client.DS['temp_h'][...,0] += 20.     # question 04b
    # for var in ['alb_dir_uv', 'alb_dif_uv', 'alb_dir_nir', 'alb_dif_nir']:    # question 05
    #     tcars_client.DS[var][:] = 0.3           # question 05
    
    cloudtypes = ['ice', 'liquid']
    new_cloud_vars, cloudvar = define_cloud(cloudtypes)

    OUT_DS_list = list()
    count = 0
    
    for k in tqdm(range(len(new_cloud_vars[cloudvar]))):
        
        for key, var in new_cloud_vars.items():
            tcars_client.DS[key].loc[{'height': slice(5000.,6000.)}][...] = var[k]
        
        
        tcars_client.set_rrtmg_input()
        tcars_client.run_tcars()
        
        
        OUT_DS = compute_radiative_flux_products(tcars_client.OUT_DS)
        OUT_DS_list.append(OUT_DS)
        
        if np.any(np.abs(new_cloud_vars[cloudvar][k] - np.array([30., 100., 500.])) == 0.):
            subtask = pls.panel_marker[count]
            print(f"TASK{suffix}{subtask}\nBudget surface: {OUT_DS.NET_SFC:.1f} W m-2\n" +
                  f"Budget TOA: {OUT_DS.NET_TOA:.1f} W m-2\nBudget atmosphere: {OUT_DS.NET_ATM:.1f} W m-2\n")
            
            plot_heating_rates(tcars_client.OUT_DS, 
                               filename=f"metrad_heating_rates_sw_lw{suffix}{subtask}.png",
                               path=path_plots, 
                               **set_dict)
            plot_cloud_forcing_heating_rate(tcars_client.OUT_DS,
                                            filename=f"metrad_cloud_forcing_sw_lw{suffix}{subtask}.png",
                                            path=path_plots, 
                                            **set_dict)
            
            tcars_client.save_output(path=path_output, filename=f"metrad_rrtmg{suffix}{subtask}.nc")
            OUT_DS.to_netcdf(path_output + f"metrad_radiation_budget{suffix}{subtask}.nc", mode='w', format='NETCDF4')
            
            count += 1
    
    OUT_DS_all = xr.concat(OUT_DS_list, dim=cloudvar).assign_coords({cloudvar: ([cloudvar], new_cloud_vars[cloudvar])})
    plot_transmission_direct_diffuse_rad_lwp(OUT_DS_all, 
                                             xaxis_data=cloudvar,
                                             filename=f"metrad_transmission{suffix}.png",
                                             path=path_plots,
                                             **set_dict)
    
    
def define_cloud(cloudtypes: list):
    
    cloudvars = {'liquid': 'lwp', 'ice': 'iwp'}
    re_cloud = {'lwp': 're_liq', 'iwp': 're_ice'}
    re_value = {'liquid': 5., 'ice': 30.}
    new_cloud_vars = dict()
    for cloudtype in cloudtypes:
        cloudvar = cloudvars[cloudtype]
        new_cloud_vars[cloudvar] = np.arange(0., 500.1, 10.)
        new_cloud_vars['clc'] = np.full(new_cloud_vars[cloudvar].shape, 1.)
        new_cloud_vars[re_cloud[cloudvar]] = np.full(new_cloud_vars[cloudvar].shape, re_value[cloudtype])
        
    return new_cloud_vars, cloudvar
    
    
def compute_radiative_flux_products(DS: xr.Dataset):
    
    DS = DS.mean('time')
    DS_TOA = DS.isel(height_h=-1)
    DS_SFC = DS.sel(height_h=0., method='nearest')
    
    OUT_DS = xr.Dataset()
    OUT_DS['NET_TOA'] = DS_TOA['swdflx'] - DS_TOA['swuflx'] + DS_TOA['lwdflx'] - DS_TOA['lwuflx']
    OUT_DS['SW_DOWN_TOA'] = DS_TOA['swdflx']
    OUT_DS['SW_UP_TOA'] = DS_TOA['swuflx']
    OUT_DS['LW_DOWN_TOA'] = DS_TOA['lwdflx']
    OUT_DS['LW_UP_TOA'] = DS_TOA['lwuflx']
    
    OUT_DS['NET_SFC'] = DS_SFC['swdflx'] - DS_SFC['swuflx'] + DS_SFC['lwdflx'] - DS_SFC['lwuflx']
    OUT_DS['SW_DOWN_SFC'] = DS_SFC['swdflx']
    OUT_DS['SW_UP_SFC'] = DS_SFC['swuflx']
    OUT_DS['LW_DOWN_SFC'] = DS_SFC['lwdflx']
    OUT_DS['LW_UP_SFC'] = DS_SFC['lwuflx']
    
    OUT_DS['NET_ATM'] = OUT_DS['NET_TOA'] - OUT_DS['NET_SFC']
    
    OUT_DS['DIR_DIFF'] = DS_SFC['swdirflx'] / (DS_SFC['swdflx'] - DS_SFC['swdirflx'])
    
    OUT_DS['TRANS'] = DS_SFC['swdflx'] / DS_TOA['swdflx']
    
    return OUT_DS


def plot_heating_rates(
    DS: xr.Dataset,
    filename="metrad_heating_rates_sw_lw.png",
    path=os.getcwd()+"/",
    save_figures=False):
    
    DS = DS.mean('time')
    
    f1, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(7,5))
    
    plt.subplots_adjust(left=0.125, right=0.96, top=0.96, bottom=0.12, wspace=0.08)
    
    all_sky_style = dict(color='tab:orange', linewidth=pls.linewidth, marker='o', markersize=6)
    clear_sky_style = dict(color='tab:blue', linewidth=pls.linewidth, marker='x', linestyle='dashed', markersize=6)
    axs[0].plot(DS.lwhr, DS.height*0.001, label='All sky', **all_sky_style)
    axs[0].plot(DS.lwhrc, DS.height*0.001, label='Clear sky', **clear_sky_style)
    
    axs[1].plot(DS.swhr, DS.height*0.001, label='All sky', **all_sky_style)
    axs[1].plot(DS.swhrc, DS.height*0.001, label='Clear sky', **clear_sky_style)
    
    for ax in axs:
        lh, ll = ax.get_legend_handles_labels()
        ax.legend(lh, ll, loc='best', frameon=False)
        
        ax.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.3)
    
    axs[0].set_ylabel("Height (km)")
    axs[0].set_xlabel("Longwave heating rate (K$\,$day$^{-1}$)")
    axs[1].set_xlabel("Shortwave heating rate (K$\,$day$^{-1}$)")
    
    if save_figures:
        os.makedirs(path, exist_ok=True)
        
        plotfile = os.path.join(path, filename)
        f1.savefig(plotfile, dpi=150, bbox_inches='tight')

        print(f"Saved {plotfile} ....")
    else:
        plt.show()
        
    plt.close()


def plot_cloud_forcing_heating_rate(
    DS: xr.Dataset,
    filename="metrad_cloud_forcing_sw_lw.png",
    path=os.getcwd()+"/",
    save_figures=False):
    
    DS = DS.mean('time')
    
    f1 = plt.figure(figsize=(4,4))
    a1 = plt.axes()
    
    plt.subplots_adjust(left=0.15, right=0.96, top=0.96, bottom=0.15)
    
    lw_style = dict(color='tab:blue', linewidth=pls.linewidth, marker='o', markersize=6)
    sw_style = dict(color='tab:orange', linewidth=pls.linewidth, marker='o', markersize=6)
    sum_style = dict(color='k', linewidth=pls.linewidth, marker='o', markersize=6)
    a1.plot(DS.lwhr - DS.lwhrc, DS.height*0.001, **lw_style, label='LW')
    a1.plot(DS.swhr - DS.swhrc, DS.height*0.001, **sw_style, label='SW')
    a1.plot((DS.lwhr - DS.lwhrc) + (DS.swhr - DS.swhrc), DS.height*0.001, **sum_style, label="LW+SW")
    
    lh, ll = a1.get_legend_handles_labels()
    a1.legend(lh, ll, loc='best', frameon=False)
    a1.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.3)
    
    a1.set_ylabel("Height (km)")
    a1.set_xlabel("Cloud forcing (K$\,$day$^{-1}$)")
    
    if save_figures:
        os.makedirs(path, exist_ok=True)
        
        plotfile = os.path.join(path, filename)
        f1.savefig(plotfile, dpi=150, bbox_inches='tight')

        print(f"Saved {plotfile} ....")
    else:
        plt.show()
        
    plt.close()


def plot_transmission_direct_diffuse_rad_lwp(
    DS: xr.Dataset,
    xaxis_data='lwp',
    filename="metrad_transmission.png",
    path=os.getcwd()+"/",
    save_figures=False):
    
    f1, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(5,5))
    
    plt.subplots_adjust(left=0.1, right=0.96, top=0.96, bottom=0.12, hspace=0.08)
    
    axs[0].plot(DS[xaxis_data], DS.TRANS*100., linewidth=pls.linewidth)
    axs[1].plot(DS[xaxis_data], DS.DIR_DIFF, linewidth=pls.linewidth)
    
    for ax in axs:
        ax.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.3)
    
    axs[0].set_ylabel("Transmission (%)")
    axs[1].set_ylabel("Ratio direct/diffuse radiation ( )")
    axs[1].set_xlabel(f"{xaxis_data.upper()} and IWP" + " (g$\,$m$^{-1}$)")
    
    if save_figures:
        os.makedirs(path, exist_ok=True)
        
        plotfile = os.path.join(path, filename)
        f1.savefig(plotfile, dpi=150, bbox_inches='tight')

        print(f"Saved {plotfile} ....")
    else:
        plt.show()
        
    plt.close()


if __name__ == "__main__":
    main()