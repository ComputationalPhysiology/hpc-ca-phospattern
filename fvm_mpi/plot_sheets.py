import matplotlib.pyplot as plt
import time
import h5py

def plot_sheets(casename):

    with h5py.File("{}.h5".format(casename)) as f:
        if "data_0000000" not in f.keys():
            raise ValueError("No data in file")
        sheet_names = [data_key for data_key in f["data_0000000"].keys() if "sheet" in data_key]

        species_names = sorted(set(name.split("_sheet_")[0] for name in sheet_names))
        print species_names
        z_indices = sorted(set(int(name.split("_sheet_")[1]) for name in sheet_names))
        
        if len(sheet_names) == 0:
            raise ValueError("No sheets stored in file.")
        
        base_scale = 6
        figsize = [len(species_names)*base_scale, base_scale*len(z_indices)]
        plt.rcParams.update({'figure.figsize': figsize})
        plt.interactive(True)
        fig = plt.figure()

        #extent = [0, ., 0, nz*h/1000.]
        for data_ind, data_name in enumerate(f.keys()):
            plt.clf()
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

            plot_ind = 1
            for z_ind in z_indices:
                for species in species_names:
                    data = f[data_name]["{}_sheet_{}".format(species, z_ind)].value
                    plt.subplot(len(z_indices)*100 + len(species_names)*10 + plot_ind)
                    #plt.imshow(data[1:-1,1:-1], extent=extent, origin="lower")
                    plt.imshow(data, origin="lower")
                    #plt.clim(0.,3.0)
                    plt.colorbar()
                    plot_ind += 1
                    plt.title("{}:{}".format(species,data_ind))
                    plt.xlabel("[um]")
                    plt.ylabel("[um]")
                    
            
            plt.draw()
            time.sleep(1.0)

        plt.interactive(True)
        plt.show()
        time.sleep(100000.0)
            
if __name__ == "__main__":
    plot_sheets("casename")
    #plt.show()
