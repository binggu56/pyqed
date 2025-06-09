from pyqed.floquet import Floquet
from pyqed import Mol
import numpy as np
import matplotlib.pyplot as plt
from pyqed.floquet.Floquet import TightBinding

def test_Gomez_Leon_2013(E0 = 200, number_of_step_in_b = 11, nt = 21, omega = 10, relative_Hopping = [1.5,1], save_band_structure=False, data_saving_root = 'local_data_GL2013'):
    """
    Test the Gomez-Leon 2013 model but using TightBinding and FloquetBloch classes, calculate the winding number and finally plot the heatmap
    """
    # Parameters
    # time_start = time.time()
    omega = omega
    E_over_omega = np.linspace(0, E0/omega, 101)
    E = [e * omega for e in E_over_omega]
    k_vals = np.linspace(-np.pi, np.pi, 100)
    b_grid = np.linspace(0,1,number_of_step_in_b)
    winding_number_grid = np.zeros((len(b_grid), len(E)), dtype=complex)
    winding_number_band = 0
    for b_idx, b in enumerate(b_grid):
        # Create tight-binding model
        coords = [[0], [b]]   
        tb_model = TightBinding(coords, lambda_decay=1.0, lattice_constant=[1.0], nk=100, mu=0.0, relative_Hopping=relative_Hopping)     
        # Run Floquet analysis
        floquet_model = tb_model.Floquet(omegad=omega, E0=E, nt=nt, polarization=[1], data_path=f'{data_saving_root}/floquet_data_Gomez_Leon_test_b={b:.2f}/')
        energies, states = floquet_model.run(k_vals)
        winding_number_grid[b_idx]=floquet_model.winding_number(band=0)
        print(f"Winding number for b={b:.2f}: {winding_number_grid[b_idx]}")

        if save_band_structure:
            # run the following line to save the plots of band structure or when you feel not sure about the results, then checking the band closing behaviour could help verify
            floquet_model.plot_band_structure(k_vals,save_band_structure=True)

        print('')
    # Convert b_grid and E to 2D meshgrid for plotting
    B, E_mesh = np.meshgrid(b_grid, E_over_omega)

    # Plot the winding number map (real part only if complex)
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(B, E_mesh, winding_number_grid.T.real, shading='auto', cmap='viridis')
    plt.colorbar(label='Winding Number')
    plt.xlabel('Bond Length b')
    plt.ylabel(r'$E_0 / \omega$')
    plt.title(f'Floquet Winding Number Map (Band {winding_number_band}/)')
    plt.tight_layout()
    plt.show()



def test_1D_2norbs(E0 = np.linspace(0, 1, 101), omega = np.linspace(5,10, 10), nt = 21, b = 0.4, relative_Hopping = [1.5,1], save_band_structure=False, data_saving_root = 'local_data'):
    """
    Test the Gomez-Leon 2013 model but using TightBinding and FloquetBloch classes, calculate the winding number and finally plot the heatmap
    """
    # Parameters
    # time_start = time.time()
    omega = omega
    E = E0
    k_vals = np.linspace(-np.pi, np.pi, 32)
    winding_number_grid = np.zeros((len(omega), len(E)), dtype=complex)
    winding_number_band = 0
    for omg_idx, omg in enumerate(omega):
        # Create tight-binding model
        coords = [[0], [b]]   
        tb_model = TightBinding(coords, lambda_decay=1.0, lattice_constant=[1.0], nk=100, mu=0.0, relative_Hopping=relative_Hopping)      
        # Run Floquet analysis
        floquet_model = tb_model.Floquet(omegad=omg, E0=E, nt=nt, polarization=[1], 
                                         data_path=f'{data_saving_root}/floquet_data_1D_2norbs_test_omega={omg:.5f}/')
        
        energies, states = floquet_model.run(k_vals)

        winding_number_grid[omg_idx]=floquet_model.winding_number(band=0)
        print(f"Winding number for omega={omg:.2f}: {winding_number_grid[omg_idx]}")
        
        if save_band_structure:
            # run the following line to save the plots of band structure or when you feel not sure about the results, then checking the band closing behaviour could help verify
            floquet_model.plot_band_structure(k_vals,save_band_structure=True)
        

        print('')
        
    # Convert b_grid and E to 2D meshgrid for plotting
    B, E_mesh = np.meshgrid(omega, E)

    # Plot the winding number map (real part only if complex)
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(B, E_mesh, winding_number_grid.T.real, shading='auto', cmap='viridis')
    plt.colorbar(label='Winding Number')
    plt.xlabel('Driving Frequency omega')
    plt.ylabel(r'$E_0')
    plt.title(f'Floquet Winding Number Map (Band {winding_number_band}/)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    # test_Gomez_Leon_2013(E0 = 200, number_of_step_in_b = 21, nt = 21, omega = 10, relative_Hopping = [1.5,1], save_band_structure=False, data_saving_root = 'local_data_GL2013')
    # test_1D_2norbs(E0 = np.linspace(0, 50, 20), omega = np.linspace(3,7, 10), relative_Hopping = [1.5,1], nt = 21, b = 0.6, save_band_structure=False, data_saving_root = 'local_data')

    omega = 6
    # E_over_omega = np.linspace(0, E0/omega, 101)
    # E = [e * omega for e in E_over_omega]
    E = 1
    

    # b_grid = np.linspace(0,1,number_of_step_in_b)
    # winding_number_grid = np.zeros((len(b_grid), len(E)), dtype=complex)
    winding_number_band = 0
    b = 0.5
    
    coords = [[0], [b]]   
    
    tb = TightBinding(coords, lambda_decay=1.0, lattice_constant=[1.0], nk=100, mu=0.0, relative_Hopping=[1.5, 1])     
    
    # tb.discretize_b___zone(nk)
    
    # Run Floquet analysis
    floquet_model = tb.Floquet(omegad=omega, E0=E, nt=21, data_path='./')
    
    tb.E = E

    # compute Floquet-Bloch quasienergy band structure
    k_vals = np.linspace(-np.pi, np.pi, 100)
    energies, states = floquet_model.run(k_vals, fold=True)
    w = floquet_model.winding_number(band_id)
    
    floquet.run_along_path(path)
    
    np.savez(energies)
    
    print(energies)
    
    
    # winding_number = floquet_model.winding_number(band=0)
    # print(f"Winding number for b={b:.2f}: {winding_number}")

    # if save_band_structure:
    #     # run the following line to save the plots of band structure or when you feel not sure about the results, then checking the band closing behaviour could help verify
    #     floquet_model.plot_band_structure(k_vals,save_band_structure=True)

    
    