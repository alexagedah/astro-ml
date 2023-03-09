"""
The simulation module contains the Simulation class.
"""
# Standard Library
import pathlib
# 3rd Party
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
# Local
from . import convert_format, read

mpl.rcParams["font.family"] = "Times New Roman"
plt.style.use("default")

# Constants
mu_0 = constants.physical_constants["vacuum mag. permeability"][0]

class Simulation():
    """
    A simulation object represents a single accretion disc simulation.

    Parameters
    ----------
    filename : str
        The name of the file containing the simulation data

    Attributes
    ----------
    x : numpy.ndaaray
        3D numpy.ndarray representing the x-coordiantes
    y : numpy.ndarray
        3D numpy.ndarray representing the y-coordiantes
    z : numpy.ndarray
        3D numpy.ndarray representing the z-coordiantes
    t : numpy.ndarray
        1D numpy.ndarray representing the time coordinates
    fluid_variables : dict of {str:numpy.ndarray}
        Dictionary containing all the fluid variables. The keys are strings which
        specify the fluid variable and the values are 4D numpy.ndarrays which
        represent the corresponding variable.

            * The 0th axis represents the x direction
            * The 1st axis represents the y direction
            * The 2nd axis represents the z direction
            * The 3rd axis represnets the time

    chi : float
        The magnetisation for the simulation

    Methods
    -------
    get_observations(grid_size, features, response)
        Return a matrix of features and a vector of responses.
    get_feature_map(features)
        Return a dictionary mapping indicies to features
    plot_all()
        Plot the distributions of all the initialised fluid variables
    plot_distribution_at_time(variable_name, time=None, show_fig=False)
        Plot a histogram showing the distribution of a variable
    plot_all_distributions_at_time(time=None)
         Plot the distributions of all the variables at a certain time
    plot_distributions_over_time(variable_name, show_fig=False)
        Plot the distribution of a variable at various times through the
        simulation
    plot_all_distributions_over_time()
        Plot the distributions of all the variables at various times through
        the simulation
    plot_contour_at_time(self, variable_name, time, z=0, show_fig=False)
        Plot contours for a variable at a specific time
    plot_all_contours_at_time(time, z=0)
        Plot contours of all the variables at a specific time 
    plot_contours_over_time(self, variable_name, z=0, show_fig=False)
        Plot contours for a variable in a plane at various times
    plot_all_contours_over_time(z=0)
        Plot contours for all the variables in a plane at various times
    add_all_variables()
        Add all the possible additional fluid variables
    add_magnetic_field_magnitude()
        Add the magnitude of the magnetic field to the fluid variables
    add_magnetic_energy_density()
        Add the magnetic energy density to the fluid variables
    add_speed()
        Add speed to the fluid variables
    add_speed_squared()
        Add speed squared to the fluid variables
    add_plasma_beta()
        Add plasma beta the the fluid variables
    add_alfven_wave_speed()
        Add alfven wave speed to the fluid variables
    add_cross_helicity()
    """
    def __init__(self, filename):
        self.filename = filename
        self.chi = float(filename[10:].replace("dot","."))
        self.x, self.y, self.z = read.get_cell_coordinates(filename)
        self.t = read.get_time_coordinates(filename)
        self.fluid_variables = read.get_fluid_variables(filename)
        self.exploration_folder = f"exploration/{filename[5:]}/"

    # Machine Learning
    def get_observations(self, observation_size, features, response):
        """
        Return a matrix of features and a vector of responses.

        Parameters
        ----------
        snapsht : int
            The length of each 2D grid or 3D cube
        features : list of str
            The list of features to use
        response : str
            The response variable

        Returns
        -------
        X : numpy.ndarray
            5D nuumpy.ndarray representing the matrix of features

                * The 0th axis specifies the observation
                * The 1st axis represents the x-direction
                * The 2nd axis represents the y-direction
                * The 3rd axis represents the z-direction
                * The 4th axis represents the different features

        y : numpy.ndarray
            1D numpy.ndarray representing the response vector
        """
        X_t_list = []
        y_t_list = []
        for t in self.t:
            X_t = convert_format.disc_t_to_feature_matrix(self.fluid_variables,
                features,
                t,
                observation_size)
            X_t_list.append(X_t)
            if response == "time":
                y_t = np.full((X_t.shape[0], ), t)
            elif response == "chi":
                y_t = np.full((X_t.shape[0], ), self.chi)
            else:
                raise ValueError("Please choose time or chi as a response.")
            y_t_list.append(y_t)
        X = np.concatenate(X_t_list, axis = 0)
        y = np.concatenate(y_t_list, axis = 0)
        return X, y

    def get_feature_map(self, features):
        """
        Return a dictionary mapping indicies to features

        Parameters
        ----------
        features : list of str
            The list of features to use

        Returns
        -------
        feature_map : dict of {int:str}
            Dictionary where the keys are the indicies of the 4th axis of the
            matrix of features and the values are the corresponding features
            at that index. The default is {0:"B_x",1:"B_y",2:"B_z"}
        """
        indices = range(len(features))
        feature_map = dict(zip(indices, features))
        return feature_map

    # Plotting
    def save_show_plot(self, figure, plot_name, show_fig):
        """
        Save a figure to the exploration folder for the simulation and show it
        if specified

        Parameters
        ----------
        figure : matplotlib.figure
            The figure to save
        plot_name : str
            The name of the plot
        show_fig : bool
            Whether to show the figure
        """
        self.save_plot(figure, plot_name)
        if show_fig:
            plt.show()
        plt.close()

    def save_plot(self, figure, plot_name):
        """
        Save a figure to the exploration folder for the simulation

        Parameters
        ----------
        figure : matplotlib.figure
            The figure to save
        plot_name : str
            The name of the plot
        """
        save_name = self.exploration_folder + plot_name
        figure.savefig(save_name)

    def plot_all(self):
        """
        Plot the distributions of all the initialised fluid variables
        """
        method_list = [func for func in dir(self) if callable(getattr(self, func)) and func.startswith("plot_all_")]
        for method_name in method_list:
            method = getattr(self,method_name)
            method()

    def plot_distribution_at_time(self, variable_name, time=None, show_fig=False):
        """
        Plot the distribution of a variable at a time

        Parameters
        ----------
        data_set : str
            A relative path to the data set to produce histograms for
        variable : numpy.ndarray
            4D numpy.ndarray represnting the variable
        time : int, default=None
            The time to plot the histogram for. The default None plots for the
            variable across all time
        show_fig : bool, default=False
            Whether to show the figure
        """
        variable_to_plot = self.fluid_variables[variable_name]
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel(variable_name)
        ax.set_ylabel("Frequency")
        if time == None:
            ax.set_title(f"The Distribution of {variable_name}")
            ax.hist(variable_to_plot.flatten(), bins = 50)
            self.save_show_plot(fig, f"{variable_name}_distribution", show_fig)
        else:
            ax.set_title(f"The Distribution of the {variable_name} at time = {time}")
            ax.hist(variable_to_plot[:,:,:,time].flatten(), bins = 50)
            self.save_show_plot(fig, f"{variable_name}_distribution_at_{time}", show_fig)

    def plot_all_distributions_at_time(self, time=None):
        """
        Plot the distributions of all the variables at a certain time

        Parameters
        ----------
        time : int, default=None
            The time to plot the histogram for. The default None plots for the
            variable across all time
        """
        for variable_name in self.fluid_variables.keys():
            self.plot_distribution_at_time(variable_name, time)

    def plot_distributions_over_time(self, variable_name, show_fig=False):
        """
        Plot the distributions of a variable at various times through the
        simulation

        Parameters
        ----------
        variable_name : numpy.ndarray
            The name of the variabele
        show_fig : bool, default=False
            Whether to show the figure
        """
        times_to_plot = np.linspace(0, self.t[-1], 9).astype(np.int64)
        variable_to_plot = self.fluid_variables[variable_name]
        array = variable_to_plot.reshape(-1, variable_to_plot.shape[-1])[:,times_to_plot]
        mpl.rc('xtick', labelsize=4) 
        mpl.rc('ytick', labelsize=4) 
        fig, axes = plt.subplots(3,3, figsize=(16,10))
        for i in range(3):
            for j in range(3):
                time = times_to_plot[i*3+j]
                axes[i,j].hist(array[:,i*3+j], bins = 100)
                axes[i,j].set_title(f"The Distribution of {variable_name} at time = {time}", {'fontsize':10})
                axes[i,j].set_xlabel(variable_name, {'fontsize': 4})
                axes[i,j].set_ylabel("Frequency", {'fontsize': 4})
        self.save_show_plot(fig, f"{variable_name}_distributions_over_time", show_fig)

    def plot_all_distributions_over_time(self):
        """
        Plot the distributions of all the variables at various times through the
        simulation

        Parameters
        ----------
        variable_name : numpy.ndarray
            The name of the variabele
        show_fig : bool, default=False
            Whether to show the figure
        """
        for variable_name in self.fluid_variables.keys():
            self.plot_distributions_over_time(variable_name)

    def plot_contour_at_time(self, variable_name, time, z=0, show_fig=False):
        """
        Plot contours for a variable at a specific time

        Parameters
        ----------
        variable_name : str
            The name of the variable to plot contours for
        time : int
            The time to plot the contours for
        z : int, default=0
            The z-coordinate to plot the variable at. The default is 0 which plots
            the variable in the z = 0 plane. This should be used if the data set is
            2D.
        show_fig : bool, default=False
            Whether to show the figure
        """
        x = self.x
        y = self.y
        variable = self.fluid_variables[variable_name]

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title(f"{variable_name} at t = {time}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        CS = ax.contour(x[:,:,z], y[:,:,z],variable[:,:,z,time]) 
        ax.clabel(CS, inline=True, fontsize=5)
        ax.grid(True)
        self.save_show_plot(fig, f"{variable_name}_{z}_contour_at_{time}", show_fig)

    def plot_contours_over_time(self, variable_name, z=0, show_fig=False):
        """
        Plot contours for a variable in a plane at various times

        Parameters
        ----------
        variable_name : str
            The name of the variable to plot contours for
        z : int, default=0
            The z-coordinate to plot the variable at. The default is 0 which plots
            the variable in the z = 0 plane. This should be used if the data set is
            2D.
        show_fig : bool, default=False
            Whether to show the figure
        """
        x = self.x
        y = self.y
        variable = self.fluid_variables[variable_name]

        times = np.linspace(0, variable.shape[-1]-1, 9).astype(np.int64)
        mpl.rc('xtick', labelsize=4) 
        mpl.rc('ytick', labelsize=4) 
        fig, axes = plt.subplots(3,3, figsize=(16,10))
        for i in range(3):
            for j in range(3):
                time = times[i*3+j]
                axes[i,j].set_title(f"{variable_name} at time = {time}")
                axes[i,j].set_xlabel("x", {'fontsize': 4})
                axes[i,j].set_ylabel("y", {'fontsize': 4})
                CS = axes[i,j].contour(x[:,:,z], y[:,:,z],variable[:,:,z,time]) 
                axes[i,j].clabel(CS, inline=True, fontsize=5)
                axes[i,j].grid(True)
        self.save_show_plot(fig, f"{variable_name}_{z}_contours", show_fig)

    def plot_all_contours_over_time(self, z=0):
        """
        Plot contours for all the variables at various times

        Parameters
        ----------
        z : int, default=0
            The z-coordinate to plot the variable at. The default is 0 which plots
            the variable in the z = 0 plane. This should be used if the data set is
            2D.
        """
        for variable_name in self.fluid_variables.keys():
            self.plot_contours_over_time(variable_name)
    # Additional Variables
    def add_all_variables(self):
        """
        Add all the possible additional fluid variables
        """
        method_list = [func for func in dir(self) if callable(getattr(self, func)) and func.startswith("add") and not func.startswith("add_all")]
        for method_name in method_list:
            method = getattr(self,method_name)
            method()

    def add_magnetic_field_magnitude(self):
        """
        Add the magnitude of the magnetic field to the fluid variables
        """
        self.fluid_variables["B"] = np.sqrt(self.fluid_variables["B_x"]**2 
            + self.fluid_variables["B_y"]**2 
            + self.fluid_variables["B_z"]**2)

    def add_magnetic_energy_density(self):
        """
        Add the magnetic energy density to the fluid variables

        We set the permitivity of free space to 1 therefore the magnetic energy
        density is equivalent to the squared magnitude of the magnetic field
        """
        self.fluid_variables["p_B"] = (self.fluid_variables["B_x"]**2 
            + self.fluid_variables["B_y"]**2 
            + self.fluid_variables["B_z"]**2)/(2*mu_0)

    def add_speed(self):
        """
        Add speed to the fluid variables
        """
        self.fluid_variables["u"] = np.sqrt(self.fluid_variables["u_x"]**2 
            + self.fluid_variables["u_y"]**2 
            + self.fluid_variables["u_z"]**2)

    def add_speed_squared(self):
        """
        Add speed squared to the fluid variables
        """
        self.fluid_variables["u_squared"] = (self.fluid_variables["u_x"]**2 
            + self.fluid_variables["u_y"]**2 
            + self.fluid_variables["u_z"]**2)

    def add_plasma_beta(self):
        """
        Add plasma beta the the fluid variables
        """
        self.fluid_variables["beta"] = self.fluid_variables["p"]/((
            self.fluid_variables["B_x"]**2 +
            self.fluid_variables["B_y"]**2 + 
            self.fluid_variables["B_z"]**2)/(2*mu_0))

    def add_alfven_wave_speed(self):
        """
        Add Alfven wave speed to the fluid variables
        """
        self.fluid_variables["v_A"] = np.sqrt(self.fluid_variables["B_x"]**2 
            + self.fluid_variables["B_y"]**2 
            + self.fluid_variables["B_z"]**2)/np.sqrt(mu_0*self.fluid_variables["rho"])

    def add_cross_helicity(self):
        """
        Add cross-helicity to the fluid variables
        """
        self.fluid_variables["H_m"] = (self.fluid_variables["B_x"]*self.fluid_variables["u_x"] 
            + self.fluid_variables["B_y"]*self.fluid_variables["u_y"]
            + self.fluid_variables["B_z"]*self.fluid_variables["u_z"])
    

















