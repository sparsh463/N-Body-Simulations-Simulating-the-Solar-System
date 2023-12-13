## Do runs with accel_numba

import cProfile
import pstats

def main():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from astropy import units as u
    from astropy.time import Time
    from astropy.coordinates import get_body_barycentric_posvel
    import accel_numba as accel ## Acceleration script
    from mpl_toolkits.mplot3d import Axes3D
    from tqdm import tqdm
    import plotly.express as px 

    yr = 3.15576e7     # [s]     year

    # Celestial Objects List
    objects = ['sun', 'mercury', 'venus', 'earth', 'moon', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto']

    # Initialising an Array of Positions, Velocities and Mass
    X = np.ones((len(objects), 3))  # [cm]
    V = np.ones((len(objects), 3))  # [cm]
    M = np.array([1.989e33, 3.285e26, 4.867e27, 5.9742e27, 7.348e25, 6.4e26, 1.89e30, 5.68e29, 8.68e28, 1.024e29, 1.31e28])  # [g]

    # Setting a start time for the simulation 
    # (astropy uses this time to determine pos, vel)
    time = Time('2023-10-15 00:00') 

    # Getting barycentric position and velocities and filling the X,V arrays.
    for i, body in enumerate(objects):
        """Populate X and V arrays with pos and vel values of each solar 
        system object using astropy's get_body_.... function"""
        
        pos, vel = get_body_barycentric_posvel(body, time, ephemeris='jpl')
        X[i, :] = pos.xyz.to(u.cm).value  # [cm]
        V[i, :] = vel.xyz.to(u.cm/u.s).value

    # Integration parameters
    n_step = 1000
    dt = 0.01 * yr  # [s]
    acc = accel.get_accel(X, M)

    ## Creating a multi-dimensional array to store pos, vel after every n-step
    X_values = np.ones((n_step, len(objects), 3))
    V_values = np.ones((n_step, len(objects), 3))

    # Create a list to store the data (to be later converted to a Pandas DF)
    df_list = []

    # Integrating this trial
    for i in tqdm(range(n_step)):
        
        """Calcualting X and V for each n_step"""
        
        # First Step of Leapfrog, Updating Velocities
        V += acc * dt / 2.

        # Updating Positions
        X += V * dt

        # Updating Accelerations
        acc = accel.get_accel(X, M)

        # 2nd Step of Leapfrog, Update Velocities
        V += acc * dt / 2.

        ## Populating the X_values and V_values with X and V values of a particualr n-step
        X_values[i, :, :] = X
        V_values[i, :, :] = V

        # Append Data for this n_step to the Data List
        for j in range(len(objects)):
            df_list.append([objects[j], M[j], i, X_values[i, j, 0], X_values[i, j, 1], X_values[i, j, 2], V_values[i, j, 0], V_values[i, j, 1], V_values[i, j, 2]])

    # Creating a DataFrame from the data list
    df = pd.DataFrame(df_list, columns=['object', 'mass', 'n_steps', 'x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel', 'z_vel'])

    df.to_csv("Simulation_Results.csv")

    "3D Plot"

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.axes.set_xlim3d(left= -5e14, right= 5e14)
    ax.axes.set_ylim3d(bottom= -5e14, top= 5e14)
    ax.axes.set_zlim3d(bottom= -5e14, top= 5e14)

    for i in range(len(objects)):
        ax.plot3D(X_values[:,i,0],X_values[:,i,1], X_values[:,i,2], label = objects[i])
    plt.legend()
    plt.show()

    "Making an Interactive 3D Plot with Plotly"

    ## Defining Color Map

    color_map = {
        'sun': 'yellow',
        'mercury': 'gray',
        'venus': 'orange',
        'earth': 'blue',
        'moon': 'gray',
        'mars': 'red',
        'jupiter': 'orange',
        'saturn': 'brown',
        'uranus': 'violet',
        'neptune': 'blue',
        'pluto':'gray'
    }

    print(df)

    ## Adding the Interactive 3D scatter Plot
    fig = px.scatter_3d(df, x='x_pos', y='y_pos', z='z_pos', color='object', animation_frame='n_steps', text='object',
                        title="Solar System Simulation", labels={'x_pos': 'X Position', 'y_pos': 'Y Position', 'z_pos': 'Z Position'},
                        range_x=[-1e15, 1e15], range_y=[-1e15, 1e15], range_z=[-1e15, 1e15],
                        color_discrete_map=color_map)

    # Adding lines to connect positions in each frame
    for obj in df['object'].unique():
        obj_data = df[df['object'] == obj]
        line_trace = px.line_3d(obj_data, x='x_pos', y='y_pos', z='z_pos')
        
        # Set the line style (e.g., dash pattern) and line color based on the color_map
        line_trace.update_traces(line=dict(dash='dot', color=color_map[obj]))
        
        fig.add_traces(line_trace.data)

    # Showing the interactive plot
    fig.update_traces(textposition='top center')  # Adjust the position of the text labels
    fig.show()
    
if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    
    main()  # Call the main function of your script

    profiler.disable()
    profiler.dump_stats('output_numba.pstats')  # Save stats to a file

p = pstats.Stats('output_numba.pstats')
p.strip_dirs().sort_stats(-1).print_stats()
p.sort_stats(SortKey.CUMULATIVE).print_stats(30)
