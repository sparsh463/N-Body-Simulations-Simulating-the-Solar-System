# N-body Sim. for obejcts of equal mass

def create_nbody_sim(N =10, mass=1e8, radius=1e5, n_step=1000, dt=0.01, create_vis=True, accel_type ='numba'):
  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd
  from numba import njit
  from mpl_toolkits.mplot3d import Axes3D
  from tqdm import tqdm
  import plotly.express as px

  if accel_type=='numba':
    import accel_numba as accel ## Acceleration script
  elif accel_type == 'python':
    import accel_python as accel


  yr = 3.15576e7
  N = N
  # Mass of each particle
  M_particle = np.ones(N)*mass # [g]

  # Setting up Initial Parameters
  radius = radius
  theta = np.random.uniform(0, 2 * np.pi, size=N)
  phi = np.random.uniform(0, np.pi, size=N)

  X = np.zeros((N, 3), dtype=np.float64)
  X[:, 0] = radius * np.sin(phi) * np.cos(theta)
  X[:, 1] = radius * np.sin(phi) * np.sin(theta)
  X[:, 2] = radius * np.cos(phi)

  V = np.zeros((N, 3), dtype=np.float64)  # Stationary particles, so initial velocities are zero

  # Integration parameters
  n_step = n_step
  dt = dt*yr # [s]

  # Create a multi-dimensional array to store pos, vel, and potential energy after every n-step
  X_values = np.ones((n_step, N, 3))
  V_values = np.ones((n_step, N, 3))

  acc = accel.get_accel(X, M_particle)

  # Create a list to store the data (to be later converted to a Pandas DF)
  df_list = []

  # Integrating this trial
  for i in tqdm(range(n_step)):
      """Calculating X, V, and PE for each n_step"""

      # First Step of Leapfrog, Updating Velocities
      V += acc * dt / 2.

      # Updating Positions
      X += V * dt

      # Updating Accelerations and Potential Energy
      acc = accel.get_accel(X, M_particle)


      # 2nd Step of Leapfrog, Update Velocities
      V += acc * dt / 2.

      # Populating the X_values, V_values, PE_values, KE_values with X, V, PE, and KE values of a particular n-step
      X_values[i, :, :] = X
      V_values[i, :, :] = V


      # Append Data for this n_step to the Data List
      for j in range(N):
          df_list.append([f'Particle {j+1}', M_particle, i, X_values[i, j, 0], X_values[i, j, 1], X_values[i, j, 2],
                          V_values[i, j, 0], V_values[i, j, 1], V_values[i, j, 2]])

  # Creating a DataFrame from the data list
  df = pd.DataFrame(df_list, columns=['particle', 'mass', 'n_steps', 'x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel', 'z_vel'])

  if create_vis==True:

    # 3D Plot
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.axes.set_xlim3d(left=-100 * radius, right=100 * radius)
    ax.axes.set_ylim3d(bottom=-100 * radius, top=100 * radius)
    ax.axes.set_zlim3d(bottom=-100 * radius, top=100 * radius)

    for j in range(N):
        ax.plot3D(X_values[:, j, 0], X_values[:, j, 1], X_values[:, j, 2], label=f'Particle {j+1}')
    plt.legend()
    plt.title("Particle Simulation - 3D Plot")
    plt.xlabel("X Position (cm)")
    plt.ylabel("Y Position (cm)")
    ax.set_zlabel("Z Position (cm)")
    plt.show()

    # Adding the Interactive 3D scatter Plot
    fig = px.scatter_3d(df, x='x_pos', y='y_pos', z='z_pos', color='particle', animation_frame='n_steps',
                        hover_name='particle', hover_data={'particle': False},
                        title="Particle Simulation",
                        labels={'x_pos': 'X Position (cm)', 'y_pos': 'Y Position (cm)', 'z_pos': 'Z Position (cm)'},
                        range_x=[-100 * radius, 100 * radius], range_y=[-100 * radius, 100 * radius], range_z=[-100 * radius, 100 * radius])

    # Adding lines to connect positions in each frame
    for j in range(N):
        particle_data = df[df['particle'] == f'Particle {j+1}']
        line_trace = px.line_3d(particle_data, x='x_pos', y='y_pos', z='z_pos')

        fig.add_traces(line_trace.data)

    # Showing the interactive plot
    fig.update_traces(textposition='top center')  # Adjust the position of the text labels
    fig.update_layout(scene=dict(zaxis=dict(range=[-100 * radius, 100 * radius])))
    fig.show()
    fig.write_html(f"file{N}.html")

# For 10 bodies 
create_nbody_sim(N = 10, mass=1e8, radius=1e5, n_step=1000, dt=0.001, create_vis=True)

# For 100 bodies
create_nbody_sim(N = 100, mass=1e8, radius=1e5, n_step=100, dt=0.01, create_vis=True)
    
