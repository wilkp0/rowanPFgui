import numpy as np
from datetime import datetime, timedelta


'''
Change this to import data
right now only imports the timestep of simulation 
Depedning on data format Read data
Also the netwrk is hard coded, Use scipy.io loadmat to read matlab networks
'''
def get_data(time_counter):
    number_of_lines = 20
    number_of_nodes = 8
    number_of_gen = 1
    number_of_loads = 7
    date = datetime(2012, 1, 1, 2, 0, 0)
    """ Computes and returns the lines capacity usage, i.e. the elementwise division of the flows in Ampere by the lines nominal thermal limit."""
    lines_capacity_usage = np.random.uniform(0.1, 0.2, size=number_of_lines)
    #Real power flow from origin  -> active_flows_origin = to_array(branch[:, 13])  # Pf
    lines_por_values = np.full(number_of_lines, 3.5)
    #Status line operation, returns binary value for each line
    lines_service_status = np.ones(number_of_lines)
    #Dont care about this rn  1 episode
    epoch = 1 
    # Timestep of simulation 
    timestep = time_counter
    current_timestep_id = timestep  #for now this is correct because no sim #if not timestep_id else timestep_id
    #Pg power generation for each gen
    prods = np.full(number_of_gen, 3.5)
    #Pd load for each bus only if the bus has a load e.g. 20 busses and 11 loads
    #loads = np.ones(7)
    loads = [0,1,1,1,1,1,1]

    number_of_buses = 7
    number_of_loads = 7
    loads = np.zeros(number_of_buses)

    # Time counter representing the hour of the day
    timestep = time_counter % 24  # Assuming time_counter cycles through each hour of the day

    # Sine wave parameters
    midday = 12  # Midday is the peak
    amplitude = 0.45  # Adjusted amplitude so the peak is at 1 (0.45 + 0.1 + 0.45)
    vertical_shift = 0.55  # Shift up by 0.55 to ensure the minimum is 0.1
    frequency = 2 * np.pi / 24  # One full cycle over 24 hours

    # Calculate the load at the current timestep using a shifted sine wave
    sine_wave_load = amplitude * np.sin(frequency * (timestep - midday)) + vertical_shift
    # Ensuring the load values are non-negative (this is a safeguard and might not be necessary here)
    sine_wave_load = np.maximum(sine_wave_load, 0.1)

    # Assign the load to the buses with loads
    for i in range(number_of_loads):
        loads[i] = sine_wave_load
        loads[0] = 0

    #Importing date and time from the data based on timestep_id 2012-01-01 02:00:00
    date += timedelta(seconds=timestep)
    """ Computes the boolean array of changed substations from an Action (Not used)"""
    are_substations_changed = np.full(number_of_nodes, False) 
    #Nodes per a substation, 1 for each node
    number_nodes_per_substation= np.ones(number_of_nodes) 
    #If load/gen is isolated or not 
    number_loads_cut= 0 #sum(are_isolated_loads)
    number_prods_cut= 0 #sum(are_isolated_prods)
    number_nodes_splitting = 0
    number_lines_switches = 0
    distance_initial_grid = 0 #distance_ref_grid
    number_off_lines = 0 # sum(get_lines_status() == 0)
    number_unavailable_lines= 0 #number_unavailable_lines,
    number_unactionable_nodes= 0 #number_unavailable_nodes,
    max_number_isolated_loads= 0 #max_number_isolated_loads,
    max_number_isolated_prods= 0 #max_number_isolated_prods, 
    game_over = False
    #None or -1 
    cascading_frame_id= None #-1 #cascading_frame_id

    """ Retrieves the current index of scenario; this index might differs from a natural counter (some id may be
    missing within the chronic).  timestep of simulation dont line up with the index e.g. if reset after fail etc.
    :return: an integer of the id of the current scenario loaded
    """

    return lines_capacity_usage, lines_por_values, lines_service_status, epoch, timestep, current_timestep_id, \
    prods, loads, date, are_substations_changed, number_nodes_per_substation, number_loads_cut,\
    number_prods_cut, number_nodes_splitting, number_lines_switches, distance_initial_grid, number_off_lines,\
    number_unavailable_lines, number_unactionable_nodes, max_number_isolated_loads, max_number_isolated_prods, \
    game_over, cascading_frame_id




