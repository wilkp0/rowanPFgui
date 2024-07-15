import numpy as np
from data_generator import get_data

def render(self, rewards, game_over=False, cascading_frame_id=None, date=None, timestep_id=None):
    """ Initializes the renderer if not already done, then compute the necessary values to be carried to the
    renderer class (e.g. sum of consumptions).

    :param rewards: list of subrewards of the last timestep (used to plot reward per timestep)
    :param game_over: True to plot a "Game over!" over the screen if game is over
    :return: :raise ImportError: pygame not found raises an error (it is mandatory for the renderer)
    """

    def initialize_renderer():
        """ initializes the pygame gui with the parameters necessary to e.g. plot colors of productions """
        pygame.init()

        # Compute an id mapping helper for line plotting
        '''
        mpcbus = self.grid.mpc['bus']
        mpcgen = self.grid.mpc['gen'] 
        '''
        mpcbus = [1,2,3,4,5,6,7,8]
        mpcgen = [1]


        # Retrieve true substations ids of origins and extremities
        # Building one line diagrams
        idx_or= [1,3,4,5,6,7,8]
        idx_ex= [2,4,5,6,7,8,1]


        # Retrieve vector of size nodes with 0 if no prod (resp load) else 1
        nodes_ids = [1,2,3,4,5,6,7,8, 6661, 6662, 6663, 6664, 6665, 6666, 6667, 6668]
        prods_ids = [1]
        
        # Retrieve vector of size nodes with 0 if no prod (resp load) else 1
        nodes_ids = [1,2,3,4,5,6,7]
        prods_ids = [0]

        #self.are_loads = np.logical_or(self.mpc['bus'][:, 2] != 0, self.mpc['bus'][:, 3] != 0)
        are_loads = [False, True, True, True, True, True, True, True,]
        #True or Flase if they are prods or loads (total 14)
        are_prods = np.logical_or([node_id in prods_ids for node_id in nodes_ids[:len(nodes_ids) // 2]],
                                    [node_id in prods_ids for node_id in nodes_ids[len(nodes_ids) // 2:]])
        are_loads = np.logical_or(are_loads[:len(mpcbus) // 2],
                                    are_loads[len(nodes_ids) // 2:])

        #total duration = first_datetime - second_datetime
        #timestep_duration_seconds = self.__chronic.get_timestep_duration()
        timestep_duration_seconds = 3600
        from rowanpf.renderer import Renderer

        numberofbuses = 8
        return Renderer(numberofbuses, idx_or, idx_ex, are_prods, are_loads,
                        timestep_duration_seconds)

    try:
        import pygame
    except ImportError as e:
        raise ImportError(
            "{}. (HINT: install pygame using `pip install pygame` or refer to this package README)".format(e))

    # if close:
    # pygame.quit()

    if self.renderer is None:
        self.renderer = initialize_renderer()

    # Retrieve lines capacity usage (for plotting power lines with appropriate colors and widths)
    #Number of lines and values  (usage) value < 1 , value >0
    lines_capacity_usage = self.grid.export_lines_capacity_usage(safe_mode=True)

    #Values of generators (production)
    prods_values = self.grid.mpc['gen'][:, 1]

    #Values for all the loads
    loads_values = self.grid._consistent_ordering_loads()(self.grid.mpc['bus'][self.grid.are_loads, 2])
    
    #Values postive and negative (20 lines in old code)
    #active_flows_origin = to_array(branch[:, 13])  # Pf
    #reactive_flows_origin = to_array(branch[:, 14])  # Qf
    #active_flows_extremity = to_array(branch[:, 15])  # Pt
    #reactive_flows_extremity = to_array(branch[:, 16])  # Qt
    lines_por_values = self.grid.mpc['branch'][:, 13]
    #Binary value 1 for ON and 0 for OFF
    lines_service_status = self.grid.mpc['branch'][:, 10]

    substations_ids = self.grid.mpc['bus'][self.grid.n_nodes // 2:]
    # Based on the action, determine if substations has been touched (i.e. there was a topological change involved
    # in the associated substation)
    if self.last_action is not None and cascading_frame_id is not None:
        has_been_changed = self.get_changed_substations(self.last_action)
    else:
        has_been_changed = np.zeros((len(substations_ids),))

    are_isolated_loads, are_isolated_prods, _ = self.grid._count_isolated_loads(self.grid.mpc, self.grid.are_loads)
    
    # Why unavailable 
    self.timesteps_before_lines_reconnectable = np.zeros((self.grid.n_lines,))
    self.timesteps_before_lines_reactionable = np.zeros((self.grid.n_lines,))
    self.timesteps_before_nodes_reactionable = np.zeros((len(self.substations_ids),))
    number_unavailable_lines = sum(self.timesteps_before_lines_reconnectable > 0) + \
                                sum(self.timesteps_before_lines_reactionable > 0)
    number_unavailable_nodes = sum(self.timesteps_before_nodes_reactionable > 0)

    #Arrays of zeros
    initial_topo = self.initial_topology
    current_topo = self.grid.get_topology()
    #Single value
    distance_ref_grid = sum(np.asarray(initial_topo.get_zipped()) != np.asarray(current_topo.get_zipped()))

    #Values 0,1 respectively
    max_number_isolated_loads = self.__parameters.get_max_number_loads_game_over()
    max_number_isolated_prods = self.__parameters.get_max_number_prods_game_over()

    # Compute the number of used nodes per substation
    #Only one node per substation in this occurance 
    current_observation = self.export_observation()
    n_nodes_substations = []
    for substation_id in self.substations_ids:
        substation_conf = current_observation.get_nodes_of_substation(substation_id)[0]
        n_nodes_substations.append(1 + int(len(list(set(substation_conf))) == 2))

    self.renderer.render(lines_capacity_usage, lines_por_values, lines_service_status, self.epoch, self.timestep,
                            self.current_timestep_id if not timestep_id else timestep_id, prods=prods_values,
                            loads=loads_values, 
                            date=self.current_date if date is None else date,
                            are_substations_changed=has_been_changed, 
                            number_nodes_per_substation=n_nodes_substations,
                            number_loads_cut=sum(are_isolated_loads), 
                            number_prods_cut=sum(are_isolated_prods),
                            number_nodes_splitting= 0,
                            number_lines_switches= 0, 
                            distance_initial_grid=distance_ref_grid,
                            number_off_lines=sum(self.grid.get_lines_status() == 0),
                            number_unavailable_lines=number_unavailable_lines,
                            number_unactionable_nodes=number_unavailable_nodes,
                            max_number_isolated_loads=max_number_isolated_loads,
                            max_number_isolated_prods=max_number_isolated_prods, 
                            game_over=game_over,
                            cascading_frame_id=cascading_frame_id)

    if self.latency:
        sleep(self.latency)




#Running the Visuals with a ransom dataset
def run(iterations):

    data = get_data()
    lines_capacity_usage = data['lines_capacity_usage']
    lines_por_values = data['lines_por_values']
    lines_service_status = data['lines_service_status']
    epoch = data['epoch'], timestep = data['timestep'] #Check from loop
    current_timestep_id = data['current_timestep_id'],timestep_id = data['timestep_id'] #Check from loop
    prods_values = data['prods_values']
    loads_values = data['loads_values']
    current_date = data['current_date'] #Check from loop
    has_been_changed = data['has_been_changed']
    n_nodes_substations = data['n_nodes_substations']
    are_isolated_loads = data['are_isolated_loads']
    are_isolated_prods = data['are_isolated_prods']
    number_nodes_splitting = data['number_nodes_splitting'] #This is action of splitting, no splitting keep constant / Zero for no action
    number_lines_switches = data['number_lines_switches']
    distance_ref_grid = data['distance_ref_grid']
    number_off_lines = data['number_off_lines']
    number_unavailable_lines = data['number_unavailable_lines']
    max_number_isolated_loads = data['max_number_isolated_loads']
    max_number_isolated_prods = data['max_number_isolated_prods']
    game_over = data['game_over']
    cascading_frame_id = data['cascading_frame_id']

    for i_iter in range(iterations):
        render()
