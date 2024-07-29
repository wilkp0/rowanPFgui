import numpy as np 
from time import sleep



class Display():
    def __init__(self):
        self.renderer = None 
        self.latency = 0.1
        self.timestep = 0


    def render(self, game_over=False, cascading_frame_id=None, date=None, timestep_id=None):
        """ Initializes the renderer if not already done, then compute the necessary values to be carried to the
        renderer class (e.g. sum of consumptions).

        :param rewards: list of subrewards of the last timestep (used to plot reward per timestep)
        :param game_over: True to plot a "Game over!" over the screen if game is over
        :return: :raise ImportError: pygame not found raises an error (it is mandatory for the renderer)
        """
        def initialize_renderer():
            mpc = {}
            ARTIFICIAL_NODE_STARTING_STRING = '666'
            mpcbus = [1,2,3,4,5,6,7,8, 6661, 6662, 6663, 6664, 6665, 6666, 6667, 6668]
            nodes_or_ids = np.array([1,3,4,5,6,7,8])
            nodes_ex_ids = np.array([2,4,5,6,7,8,2])
            mpc = {}
            mpc['branch'] = np.array([
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 8],
                [8, 2]
            ])

            mpc['bus'] = np.array([
                [1, 3, 0, 0, 0, 0, 1, 1.06, 0, 100, 1, 1.06, 0.94],
                [2, 2, 21.7, 12.7, 0, 0, 1, 1.045, 0, 100, 1, 1.06, 0.94],
                [3, 2, 94.2, 19, 0, 0, 1, 1.01, 0, 100, 1, 1.06, 0.94],
                [4, 1, 47.8, -3.9, 0, 0, 1, 1.019, 0, 100, 1, 1.06, 0.94],
                [5, 1, 7.6, 1.6, 0, 0, 1, 1.02, 0, 100, 1, 1.06, 0.94],
                [6, 2, 11.2, 7.5, 0, 0, 1, 1.07, 0, 100, 1, 1.06, 0.94],
                [7, 1, 0, 0, 0, 0, 1, 1.062, 0, 100, 1, 1.06, 0.94],
                [8, 2, 0, 0, 0, 0, 1, 1.09, 0, 100, 1, 1.06, 0.94],
                [6661, 4, 0, 0, 0, 0, 1, 1.06, 0, 100, 1, 1.06, 0.94],
                [6662, 4, 0, 0, 0, 0, 1, 1.045, 0, 100, 1, 1.06, 0.94],
                [6663, 4, 0, 0, 0, 0, 1, 1.01, 0, 100, 1, 1.06, 0.94],
                [6664, 4, 0, 0, 0, 0, 1, 1.019, 0, 100, 1, 1.06, 0.94],
                [6665, 4, 0, 0, 0, 0, 1, 1.02, 0, 100, 1, 1.06, 0.94],
                [6666, 4, 0, 0, 0, 0, 1, 1.07, 0, 100, 1, 1.06, 0.94],
                [6667, 4, 0, 0, 0, 0, 1, 1.062, 0, 100, 1, 1.06, 0.94],
                [6668, 4, 0, 0, 0, 0, 1, 1.09, 0, 100, 1, 1.06, 0.94]])

            mpc['gen'] = np.array([
                [1, 2]
            ])

            mpcbus = mpc['bus']
            half_nodes_ids = mpcbus[:len(mpcbus) // 2, 0]
            node_to_substation = lambda x: int(float(str(x).replace(ARTIFICIAL_NODE_STARTING_STRING, '')))
            nodes_or_ids = np.asarray(list(map(node_to_substation, mpc['branch'][:, 0])))
            nodes_ex_ids = np.asarray(list(map(node_to_substation, mpc['branch'][:, 1])))
            idx_or = [np.where(half_nodes_ids == or_id)[0][0] for or_id in nodes_or_ids]
            idx_ex = [np.where(half_nodes_ids == ex_id)[0][0] for ex_id in nodes_ex_ids]

            #print("idx_or",idx_or)
            #print("idx_ex",idx_ex)
            # Retrieve vector of size nodes with 0 if no prod (resp load) else 1
            mpcgen = mpc['gen']
            nodes_ids = mpcbus[:, 0]
            prods_ids = mpcgen[:, 0]

            #nodes_ids = [1,2,3,4,5,6,7,8, 6661, 6662, 6663, 6664, 6665, 6666, 6667, 6668]
            #prods_ids = [1]

            #print("nodes_ids",nodes_ids)
            #print("prods_ids",nodes_ids)

            #self.are_loads = np.logical_or(self.mpc['bus'][:, 2] != 0, self.mpc['bus'][:, 3] != 0)
            are_loads = [False, True, True, True, True, True, True, True, False,False,False,False,False,False,False, False]
            #True or Flase if they are prods or loads (total 14)
            are_prods = np.logical_or([node_id in prods_ids for node_id in nodes_ids[:len(nodes_ids) // 2]],
                                        [node_id in prods_ids for node_id in nodes_ids[len(nodes_ids) // 2:]])
            are_loads = np.logical_or(are_loads[:len(nodes_ids) // 2],
                                        are_loads[len(nodes_ids) // 2:])
            print("are_prods", are_prods)
            print("loads", are_loads)
            timestep_duration_seconds = 3600

            from renderer import Renderer
            #from renderer import Renderer
            number_of_nodes = 8
            #print("len(self.grid.number_elements_per_substations)",len(self.grid.number_elements_per_substations))
            return Renderer(number_of_nodes, idx_or, idx_ex, are_prods, are_loads,
                            timestep_duration_seconds)
        
        if self.renderer is None:
            self.renderer = initialize_renderer()
        
        from data_generator import get_data
        lines_capacity_usage, lines_por_values, lines_service_status, epoch, timestep, current_timestep_id, \
        prods, loads, date, are_substations_changed, number_nodes_per_substation, number_loads_cut, \
        number_prods_cut, number_nodes_splitting, number_lines_switches, distance_initial_grid, number_off_lines, \
        number_unavailable_lines, number_unactionable_nodes, max_number_isolated_loads, max_number_isolated_prods, \
        game_over, cascading_frame_id = get_data(time_counter=self.timestep)

        self.timestep += 1

        self.renderer.render(lines_capacity_usage, lines_por_values, lines_service_status, epoch, timestep, current_timestep_id,
                            prods, loads, date, are_substations_changed, number_nodes_per_substation, number_loads_cut,
                            number_prods_cut, number_nodes_splitting, number_lines_switches, distance_initial_grid,
                            number_off_lines, number_unavailable_lines, number_unactionable_nodes, max_number_isolated_loads,
                            max_number_isolated_prods, game_over, cascading_frame_id)

        if self.latency:
            sleep(self.latency)


iterations = 100000
display = Display()
for i_iter in range(iterations):
    display.render()