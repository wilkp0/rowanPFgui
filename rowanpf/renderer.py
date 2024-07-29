import pygame
import math
import numpy as np
import matplotlib
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import pylab
from copy import deepcopy
import pygame.gfxdraw


matplotlib.use("Agg")

case_layouts = {
    8: [(50, 500), (150, 500), (350, 375), (500, 375), (600, 250), (475, 175), (300, 315), (50, 350)]
}

class Renderer(object):
    def __init__(self, grid_case, or_ids, ex_ids, are_prods, are_loads, timestep_duration_seconds):

        self.video_width, self.video_height = 1300, 700
        pygame.init()
        self.timestep_duration_seconds = timestep_duration_seconds
        info = pygame.display.Info()
        #self.video_width, self.video_height = info.current_w, info.current_h

        #Create screen    
        self.screen = pygame.display.set_mode((self.video_width, self.video_height), pygame.RESIZABLE)
        pygame.display.set_caption('Rowan Grid Renderer')
        self.background_color = [70, 70, 73]
        self.screen.fill(self.background_color)

        #Grid case is number of nodes
        self.grid_case = grid_case
        self.grid_layout = np.asarray(case_layouts[grid_case])


        topology_width = 0.77
        self.topology_layout_shape = [int(self.video_width*topology_width), self.video_height]
        self.topology_layout = pygame.Surface(self.topology_layout_shape, pygame.SRCALPHA, 32).convert_alpha()
        self.nodes_surface = pygame.Surface(self.topology_layout_shape, pygame.SRCALPHA, 32).convert_alpha()
        self.left_menu_shape = [int(self.video_width*.231), self.video_height]
        self.left_menu = pygame.Surface(self.left_menu_shape, pygame.SRCALPHA, 32).convert_alpha()
        self.left_menu_tile_color = [e + 10 for e in self.background_color]
        
        #self.injections_surface = pygame.Surface(self.topology_layout_shape, pygame.SRCALPHA, 32).convert_alpha()
        #self.map_surface = pygame.Surface(self.topology_layout_shape, pygame.SRCALPHA, 32).convert_alpha()
        #self.lines_labels_surface = pygame.Surface(self.topology_layout_shape, pygame.SRCALPHA, 32).convert_alpha()
        #self.lines_surface = pygame.Surface(self.topology_layout_shape, pygame.SRCALPHA, 32).convert_alpha()


        self.nodes_outer_radius = 8
        self.nodes_inner_radius = 5
        self.are_prods = are_prods
        self.are_loads = are_loads
        self.lines_ids_or = or_ids
        self.lines_ids_ex = ex_ids


        pygame.font.init()
        font = 'Arial'
        self.default_font = pygame.font.SysFont(font, 15)
        text_color = (180, 180, 180)
        value_color = (220, 220, 220)
        self.text_render = lambda s: self.default_font.render(s, False, text_color)
        self.value_render = lambda s: self.default_font.render(s, False, value_color)
        big_value_font = pygame.font.SysFont('Arial', 18)
        self.big_value_render = lambda s: big_value_font.render(s, False, value_color)

        self.bold_white_font = pygame.font.SysFont(font, 15)
        bold_white = (220, 220, 220)
        self.bold_white_font.set_bold(True)
        self.bold_white_render = lambda s: self.bold_white_font.render(s, False, bold_white)

        self.black_bold_font = pygame.font.SysFont(font, 15)
        blackish = (70, 70, 70)
        self.black_bold_font.set_bold(True)
        self.black_bold_font_render = lambda s: self.black_bold_font.render(s, False, blackish)

        self.last_rewards_surface = None
        self.game_over_surface = self.draw_plot_game_over()
        self.boolean_dynamic_arrows = True

        self.data = None
        self.loads = []
        self.relative_thermal_limits = []

        #Import images 
        self.wind_img = Image.open('windtb.png').convert('RGBA')
        self.building_img = Image.open('wbuildingload.png').convert('RGBA')
        self.substation_img = Image.open('substation.png').convert('RGBA')
        self.wind_img_nc = Image.open('windtb.png').convert('RGBA')

    
    def draw_surface_nodes_headers(self, scenario_id, date, cascading_result_frame):
        surface = self.nodes_surface

        # Scaling factors
        scale_x = self.video_width / 1300
        scale_y = self.video_height / 700

        # Adjust positions and sizes dynamically
        date_pos = (int(25 * scale_x), int(15 * scale_y))
        date_value_pos = (int(75 * scale_x), int(12 * scale_y))
        timestep_id_pos = (int(330 * scale_x), int(15 * scale_y))
        timestep_value_pos = (int(425 * scale_x), int(12 * scale_y))

        width = int(400 * scale_x)
        height = int(25 * scale_y)
        x_offset = int(25 * scale_x)
        y_offset = int(40 * scale_y)

        surface.blit(self.text_render('Date'), date_pos)
        surface.blit(self.big_value_render(date.strftime("%A %d %b  %H:%M")), date_value_pos)
        surface.blit(self.text_render('Timestep id'), timestep_id_pos)
        surface.blit(self.big_value_render(str(scenario_id)), timestep_value_pos)

        # Adjust header position and size dynamically
        pygame.gfxdraw.filled_polygon(surface,
                                    ((x_offset, y_offset + height), 
                                    (x_offset, y_offset), 
                                    (x_offset + width, y_offset), 
                                    (x_offset + width, y_offset + height)),
                                    (200, 250, 200, 240))
        
        header_text_pos = (x_offset + int(120 * scale_x), y_offset + int(4 * scale_y))
        surface.blit(self.black_bold_font_render('Rowan University Campus'), header_text_pos)


    def create_plot_loads_curve(self, n_timesteps, left_xlabel):
        facecolor_asfloat = np.asarray(self.left_menu_tile_color) / 255.
        layout_config = {'pad': 0.2}

        # Scaling factors
        scale_x = self.video_width / 1300
        scale_y = self.video_height / 700

        # Adjust figsize dynamically
        fig_width = 3 * scale_x
        fig_height = 1.5 * scale_y

        fig = pylab.figure(figsize=[fig_width, fig_height], dpi=100, facecolor=facecolor_asfloat, tight_layout=layout_config)
        ax = fig.gca()
        data = np.sum(self.loads, axis=-1)
        data = data[-min(len(data), n_timesteps):]
        n_data = len(data)
        ax.plot(np.linspace(n_data, 0, num=n_data), data, '#d24dff')
        ax.set_xlim([n_timesteps, 1])
        ax.set_xticks([1, n_timesteps])
        ax.set_xticklabels(['now', left_xlabel])
        ax.set_ylim([0, np.max(data) * 1.05])
        ax.set_yticks([0, np.max(data)])
        ax.set_yticklabels(['', '%.0f MW' % (np.max(data))])
        label_color_hexa = '#D2D2D2'
        ax.tick_params(axis='y', labelsize=6, pad=-30, labelcolor=label_color_hexa, direction='in')
        ax.tick_params(axis='x', labelsize=6, labelcolor=label_color_hexa)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_facecolor(np.asarray(self.background_color) / 255.)
        fig.tight_layout()

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()

        return pygame.image.fromstring(raw_data, size, "RGB")
    
    def draw_surface_loads_curves(self, n_hours_to_display_top_loadplot, n_hours_to_display_bottom_loadplot):
        n_days_horizon = n_hours_to_display_top_loadplot // 24
        img_loads_curve_week = self.create_plot_loads_curve(
            n_timesteps=int(n_hours_to_display_top_loadplot * 3600 // self.timestep_duration_seconds),
            left_xlabel=' {} day{} ago  '.format(n_days_horizon, 's' if n_days_horizon > 1 else ''))
        n_hours_horizon = n_hours_to_display_bottom_loadplot
        img_loads_curve_day = self.create_plot_loads_curve(
            n_timesteps=int(n_hours_to_display_bottom_loadplot * 3600 // self.timestep_duration_seconds),
            left_xlabel='{} hours ago'.format(n_hours_horizon))
        

        # Dynamically calculate surface height
        surface_height = 2 * img_loads_curve_week.get_height() + int(30 * (self.video_height / 700))
        loads_curve_surface = pygame.Surface(
            (img_loads_curve_week.get_width(), surface_height),
            pygame.SRCALPHA, 32).convert_alpha()
        loads_curve_surface.fill(self.left_menu_tile_color)

        # Adjust label positions dynamically
        title_x = int(30 * (self.video_width / 1300))
        title_y = int(10 * (self.video_height / 700))
        loads_curve_surface.blit(self.bold_white_render('Historical total consumption'), (title_x, title_y))
    
        curve_y_spacing = int(30 * (self.video_height / 700))
        loads_curve_surface.blit(img_loads_curve_week, (0, curve_y_spacing))
        loads_curve_surface.blit(img_loads_curve_day, (0, curve_y_spacing + img_loads_curve_week.get_height()))
        
        pygame.gfxdraw.hline(loads_curve_surface, 0, loads_curve_surface.get_width(), 0, (64, 64, 64))
        pygame.gfxdraw.hline(loads_curve_surface, 0, loads_curve_surface.get_width(), loads_curve_surface.get_height() - 1,
                    (64, 64, 64))
        return loads_curve_surface

    def draw_surface_relative_thermal_limits(self, n_timesteps, left_xlabel='24 hours ago'):
        facecolor_asfloat = np.asarray(self.left_menu_tile_color) / 255.
        layout_config = {'pad': 0.2}
        # Scaling factors
        scale_x = self.video_width / 1300
        scale_y = self.video_height / 700

        # Adjust figsize dynamically
        fig_width = 3 * scale_x
        fig_height = 1.5 * scale_y

        fig = pylab.figure(figsize=[fig_width, fig_height], dpi=100, facecolor=facecolor_asfloat, tight_layout=layout_config)
        ax = fig.gca()
        data = self.relative_thermal_limits
        data = data[-min(len(data), n_timesteps):]
        n_data = len(data)
        medians = np.median(data, axis=-1)
        p25 = np.percentile(data, 25, axis=-1)
        p75 = np.percentile(data, 75, axis=-1)
        p90 = np.percentile(data, 90, axis=-1)
        p10 = np.percentile(data, 10, axis=-1)
        maxes = np.max(data, axis=-1)
        mines = np.min(data, axis=-1)
        ax.fill_between(np.linspace(n_data, 0, num=n_data), p10, p90, color='#16AA16')
        ax.fill_between(np.linspace(n_data, 0, num=n_data), p25, p75, color='#16DC16')
        ax.plot(np.linspace(n_data, 0, num=n_data), medians, '#AAFFAA')
        ax.plot(np.linspace(n_data, 0, num=n_data), maxes, '#339966', '.', linewidth=.75)
        ax.plot(np.linspace(n_data, 0, num=n_data), mines, '#339966', '.', linewidth=.75)
        ax.set_xlim([n_timesteps, 1])
        ax.set_xticks([1, n_timesteps])
        ax.set_xticklabels(['now', left_xlabel])
        ax.set_ylim([0, max(1.05, min(2., np.max([medians, p90, p10]) * 1.05))])
        ax.set_yticks([0, .2, .4, .6, .8, 1])
        ax.set_yticklabels(['', '20%  ', '', '60%  ', '', '100%'])
        label_color_hexa = '#D2D2D2'
        ax.tick_params(axis='y', labelsize=6, pad=-23, labelcolor=label_color_hexa, direction='in')
        ax.tick_params(axis='x', labelsize=6, labelcolor=label_color_hexa)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_facecolor(np.asarray(self.background_color) / 255.)
        fig.tight_layout()

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()

        img_rtl = pygame.image.fromstring(raw_data, size, "RGB")
        # Dynamically calculate surface height
        surface_height = 2 * img_rtl.get_height() + int(30 * scale_y)
        rtl_curves_surface = pygame.Surface((img_rtl.get_width(), surface_height),
                                            pygame.SRCALPHA, 32).convert_alpha()
        rtl_curves_surface.fill(self.left_menu_tile_color)
        rtl_curves_surface.blit(self.bold_white_render('Last 24h lines capacity usage'), (30, 10))
#CLOSE MIGHT NEED MORE
        curve_y_spacing = int(30 * scale_y)
        rtl_curves_surface.blit(img_rtl, (0, curve_y_spacing))
        pygame.gfxdraw.hline(rtl_curves_surface, 0, rtl_curves_surface.get_width(), 0, (64, 64, 64))
        pygame.gfxdraw.hline(rtl_curves_surface, 0, rtl_curves_surface.get_width(), rtl_curves_surface.get_height() - 1,
                      (64, 64, 64))

        return rtl_curves_surface

    def draw_surface_n_overflows(self, n_timesteps, left_xlabel=' 7 days ago  '):
        facecolor_asfloat = np.asarray(self.left_menu_tile_color) / 255.
        layout_config = {'pad': 0.2}

        # Scaling factors
        scale_x = self.video_width / 1300
        scale_y = self.video_height / 700

        # Adjust figsize dynamically
        fig_width = 3 * scale_x
        fig_height = 1.2 * scale_y

        fig = pylab.figure(figsize=[fig_width, fig_height], dpi=100, facecolor=facecolor_asfloat, tight_layout=layout_config)
        ax = fig.gca()
        data = np.sum(np.asarray(self.relative_thermal_limits) >= 1., axis=-1)
        data = data[-min(len(data), n_timesteps):]
        n_data = len(data)
        ax.plot(np.linspace(n_data, 0, num=n_data), data, '#ff3333')
        ax.set_xlim([n_timesteps, 1])
        ax.set_xticks([1, n_timesteps])
        ax.set_xticklabels(['now', left_xlabel])
        ax.set_ylim([0, max(1, np.max(data) * 1.05)])
        ax.set_yticks([0, max(1, np.max(data))])
        ax.set_yticklabels(['', '%d' % max(1, np.max(data))])
        label_color_hexa = '#D2D2D2'
        ax.tick_params(axis='y', labelsize=6, pad=-12, labelcolor=label_color_hexa, direction='in')
        ax.tick_params(axis='x', labelsize=6, labelcolor=label_color_hexa)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_facecolor(np.asarray(self.background_color) / 255.)
        fig.tight_layout()

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()

        img_rtl = pygame.image.fromstring(raw_data, size, "RGB")

        # Dynamically calculate surface height
        surface_height = 2 * img_rtl.get_height() + int(30 * scale_y)
        n_overflows_surface = pygame.Surface((img_rtl.get_width(), surface_height),
                                            pygame.SRCALPHA, 32).convert_alpha()
        
        n_overflows_surface.fill(self.left_menu_tile_color)
        n_overflows_surface.blit(self.bold_white_render('Number of overflows'), (30, 10))

        curve_y_spacing = int(30 * scale_y)
        n_overflows_surface.blit(img_rtl, (0, curve_y_spacing))

        pygame.gfxdraw.hline(n_overflows_surface, 0, n_overflows_surface.get_width(), 0, (64, 64, 64))
        pygame.gfxdraw.hline(n_overflows_surface, 0, n_overflows_surface.get_width(), n_overflows_surface.get_height() - 1,
                      (64, 64, 64))

        return n_overflows_surface
    


    
    def draw_surface_grid(self, relative_thermal_limits, lines_por, lines_service_status, prods, loads,
                        are_substations_changed, number_nodes_per_substation):
        
        layout = self.grid_layout

        # Scaling factors
        scale_x = self.video_width / 1300
        scale_y = self.video_height / 700

        # Adjust figsize dynamically
        fig_width = 1000 * scale_x / 200
        fig_height = 700 * scale_y / 200
        my_dpi = 200

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=my_dpi,
                        facecolor=[c / 255. for c in self.background_color], clear=True)
        

        # Create a new Axes for the map image with a low zorder
        map_ax = fig.add_axes([0, 0, 1, 1], frame_on=False, zorder=1)
        map_ax.set_xlim(0, 1000 * scale_x)
        map_ax.set_ylim(-50 * scale_y, 650 * scale_y)
        map_ax.set_xticks([])
        map_ax.set_yticks([])
        # Load the rowan_map image
        map_img_path = 'RowanMap.png'
        map_img = Image.open(map_img_path).convert('RGBA')
        base_width = int(800 * scale_x)
        w_percent = (base_width / float(map_img.size[0]))
        h_size = int((float(map_img.size[1]) * float(w_percent)))

        # Resize the image
        map_img = map_img.resize((base_width, h_size), Image.Resampling.LANCZOS)
        map_array = np.array(map_img)

        # Add the image to the map_ax
        map_ax.imshow(map_array, extent=[0, base_width, -50 * scale_y, h_size - 50 * scale_y])
    
        #ax = fig.gca(frame_on=False, autoscale_on=False, zorder=10)
        ax = fig.add_axes([0, 0, 1, 1], frame_on=False, zorder=10)
        ax.set_xlim(0, 1000 * scale_x)
        ax.set_ylim(-50 * scale_y, 650 * scale_y)
        fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        ax.set_xticks([])
        ax.set_yticks([])

        l = []
        layout = np.asarray(deepcopy(layout))
        min_x = np.min(layout[:, 0])
        min_y = np.min(layout[:, 1])
        layout[:, 0] -= min_x * 0
        layout[:, 1] -= min_y * 0
        color_low = np.asarray((51, 204, 51))
        color_middle = np.asarray((255, 93, 0))
        color_high = np.asarray((255, 50, 30))
        shift_y = 50 * scale_y  # Introduce a shift for y-coordinates

        for or_id, ex_id, rtl, line_por, is_on in zip(self.lines_ids_or, self.lines_ids_ex, relative_thermal_limits,
                                                    lines_por, lines_service_status):
            if rtl < .5:
                color = color_low + 2. * rtl * (color_middle - color_low)
            elif rtl < 1.:
                color = color_low + min(1., rtl) * (color_high - color_low)
            else:
                color = (255, 20, 20)
            if line_por >= 0:
                ori = layout[or_id]
                ext = layout[ex_id]
            else:
                ori = layout[ex_id]
                ext = layout[or_id]
            if not is_on:
                l.append(lines.Line2D([ori[0], ext[0]], [ori[1]+ shift_y, ext[1]+ shift_y], linewidth=.8,
                                    color=[.8, .8, .8], figure=fig, linestyle='dashed'))
            else:
                l.append(lines.Line2D([ori[0], ext[0]], [ ori[1]+ shift_y, ext[1]+ shift_y], linewidth=1.0,
                                    color=[c / 255. for c in color], figure=fig, 
                                    linestyle='--' if rtl > 1. else '-',
                                    dashes=(2., .8) if rtl > 1. else (None, None)))
        fig.lines.extend(l)

        activated_node_color = (255, 255, 0)

        prods_iter, loads_iter = iter(prods), iter(loads)
        for i, ((x, y), is_prod, is_load, is_changed, n_used_nodes) in enumerate(
                zip(layout, self.are_prods, self.are_loads, are_substations_changed, number_nodes_per_substation)):
            prod = next(prods_iter) if is_prod else 0.
            load = next(loads_iter) if is_load else 0.
            prod_minus_load = prod - load
            linewidth_min = 1.
            outer_radius = self.nodes_outer_radius
            if prod_minus_load > 0:
                color = [c / 255. for c in (0, 153, 255)]
                inner_circle_color = activated_node_color if is_changed else self.background_color
                inner_circle_color = [c / 255. for c in inner_circle_color]
                if n_used_nodes > 1:
                    c = Circle((x, y), outer_radius + linewidth_min, linewidth_min=0., fill=True,
                            color=[c / 255. for c in self.background_color], zorder=10)
                    ax.add_artist(c)
                    c = Circle((x, y), outer_radius + linewidth_min, linewidth_min=.75, fill=False, color=color,
                            zorder=10)
                    ax.add_artist(c)
                c = Circle((x, y), outer_radius, linewidth=0, fill=True, color=inner_circle_color, zorder=9)
                ax.add_artist(c)
                c = Circle((x, y), outer_radius, linewidth=linewidth_min, fill=False, color=color, zorder=10)
                ax.add_artist(c)
                #self.wind_img = self.wind_img.transpose(Image.FLIP_LEFT_RIGHT)
                #img = self.wind_img
                #background_color = (0, 0, 0, 0)
                #background_image = Image.new('RGBA', img.size, background_color)
                #background_image.paste(img, (0, 0), img)
                #image_array = np.array(background_image)
                #zoom_factor = 2 * outer_radius / max(img.size)
                #imagebox = OffsetImage(image_array, zoom=zoom_factor)
                #c = AnnotationBbox(imagebox, (x, y), frameon=False, box_alignment=(0.5, 0.5))
                #ax.add_artist(c)

            elif prod_minus_load < 0:
                color = [c / 255. for c in (210, 77, 255)]
                inner_circle_color = activated_node_color if is_changed else self.background_color
                inner_circle_color = [c / 255. for c in inner_circle_color]
                if n_used_nodes > 1:
                    c = Rectangle((x - outer_radius - linewidth_min , y - outer_radius - linewidth_min ),
                                2. * (outer_radius + linewidth_min ), 2. * (outer_radius + linewidth_min ),
                                linewidth=0., fill=True, color=[c / 255. for c in self.background_color], zorder=10)
                    ax.add_artist(c)
                    c = Rectangle((x - outer_radius - linewidth_min, y - outer_radius - linewidth_min),
                                2. * (outer_radius + linewidth_min ), 2. * (outer_radius + linewidth_min ),
                                linewidth=.6, fill=False, color=color, zorder=10)
                    ax.add_artist(c)
                c = Rectangle((x - outer_radius, y - outer_radius), 2. * outer_radius, 2. * outer_radius,
                              linewidth=0, fill=True, color=inner_circle_color, zorder=9)
                ax.add_artist(c)
                c = Rectangle((x - outer_radius, y - outer_radius), 2. * outer_radius, 2. * outer_radius,
                              linewidth=linewidth, fill=False, color=color, zorder=10)
                ax.add_artist(c)
                #img = self.building_img
                #background_color = (0, 0, 0, 0)
                #background_image = Image.new('RGBA', img.size, background_color)
                #background_image.paste(img, (0, 0), img)
                #image_array = np.array(background_image)
                #zoom_factor = 2 * outer_radius / max(img.size)
                #imagebox = OffsetImage(image_array, zoom=zoom_factor)
                #c = AnnotationBbox(imagebox, (x, y), frameon=False, box_alignment=(0.5, 0.5))
                #ax.add_artist(c)
                #name = ['Generation', 'Sub', 'Engineering Hall', 'Rowan Hall', 'Business Hall', 'Science Hall', 'Discovery Hall', 'Robinson Hall']
                #ax.text(x, y + 40, name[i], ha='center', va='top', fontsize=5, color='white')

            else:
                color = [c / 255. for c in (255, 255, 255)]
                inner_circle_color = activated_node_color if is_changed else self.background_color
                inner_circle_color = [c / 255. for c in inner_circle_color]
                linewidth = linewidth_min
                outer_radius = self.nodes_outer_radius
                if n_used_nodes > 1:
                    c = Rectangle((x, y - math.sqrt(2.) * (outer_radius )),
                                2. * (outer_radius), 2. * (outer_radius ),
                                linewidth=0., fill=True, color=[c / 255. for c in self.background_color],
                                zorder=10, angle=45.)
                    ax.add_artist(c)
                    c = Rectangle((x, y - math.sqrt(2.) * (outer_radius )),
                                2. * (outer_radius ), 2. * (outer_radius ),
                                linewidth=.6, fill=False, color=color, zorder=10, angle=45.)
                    ax.add_artist(c)
                img = self.substation_img
                background_color = (0, 0, 0, 0)
                background_image = Image.new('RGBA', img.size, background_color)
                background_image.paste(img, (0, 0), img)
                image_array = np.array(background_image)
                zoom_factor = 3 * outer_radius / max(img.size)
                imagebox = OffsetImage(image_array, zoom=zoom_factor)
                c = AnnotationBbox(imagebox, (x, y), frameon=False, box_alignment=(0.5, 0.5))
                ax.add_artist(c)

        l = []
        for or_id, ex_id, rtl, line_por, is_on in zip(self.lines_ids_or, self.lines_ids_ex, relative_thermal_limits,
                                                    lines_por, lines_service_status):
            if not is_on:
                continue
            thickness = .6 
            if rtl < .5:
                color = color_low + 2. * rtl * (color_middle - color_low)
            elif rtl < 1.:
                color = color_low + min(1., rtl) * (color_high - color_low)
            else:
                color = (255, 20, 20)
            if line_por >= 0:
                ori = layout[or_id]
                ext = layout[ex_id]
            else:
                ori = layout[ex_id]
                ext = layout[or_id]
            length = math.sqrt((ori[0] - ext[0]) ** 2. + (ori[1] - ext[1]) ** 2.) - 2. * self.nodes_outer_radius
            center = ((ori[0] + ext[0]) / 2., (ori[1] + ext[1]) / 2.)
            angle = math.atan2(ori[1] - ext[1], ori[0] - ext[0])
            distance_arrow_heads = 25
            n_arrow_heads = int(max(1, length // distance_arrow_heads))
            for a in range(n_arrow_heads):
                if n_arrow_heads != 1:
                    offset = a + .25 if self.boolean_dynamic_arrows else a + .75
                    x = center[0] + (offset * distance_arrow_heads - length / 2.) * math.cos(angle)
                    y = center[1] + (offset * distance_arrow_heads - length / 2.) * math.sin(angle)
                else:
                    x = center[0]
                    y = center[1]+ shift_y
                head_angle = math.pi / 6.
                width = 8 + 20 * (thickness - .6)
                x -= width / 2. * math.cos(angle)
                y -= width / 2. * math.sin(angle)
                x1 = x + width * math.cos(angle + head_angle)
                y1 = y + width * math.sin(angle + head_angle)
                x2 = x + width * math.cos(angle - head_angle)
                y2 = y + width * math.sin(angle - head_angle)
                l.append(lines.Line2D([x, x2], [y + shift_y, y2+ shift_y], linewidth=thickness,
                                    color=[c / 255. for c in color], figure=fig,zorder=11, linestyle='-'))
                l.append(lines.Line2D([x, x1], [ y + shift_y, y1 + shift_y], linewidth=thickness,
                                    color=[c / 255. for c in color], figure=fig,zorder=11, linestyle='-'))
        fig.lines.extend(l)

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()

        img_loads_curve_week = pygame.image.fromstring(raw_data, size, "RGB")
        loads_curve_surface = pygame.Surface(self.topology_layout_shape, pygame.SRCALPHA, 32).convert_alpha()
        loads_curve_surface.fill(self.background_color)
        loads_curve_surface.blit(img_loads_curve_week, (0, 0))

        return loads_curve_surface

    def draw_combined_legend_and_diagnosis(self, number_loads_cut, number_prods_cut, number_nodes_splitting, number_lines_switches,
                                distance_initial_grid, line_capacity_usage, n_offlines_lines, number_unavailable_lines,
                                number_unavailable_nodes, max_number_isolated_loads, max_number_isolated_prods):
        
        # Scaling factors
        scale_x = self.video_width / 1300
        scale_y = self.video_height / 700

        # Adjusted width and height
        width = int(250 * scale_x)
        height = int(800 * scale_y)

        surface = pygame.Surface((width, height), pygame.SRCALPHA, 32).convert_alpha()
        surface.fill(self.left_menu_tile_color)
        my_dpi = 100
        fig_width = width / my_dpi
        fig_height = height / my_dpi

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=my_dpi,
                        facecolor=[c / 255. for c in self.left_menu_tile_color], clear=True, tight_layout={'pad': -.3})
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        plt.axis('off')
        plt.ylim(0, height)
        plt.xlim(0, width)

        ax = fig.gca()
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_xticks([])
        ax.set_yticks([])

        offset_text = int(40 * scale_x)
        string_color = (180 / 255., 180 / 255., 180 / 255.)
        header2_color = (200 / 255., 200 / 255., 200 / 255.)
        header_color = (220 / 255., 220 / 255., 220 / 255.)
        value_color = (1., 1., 1.)
        
        plt.text(45 * scale_x, height - 20 * scale_y, 'Legend', fontdict={'size': 12}, color=header_color)
        plt.text(5 * scale_x, height - 50 * scale_y, 'Sources', fontdict={'size': 8.5}, color=header2_color)
        plt.text(offset_text, height - 70 * scale_y, 'Substation', fontdict={'size': 8.5}, color=string_color)
        img = self.substation_img
        background_color = (0, 0, 0, 0)  # RGBA: Transparent background
        background_image = Image.new('RGBA', img.size, background_color)
        background_image.paste(img, (0, 0), img)  # Paste the image onto the background with transparency
        image_array = np.array(background_image)
        zoom_factor = 2 * self.nodes_outer_radius / max(img.size)
        imagebox = OffsetImage(image_array, zoom=zoom_factor)
        c = AnnotationBbox(imagebox, (21 * scale_x, height - 66 * scale_y), frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(c)

        plt.text(offset_text, height - 95 * scale_y, 'Load', fontdict={'size': 8.5}, color=string_color)
        img = self.building_img
        background_color = (0, 0, 0, 0)  # RGBA: Light Green with full opacity
        background_image = Image.new('RGBA', img.size, background_color)
        background_image.paste(img, (0, 0), img)  # img as mask for transparency
        image_array = np.array(background_image)
        zoom_factor = 2. * self.nodes_outer_radius / max(img.size)
        imagebox = OffsetImage(image_array, zoom=zoom_factor)
        c = AnnotationBbox(imagebox, (22 * scale_x, height - 89 * scale_y), frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(c)

        plt.text(offset_text, height - 120 * scale_y, 'Generator', fontdict={'size': 8.5}, color=string_color)
        img = self.wind_img_nc
        background_color = (0, 0, 0, 0)  # RGBA: Light Green with full opacity
        background_image = Image.new('RGBA', img.size, background_color)
        background_image.paste(img, (0, 0), img)  # img as mask for transparency
        image_array = np.array(background_image)
        zoom_factor = 2. * self.nodes_outer_radius / max(img.size)
        imagebox = OffsetImage(image_array, zoom=zoom_factor)
        c = AnnotationBbox(imagebox, (22 * scale_x, height - 118 * scale_y), frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(c)

        plt.text(5 * scale_x, height - 170 * scale_y, 'Power lines', fontdict={'size': 8.5}, color=header2_color)
        fig.lines.append(lines.Line2D([10 * scale_x, 33 * scale_x], [height - 190 * scale_y, height - 190 * scale_y], linewidth=1.,
                                    color=[c / 255. for c in (51, 204, 51)]))
        fig.lines.append(lines.Line2D([20 * scale_x, 23 * scale_x], [height - 187 * scale_y, height - 190 * scale_y], linewidth=1.,
                                    color=[c / 255. for c in (51, 204, 51)]))
        fig.lines.append(lines.Line2D([20 * scale_x, 23 * scale_x], [height - 194 * scale_y, height - 191 * scale_y], linewidth=1.,
                                    color=[c / 255. for c in (51, 204, 51)]))
        plt.text(offset_text, height - 194 * scale_y, 'direction of current', fontdict={'size': 8.5},
                color=string_color)

        color_low = np.asarray((51, 204, 51))
        color_middle = np.asarray((255, 93, 0))
        color_high = np.asarray((255, 50, 30))

        l = []
        n = 50
        for i in range(n):
            if i < n // 2:
                color = [c1 + (c2 - c1) * (i / (n // 2)) for c1, c2 in zip(color_low, color_middle)]
                l.append(lines.Line2D([15 * scale_x, 28 * scale_x], [height - (210 + i) * scale_y, height - (210 + i) * scale_y],
                                    linewidth=1., color=[c / 255. for c in color]))
            else:
                color = [c1 + (c2 - c1) * ((i - n // 2) / (n // 2)) for c1, c2 in zip(color_middle, color_high)]
                l.append(lines.Line2D([15 * scale_x, 28 * scale_x], [height - (210 + i) * scale_y, height - (210 + i) * scale_y],
                                    linewidth=1., color=[c / 255. for c in color]))
        fig.lines.extend(l)
        fig.lines.append(lines.Line2D([29 * scale_x, offset_text - 5], [height - 210 * scale_y, height - 210 * scale_y],
                                    linewidth=1., color=[c / 255. for c in [234, 234, 160]]))
        plt.text(offset_text, height - 216 * scale_y, '0% capacity usage', fontdict={'size': 8.5},
                color=string_color)
        fig.lines.append(lines.Line2D([29 * scale_x, offset_text - 5], [height - 235 * scale_y, height - 235 * scale_y],
                                    linewidth=1., color=[c / 255. for c in [234, 234, 160]]))
        plt.text(offset_text, height - 239 * scale_y, '50% cap. usage', fontdict={'size': 8.5},
                color=string_color)
        fig.lines.append(lines.Line2D([29 * scale_x, offset_text - 5], [height - 259 * scale_y, height - 259 * scale_y],
                                    linewidth=1., color=[c / 255. for c in [234, 234, 160]]))
        plt.text(offset_text, height - 262 * scale_y, '100% cap. usage', fontdict={'size': 8.5},
                color=string_color)

        fig.lines.append(lines.Line2D([10 * scale_x, 33 * scale_x], [height - 279 * scale_y, height - 279 * scale_y], linewidth=2.,
                                    color=[c / 255. for c in (255, 20, 20)], figure=fig, linestyle='--',
                                    dashes=(2., .8)))
        fig.lines.append(lines.Line2D([19 * scale_x, 23 * scale_x], [height - 275 * scale_y, height - 279 * scale_y], linewidth=2.,
                                    color=[c / 255. for c in (255, 20, 20)]))
        fig.lines.append(lines.Line2D([19 * scale_x, 23 * scale_x], [height - 283 * scale_y, height - 279 * scale_y], linewidth=2.,
                                    color=[c / 255. for c in (255, 20, 20)]))
        plt.text(offset_text, height - 283 * scale_y, 'overflowed', fontdict={'size': 8.5}, color=string_color)

        fig.lines.append(lines.Line2D([10 * scale_x, 33 * scale_x], [height - 299 * scale_y, height - 299 * scale_y], linewidth=1.,
                                    color=[.8, .8, .8], figure=fig, linestyle='dashed'))
        plt.text(offset_text, height - 303 * scale_y, 'switched OFF', fontdict={'size': 8.5}, color=string_color)

        # Live diagnosis details
        plt.text(90 * scale_x, height - 350 * scale_y, 'Live diagnosis', fontdict={'size': 12}, color=header_color)
        string_offset = int(65 * scale_x)
        value_offset = int(10 * scale_x)
        up = '^'
        up_offset = int(-11 * scale_x)
        if self.data is not None:
            old_number_loads_cut, old_number_prods_cut, old_number_nodes_splitting, old_number_lines_switches, \
            old_distance_initial_grid, old_usage, old_n_offlines_lines, old_number_unavailable_lines, \
            old_number_unavailable_nodes, old_max_number_isolated_loads, old_max_number_isolated_prods = \
                self.data['number_loads_cut'], self.data['number_prods_cut'], \
                self.data['number_nodes_splitting'], self.data['number_lines_switches'], \
                self.data['distance_initial_grid'], self.data['usage'], \
                self.data['n_offlines_lines'], self.data['number_unavailable_lines'], \
                self.data['number_unavailable_nodes'], self.data['max_number_isolated_loads'], \
                self.data['max_number_isolated_prods']
        else:
            old_number_loads_cut, old_number_prods_cut, old_number_nodes_splitting, old_number_lines_switches, \
            old_distance_initial_grid, old_usage, old_n_offlines_lines, \
            old_number_unavailable_lines, old_number_unavailable_nodes, old_max_number_isolated_loads, \
            old_max_number_isolated_prods = [0] * 11

        def print_variation(old_val, new_val, h):
            if new_val > old_val:
                plt.text(up_offset + 2, height - h - 7, up, fontdict={'size': 12}, color=(1., .5, .5))
            elif new_val < old_val:
                plt.text(up_offset, height - h + 7, up, fontdict={'size': 12}, color=(.5, 1., .5), rotation=180.)

        plt.text(string_offset, height - 380 * scale_y, '# of isolated loads', fontdict={'size': 8.5}, color=string_color)
        plt.text(value_offset, height - 381 * scale_y, '%d' % number_loads_cut,
                fontdict={'size': 8.5},
                color=(1., 0.3, 0.3) if number_loads_cut > max_number_isolated_loads else value_color)
        print_variation(old_number_loads_cut, number_loads_cut, 381 * scale_y)
        plt.text(value_offset, height - 380 * scale_y, '   / %d' % max_number_isolated_loads,
                fontdict={'size': 8.5}, color=value_color)
        plt.text(string_offset, height - 400 * scale_y, '# of isolated productions', fontdict={'size': 8.5}, color=string_color)
        plt.text(value_offset, height - 401 * scale_y, '%d' % number_prods_cut,
                fontdict={'size': 8.5},
                color=(1., 0.3, 0.3) if number_prods_cut > max_number_isolated_prods else value_color)
        plt.text(value_offset, height - 400 * scale_y, '   / %d' % max_number_isolated_prods,
                fontdict={'size': 8.5}, color=value_color)
        print_variation(old_number_prods_cut, number_prods_cut, 401 * scale_y)

        plt.text(string_offset, height - 430 * scale_y, '# of node switches of last action', fontdict={'size': 8.5},
                color=string_color)
        plt.text(value_offset, height - 431 * scale_y, '%d' % number_nodes_splitting, fontdict={'size': 8.5}, color=value_color)
        print_variation(old_number_nodes_splitting, number_nodes_splitting, 431 * scale_y)
        plt.text(string_offset, height - 450 * scale_y, '# of line switches of last action', fontdict={'size': 8.5},
                color=string_color)
        plt.text(value_offset, height - 451 * scale_y, '%d' % number_lines_switches, fontdict={'size': 8.5}, color=value_color)
        print_variation(old_number_lines_switches, number_lines_switches, 451 * scale_y)

        plt.text(string_offset, height - 480 * scale_y, 'average line capacity usage', fontdict={'size': 8.5}, color=string_color)
        usage = 100. * np.mean(line_capacity_usage)
        plt.text(value_offset, height - 481 * scale_y, '%.1f%%' % usage if usage < 5000 else '∞', fontdict={'size': 8.5},
                color=value_color)
        print_variation(old_usage, usage, 481 * scale_y)
        plt.text(string_offset, height - 500 * scale_y, '# of OFF lines', fontdict={'size': 8.5}, color=string_color)
        plt.text(value_offset, height - 501 * scale_y, '%d' % n_offlines_lines, fontdict={'size': 8.5}, color=value_color)
        print_variation(old_n_offlines_lines, n_offlines_lines, 501 * scale_y)
        plt.text(string_offset, height - 520 * scale_y, '# of unavailable lines', fontdict={'size': 8.5}, color=string_color)
        plt.text(value_offset, height - 521 * scale_y, '%d' % number_unavailable_lines, fontdict={'size': 8.5}, color=value_color)
        print_variation(old_number_unavailable_lines, number_unavailable_lines, 521 * scale_y)
        plt.text(string_offset, height - 540 * scale_y, '# of unactionable nodes', fontdict={'size': 8.5}, color=string_color)
        plt.text(value_offset, height - 541 * scale_y, '%d' % number_unavailable_nodes, fontdict={'size': 8.5}, color=value_color)
        print_variation(number_unavailable_nodes, number_unavailable_nodes, 541 * scale_y)

        plt.text(string_offset, height - 570 * scale_y, 'distance to reference grid', fontdict={'size': 8.5}, color=string_color)
        plt.text(value_offset, height - 571 * scale_y, '%d' % distance_initial_grid, fontdict={'size': 8.5}, color=value_color)
        print_variation(old_distance_initial_grid, distance_initial_grid, 571 * scale_y)

        fig.tight_layout()

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()

        img_legend = pygame.image.fromstring(raw_data, size, "RGB")
        surface.blit(img_legend, (10 * scale_x, 5 * scale_y))

        pygame.gfxdraw.hline(surface, 0, width, 0, (64, 64, 64))
        pygame.gfxdraw.hline(surface, 0, width, height - 1, (64, 64, 64))
        pygame.gfxdraw.vline(surface, 0, height - 1, 0, (64, 64, 64))
        pygame.gfxdraw.vline(surface, width, 0, height - 1, (64, 64, 64))

        # Keep current data for next data differences
        self.data = {'number_loads_cut': number_loads_cut, 'number_prods_cut': number_prods_cut,
                    'number_nodes_splitting': number_nodes_splitting, 'number_lines_switches': number_lines_switches,
                    'distance_initial_grid': distance_initial_grid, 'usage': usage,
                    'n_offlines_lines': n_offlines_lines, 'number_unavailable_lines': number_unavailable_lines,
                    'number_unavailable_nodes': number_unavailable_nodes,
                    'max_number_isolated_loads': max_number_isolated_loads,
                    'max_number_isolated_prods': max_number_isolated_prods}

        return surface


    @staticmethod
    def draw_plot_pause():
        pause_font = pygame.font.SysFont("Arial", 25)
        yellow = (255, 255, 179)
        txt_surf = pause_font.render('pause', False, (80., 80., 80.))
        alpha_img = pygame.Surface(txt_surf.get_size(), pygame.SRCALPHA)
        alpha_img.fill(yellow + (72,))
        pause_surface = pygame.Surface((200, 70), pygame.SRCALPHA, 32).convert_alpha()
        pause_surface.fill(yellow + (128,))
        pause_surface.blit(txt_surf, (64, 18))

        return pause_surface

    @staticmethod
    def draw_plot_game_over():
        game_over_font = pygame.font.SysFont("Arial", 25)
        red = (255, 26, 26)
        txt_surf = game_over_font.render('game over', False, (255, 255, 255))
        alpha_img = pygame.Surface(txt_surf.get_size(), pygame.SRCALPHA)
        alpha_img.fill(red + (128,))
        game_over_surface = pygame.Surface((200, 70), pygame.SRCALPHA, 32).convert_alpha()
        game_over_surface.fill(red + (128,))
        game_over_surface.blit(txt_surf, (38, 18))

        return game_over_surface

    def _update_left_menu(self, epoch, timestep):
        # Scaling factors
        scale_x = self.video_width / 1300
        scale_y = self.video_height / 700

        self.left_menu = pygame.Surface(self.left_menu_shape, pygame.SRCALPHA, 32).convert_alpha()
        self.left_menu.blit(self.text_render('Timestep'), (int(150 * scale_x), int(10 * scale_y)))
        self.left_menu.blit(self.value_render(str(timestep)), (int(250 * scale_x), int(10 * scale_y)))

        if self.timestep_duration_seconds > 30 * 60:
            n_hours_to_display_top_loadplot = 7 * 24
        else:
            n_hours_to_display_top_loadplot = 3 * 24
        n_hours_to_display_bottom_loadplot = 1 * 24

        loads_curve_surface = self.draw_surface_loads_curves(
            n_hours_to_display_top_loadplot=n_hours_to_display_top_loadplot,
            n_hours_to_display_bottom_loadplot=n_hours_to_display_bottom_loadplot
        )

        n_hours_to_display = 24
        rtl_curves_surface = self.draw_surface_relative_thermal_limits(
            n_timesteps=int(n_hours_to_display * 3600 // self.timestep_duration_seconds)
        )

        n_days_horizon = n_hours_to_display // 24
        horizon_scale = 'day'
        if n_days_horizon <= 1:
            n_days_horizon = n_hours_to_display
            horizon_scale = 'hour'
        n_overflows_surface = self.draw_surface_n_overflows(
            n_timesteps=int(n_hours_to_display * 3600 // self.timestep_duration_seconds),
            left_xlabel='{} {}{} ago'.format(n_days_horizon, horizon_scale, 's' if n_days_horizon > 1 else '')
        )

        # Dynamically calculate the positions
        loads_curve_y = int(50 * scale_y) -20
        rtl_curve_y = int(380 * scale_y) -20
        n_overflows_y = int(560 * scale_y) -20

        pygame.gfxdraw.vline(self.left_menu, self.left_menu_shape[0] - 1, 0, self.left_menu_shape[1], (128, 128, 128))
        self.left_menu.blit(loads_curve_surface, (0, loads_curve_y))
        self.left_menu.blit(rtl_curves_surface, (0, rtl_curve_y))
        self.left_menu.blit(n_overflows_surface, (0, n_overflows_y))

    def _update_topology(self, scenario_id, date, relative_thermal_limits, lines_por, lines_service_status, prods,
                        loads, are_substations_changed, game_over, cascading_frame_id, number_loads_cut,
                        number_prods_cut, number_nodes_splitting, number_lines_switches, distance_initial_grid,
                        line_capacity_usage, number_off_lines, number_unavailable_lines, number_unavailable_nodes,
                        max_number_isolated_loads, max_number_isolated_prods, number_nodes_per_substation):
        
        # Scaling factors
        scale_x = self.video_width / 1300
        scale_y = self.video_height / 700

        # Adjust the topology layout shape based on scaling
        scaled_topology_layout_shape = (int(self.topology_layout_shape[0] * scale_x), 
                                        int(self.topology_layout_shape[1] * scale_y))
        

        self.topology_layout = pygame.Surface(scaled_topology_layout_shape, pygame.SRCALPHA, 32).convert_alpha()
        self.nodes_surface = pygame.Surface(scaled_topology_layout_shape, pygame.SRCALPHA, 32).convert_alpha()
        #self.injections_surface = pygame.Surface(scaled_topology_layout_shape, pygame.SRCALPHA, 32).convert_alpha()
        #self.lines_surface = pygame.Surface(scaled_topology_layout_shape, pygame.SRCALPHA, 32).convert_alpha()

        # Lines
        if self.relative_thermal_limits:
            if cascading_frame_id is None:
                self.relative_thermal_limits.append(relative_thermal_limits)
        else:
            self.relative_thermal_limits.append(relative_thermal_limits)

        lines_surf = self.draw_surface_grid(relative_thermal_limits, lines_por, lines_service_status, prods, loads,
                                            are_substations_changed, number_nodes_per_substation)
        
        offset_x = (-68 if self.grid_case == 118 else -20 if self.grid_case == 30 else 0) * scale_x
        offset_y = 0
        self.topology_layout.blit(lines_surf, (offset_x, offset_y))

        if self.loads:
            if cascading_frame_id is None:
                self.loads.append(loads)
        else:
            self.loads.append(loads)
        self.draw_surface_nodes_headers(scenario_id, date, cascading_result_frame=cascading_frame_id)

        self.topology_layout.blit(self.nodes_surface, (0, 0))

        if game_over:
            game_over_x = int(320 * scale_x)
            game_over_y = int(320 * scale_y)
            self.topology_layout.blit(self.game_over_surface, (game_over_x, game_over_y))

    def render(self, lines_capacity_usage, lines_por, lines_service_status, epoch, timestep, scenario_id, prods, loads,
            date, are_substations_changed, number_nodes_per_substation, number_loads_cut, number_prods_cut,
            number_nodes_splitting, number_lines_switches, distance_initial_grid, number_off_lines,
            number_unavailable_lines, number_unactionable_nodes, max_number_isolated_loads,
            max_number_isolated_prods, game_over=False, cascading_frame_id=None):
        plt.close('all')

        def event_looper(force=False):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        exit()
                    if event.key == pygame.K_SPACE:
                        pause_surface = self.draw_plot_pause()
                        pause_x = int((320 + self.left_menu_shape[0]) * self.video_width / 1300)
                        pause_y = int(320 * self.video_height / 700)
                        self.screen.blit(pause_surface, (pause_x, pause_y))
                        pygame.display.flip()
                        return not force
            return force

        force = event_looper(force=False)
        while event_looper(force=force):
            pass

        # The game is not paused anymore (or never has been), I can render the next surface
        self.screen.fill(self.background_color)

        # Execute full plotting mechanism: order is important
        self._update_topology(scenario_id, date, lines_capacity_usage, lines_por, lines_service_status, prods, loads,
                            are_substations_changed, game_over, cascading_frame_id, number_loads_cut,
                            number_prods_cut, number_nodes_splitting, number_lines_switches, distance_initial_grid,
                            lines_capacity_usage, number_off_lines, number_unavailable_lines,
                            number_unactionable_nodes, max_number_isolated_loads, max_number_isolated_prods,
                            number_nodes_per_substation)

        if cascading_frame_id is None:
            self._update_left_menu(epoch, timestep)


        # Blit all macro surfaces on screen
        left_menu_x = 0
        left_menu_y = 0
        topology_layout_x = int(self.left_menu_shape[0] * self.video_width / 1300)
        topology_layout_y = 0
        self.screen.blit(self.left_menu, (left_menu_x, left_menu_y))
        self.screen.blit(self.topology_layout, (topology_layout_x, topology_layout_y))

        # Draw combined legend and live diagnosis
        combined_surface = self.draw_combined_legend_and_diagnosis(
            number_loads_cut, number_prods_cut, number_nodes_splitting, number_lines_switches,
            distance_initial_grid, lines_capacity_usage, number_off_lines, number_unavailable_lines,
            number_unactionable_nodes, max_number_isolated_loads, max_number_isolated_prods)
        combined_surface_x = self.video_width - combined_surface.get_width()
        combined_surface_y = 0
        self.screen.blit(combined_surface, (combined_surface_x, combined_surface_y))

        pygame.display.flip()
        # Bugfix for mac
        # pygame.event.get()

        self.boolean_dynamic_arrows = not self.boolean_dynamic_arrows

if __name__ == '__main__':
    a = np.asarray(case_layouts[30])
    print(np.min(a[:, 0]))
    print(np.max(a[:, 0]))
    print(np.min(a[:, 1]))
    print(np.max(a[:, 1]))
    a = np.asarray(case_layouts[14])
    print()
    print(np.min(a[:, 0]))
    print(np.max(a[:, 0]))
    print(np.min(a[:, 1]))
    print(np.max(a[:, 1]))
    scale(1, 1., 0)
