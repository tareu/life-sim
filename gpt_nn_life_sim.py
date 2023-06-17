import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import uuid
import copy
import inspect
import os




class Person:
    def __init__(self, x, y, parent1, parent2, lifespan, reproduction_timer, birth_hunger, hunger_max, repro_min_hunger_percent, move_cost):
        self.x, self.y = x, y
        self.lifespan, self.reproduction_timer = lifespan, random.randint(1, reproduction_timer)
        self.initial_lifespan = self.lifespan
        self.reproduction_cooldown_initial = reproduction_timer
        self.uuid = uuid.uuid4()
        self.hunger = birth_hunger
        self.hunger_max = hunger_max
        self.birth_hunger = birth_hunger
        self.surroundings_history = []
        self.repro_min_hunger_percent = repro_min_hunger_percent
        self.move_cost = move_cost

        if parent1 is None or parent2 is None:
            self.nn = self.create_neural_net()
        else:
            self.nn = self.combine_neural_nets(parent1.nn, parent2.nn)

    def get_properties(self, properties):
        return ', '.join(str(int(getattr(self, prop))) for prop in properties)

    def create_neural_net(self):
        nn_model = nn.Sequential(
            nn.Linear(267, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 43),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(43, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 6),
            nn.Softmax(dim=1)
        )
        return nn_model
    
    def combine_neural_nets(self, nn1, nn2):
        new_nn = self.create_neural_net()
        total_weights = sum([param.numel() for param in new_nn.parameters()])
        num_randomized_weights = round(0.01 * total_weights)

        def process_linear_layer(new_layer, layer1, layer2, weights_to_randomize):
            for child_param, parent1_param, parent2_param in zip(new_layer.parameters(), layer1.parameters(), layer2.parameters()):
                child_param.data = parent1_param.data.clone()
                child_param.data += parent2_param.data
                child_param.data *= 0.5
                child_param.data *= (1 + 0.01 * torch.rand_like(child_param.data).uniform_(-1, 1))

                current_weights_to_randomize = {i for i in weights_to_randomize if i < child_param.data.numel()}
                weights_to_randomize -= current_weights_to_randomize

                child_param.data.view(-1)[list(current_weights_to_randomize)] = torch.randn(len(current_weights_to_randomize)).to(child_param.data.device)

        weights_to_randomize = set(random.sample(range(total_weights), num_randomized_weights))

        for layer_idx, (layer1, layer2, new_layer) in enumerate(zip(nn1, nn2, new_nn)):
            if isinstance(layer1, nn.Linear):
                process_linear_layer(new_layer, layer1, layer2, weights_to_randomize)
            elif isinstance(layer1, (nn.ReLU, nn.Dropout, nn.Softmax)):
                new_nn[layer_idx] = random.choice([layer1, layer2])
            else:
                raise NotImplementedError("Unknown layer type encountered")

        return new_nn


    
    def move(self, grid):
        directions = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]

        surroundings = self.get_surroundings(grid)
        if len(self.surroundings_history) > 9:
            self.surroundings_history.pop(0)
        self.surroundings_history.append(surroundings)

        # Pad or truncate the surroundings list
        surroundings_data = pad_or_truncate(surroundings, 24)
        # Pad or truncate the surroundings_history_data and each of its sublists
        surroundings_history_data = pad_or_truncate(self.surroundings_history, 10)
        surroundings_history_data = [pad_or_truncate(sublist, 24) for sublist in surroundings_history_data]

        # Flatten surroundings_history_data
        surroundings_history_data_flat = [item for sublist in surroundings_history_data for item in sublist]

        # Create the input tensor
        input_tensor = torch.tensor([self.lifespan, self.reproduction_timer, self.hunger] + surroundings_data + surroundings_history_data_flat, dtype=torch.float32).unsqueeze(0)
        output = self.nn(input_tensor)
        move_idx = torch.argmax(output[0][:5]).item()
        wall_idx = torch.max(output[0][5:6]).item()
        
        dx, dy = directions[move_idx]
        grid_size = len(grid)
        new_x, new_y = (self.x + dx) % grid_size, (self.y + dy) % grid_size

        if isinstance(grid[new_x][new_y], Food):
            if self.hunger < self.hunger_max:
                self.hunger += grid[new_x][new_y].food_value
                grid[new_x][new_y] = None

        elif isinstance(grid[new_x,new_y], Wall):
            if wall_idx == 1.0:
                grid[new_x][new_y] = None
                
        elif grid[new_x][new_y] is None:
            if wall_idx == 1.0:
                grid[new_x][new_y] = Wall(new_x, new_y)
                
            if wall_idx == 0.0:
                self.hunger -= self.move_cost
                self.update_position(grid, new_x, new_y)
            

    def get_surroundings(self, grid):
        surroundings = []
        grid_width = len(grid)
        grid_height = len(grid[0])

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = (self.x + dx) % grid_width, (self.y + dy) % grid_height

            if isinstance(grid[new_x][new_y], Food):
                surroundings.extend([1, new_x, new_y])
            elif isinstance(grid[new_x][new_y], Person):
                surroundings.extend([2, new_x, new_y, int(grid[new_x][new_y].uuid), grid[new_x][new_y].lifespan])
            elif grid[new_x][new_y] is None:
                surroundings.extend([0, new_x, new_y])
            
        
        return surroundings


    def update_position(self, grid, new_x, new_y):
        grid[self.x][self.y] = None
        self.x, self.y = new_x, new_y
        grid[new_x][new_y] = self

    def can_reproduce(self, grid):
        for px, py in get_valid_neighbors(self.x,self.y,grid.size):
            
            if isinstance(grid[px][py], Person):
            
                if self.reproduction_timer > 0:
                    continue

                # no repro if your hunger is lower than minimum repro hunger
                if self.hunger < (self.birth_hunger * self.repro_min_hunger_percent) or grid[px][py].hunger < (grid[px][py].birth_hunger * self.repro_min_hunger_percent):
                    continue

                # Check if there's a None space around
                for nx, ny in get_valid_neighbors(self.x, self.y, grid.size):
                    if grid[nx][ny] == None:
                        return True

        return False

    def reproduce(self, grid):
        for nx, ny in get_valid_neighbors(self.x, self.y, grid.size):
            if grid[nx][ny] == None:
                for px, py in get_valid_neighbors(self.x, self.y, grid.size):
                    if isinstance(grid[px][py], Person):

                        # no repro if your hunger is lower than minimum repro hunger
                        if self.hunger < (self.birth_hunger * self.repro_min_hunger_percent) or grid[px][py].hunger < (grid[px][py].birth_hunger * self.repro_min_hunger_percent):
                            continue

                        self.hunger = self.hunger - ((self.birth_hunger * self.repro_min_hunger_percent) / 2)

                        #(self, parent1, parent2, grid, x, y):
                        egg = Egg(self, grid[px][py], grid, nx, ny)
                        self.reproduction_timer = self.reproduction_cooldown_initial
                        return egg
        return None



    def decrease_lifespan_and_cooldown_and_hunger(self, grid):
        if self.reproduction_timer > 0:
            self.reproduction_timer -= 1
        self.hunger -= 1
        self.lifespan -= 1
        if self.hunger <= 0 or self.lifespan <= 0:
            grid[self.x][self.y] = None
            return True
        return False

def flatten_surroundings(surroundings):
    flattened = []
    for obj in surroundings:
        if obj['type'] == 'Food':
            flattened.extend([1, 0, obj['x'], obj['y']])
        elif obj['type'] == 'Person':
            flattened.extend([0, 1, obj['x'], obj['y'], obj['uuid'], obj['lifespan']])
        elif obj['type'] == 'Empty':
            flattened.extend([0, 0, obj['x'], obj['y']])
    return flattened

class Grid:
    def __init__(self, size):
        self.grid = np.empty((size, size), dtype=object)
        self.size = size

    def __len__(self):
        return len(self.grid)

    def __getitem__(self, index):
        return self.grid[index]

    def add_people(self, num_people, initial_lifespan, reproduction_timer, birth_hunger, max_hunger, repro_min_hunger_percent, move_cost):
        people = []
        for _ in range(num_people):
            while True:
                x, y = random.randint(0, len(self.grid) - 1), random.randint(0, len(self.grid[0]) - 1)
                if self.grid[x][y] is None:
                    person = Person(x, y, None, None, initial_lifespan, reproduction_timer, birth_hunger, max_hunger, repro_min_hunger_percent, move_cost)
                    self.grid[x][y] = person
                    people.append(person)
                    break
        return people

    def add_food_sources(self, num_food_sources, food_production_rate, food_distribution_radius, food_value):
        food_sources = []
        for _ in range(num_food_sources):
            while True:
                x, y = random.randint(0, len(self.grid) - 1), random.randint(0, len(self.grid[0]) - 1)
                if self.grid[x][y] is None:
                    food_source = FoodSource(x, y, food_production_rate, food_distribution_radius, food_value)
                    self.grid[x][y] = food_source
                    food_sources.append(food_source)
                    break
        return food_sources

    def generate_food(self, food_sources):
        food_list = []
        for food_source in food_sources:
            food_generated = food_source.generate_food(self.grid)
            for food in food_generated:
                x, y = food.x, food.y
                if self.grid[x][y] is None:
                    self.grid[x][y] = food
                    food_list.append(food)
        return food_list

    def move_sources(self, sources):
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for source in sources:
            x, y = source.x, source.y
            random.shuffle(directions)
            for dx, dy in directions:
                new_x, new_y = x + (dx * 1), y + (dy * 1)
                if 0 < new_x < len(self.grid) and 0 < new_y < len(self.grid) :
                    source.x, source.y = new_x, new_y
                    break

class FoodSource:
    def __init__(self, x, y, food_production_rate, food_distribution_radius, food_value):
        self.x = x
        self.y = y
        self.food_production_rate = food_production_rate
        self.food_distribution_radius = food_distribution_radius
        self.food_value = food_value

    def generate_food(self, grid):
        food_generated = []
        for _ in range(self.food_production_rate):
            count = self.food_production_rate
            while count > 0:
                count = count - 1
                x_offset = random.randint(-self.food_distribution_radius, self.food_distribution_radius)
                y_offset = random.randint(-self.food_distribution_radius, self.food_distribution_radius)
                x, y = self.x + x_offset, self.y + y_offset

                if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] is None:
                    food = Food(x, y, self.food_value)
                    food_generated.append(food)
                    break
        return food_generated

class Graph_Line:
    def __init__(self, last, pop):
        self.last = last
        self.pop = pop

class Egg:
    def __init__(self, parent1, parent2, grid, x, y):
        self.parent1 = copy.copy(parent1)
        self.parent2 = copy.copy(parent2)
        self.grid = grid
        self.x = x
        self.y = y
        grid[x][y] = self

class Wall:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        


def hatch_egg(egg):
    parent1, parent2 = egg.parent1, egg.parent2
    child = Person(egg.x, egg.y, parent1, parent2, parent1.initial_lifespan, parent1.reproduction_cooldown_initial, parent1.birth_hunger - (parent1.repro_min_hunger_percent * parent1.birth_hunger / 2), parent1.hunger_max, parent1.repro_min_hunger_percent, parent1.move_cost)
    return child

class Simulation:
    def __init__(self, grid, people, food_sources, replenish_rate):
        self.grid = grid
        self.people = people
        
        self.food_sources = food_sources
        self.initial_replenish_rate = replenish_rate
        self.replenish_rate = replenish_rate
        self.new_eggs = []
        
    def step(self, tick):
        new_eggs = []
        for person in self.people:
            
            person.move(self.grid)
            if person.can_reproduce(self.grid):
                
                egg = person.reproduce(self.grid)
                if egg is not None:
                    
                    new_eggs.append(egg)
            if person.decrease_lifespan_and_cooldown_and_hunger(self.grid):
                self.people.remove(person)

        # Process eggs in parallel (step 4)
        self.process_eggs(new_eggs)

        if tick % 1 == 0:
            self.grid.move_sources(self.food_sources)

        if self.replenish_rate <= 0:
            self.food_list = self.grid.generate_food(self.food_sources)
            self.replenish_rate = self.initial_replenish_rate
        if self.replenish_rate > 0:
            self.replenish_rate -= 1

    def process_eggs(self, eggs):
        new_people = []
        for egg in eggs:
            
            new_people.append(hatch_egg(egg))
     

        for new_person in new_people:
            x, y = new_person.x, new_person.y
            self.grid[x][y] = new_person
            self.people.append(new_person)



class Food:
    def __init__(self, x, y, food_value):
        self.x, self.y = x, y
        self.food_value = food_value
       
       
def get_valid_neighbors(x, y, grid_size):
    valid_neighbors = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_x, new_y = (x + dx) % grid_size, (y + dy) % grid_size
        valid_neighbors.append((new_x, new_y))
    return valid_neighbors

def change_displayed_properties(new_properties):
    global displayed_properties
    displayed_properties = new_properties

def draw_rounded_rect(surface, rect, color, corner_radius):
    pygame.draw.rect(surface, color, rect, border_radius=corner_radius)

def draw_gradient_rect(surface, rect, start_color, end_color, vertical=True):
    if vertical:
        for i in range(rect.height):
            color = (
                start_color[0] + int((end_color[0] - start_color[0]) * i / rect.height),
                start_color[1] + int((end_color[1] - start_color[1]) * i / rect.height),
                start_color[2] + int((end_color[2] - start_color[2]) * i / rect.height),
            )
            pygame.draw.line(surface, color, (rect.x, rect.y + i), (rect.x + rect.width, rect.y + i))
    else:
        for i in range(rect.width):
            color = (
                start_color[0] + int((end_color[0] - start_color[0]) * i / rect.width),
                start_color[1] + int((end_color[1] - start_color[1]) * i / rect.width),
                start_color[2] + int((end_color[2] - start_color[2]) * i / rect.width),
            )
            pygame.draw.line(surface, color, (rect.x + i, rect.y), (rect.x + i, rect.y + rect.height))
            
def pad_or_truncate(data, target_length):
    if len(data) > target_length:
        return data[:target_length]
    else:
        if len(data) == 0:
            data.append(0)
        if isinstance(data[0], list):
            sublist_length = len(data[0])
            padding_element = [0] * sublist_length
        else:
            padding_element = 0
        return data + [padding_element] * (target_length - len(data))
    


def make_dict_from_vars(*args):
    variables_dict = {}
    frame = inspect.currentframe().f_back
    for var_name, var_value in frame.f_locals.items():
        if var_value in args and isinstance(var_value, (int, float)):
            variables_dict[var_name] = var_value
    return variables_dict

def parse_line(line):
    line = line.strip()
    if '=' in line:
        key, value_str = line.split('=', 1)
        key, value_str = key.strip(), value_str.strip()

        try:
            if '.' in value_str:
                value = float(value_str)
            else:
                value = int(value_str)
            return key, value
        except ValueError:
            pass
    return None, None

def load_variables_from_file(my_string, default_values):
    file_path = f"{my_string}.txt"
    loaded_values = default_values.copy()

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            for line in file:
                key, value = parse_line(line)
                if key and key in default_values:
                    loaded_values[key] = value
    return loaded_values

def save_variables_to_file(my_string, variables_dict):
    file_path = f"{my_string}.txt"

    with open(file_path, "w") as file:
        for key, value in variables_dict.items():
            file.write(f"{key} = {value}\n")

def instantiate_variables(variables_dict, local_vars):
    for key, value in variables_dict.items():
        local_vars[key] = value

def list_txt_files():
    return [file[:-4] for file in os.listdir() if file.endswith(".txt") and file != "requirements.txt"]


def main():

    framerate = 0
    
    grid_size = 50
    num_people = 200
    initial_lifespan = 4000
    reproduction_timer = 200
    max_hunger = 1000
    birth_hunger = max_hunger/2
    food_value = 250
    replenish_rate = 1
    num_food_sources = 1
    food_production_rate = 2
    food_distribution_radius = 5
    repro_min_hunger_percent = 0.50
    move_cost = 10

    save_variables_to_file("defaults", make_dict_from_vars(grid_size,num_people,initial_lifespan,reproduction_timer,max_hunger,birth_hunger,food_value,replenish_rate,num_food_sources,food_production_rate,food_distribution_radius,repro_min_hunger_percent,move_cost))

    grid = Grid(grid_size)
    people = grid.add_people(num_people, initial_lifespan, reproduction_timer, birth_hunger, max_hunger, repro_min_hunger_percent, move_cost)
    
    food_sources = grid.add_food_sources(num_food_sources, food_production_rate, food_distribution_radius, food_value)
    for _ in food_sources:
        for _ in range(10):
            grid.generate_food(food_sources)

    simulation = Simulation(grid, people, food_sources, replenish_rate)

    person_colour = (255,255,0)
    food_colour = (0,255,0)
    background_colour = (0,0,0)
    egg_colour = (255, 248, 220)
    empty_colour = (0,43,20)
    text_colour = (255,255,255)
    menu_background_colour = (4,134,103)
    menu_select_colour = (36,20, 10)
    wall_colour = (200,200,200)

    show_UI = False
   
    cell_size = 2
    
    pygame.init()
    screen = pygame.display.set_mode((1400, 800), pygame.RESIZABLE)
    pygame.display.set_caption("2D People Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    list_font = pygame.font.Font(None, 36-9)

   

    displayed_properties = ['hunger', 'lifespan', 'reproduction_timer', 'x', 'y']

    screen_width, screen_height = screen.get_size()

    WIDTH = screen_width
    list_width = WIDTH // 5
    list_x = 0
    running = True
# pygame.draw.rect(screen, (255,255,255), (last,0 ,1,len(simulation.people)), 1)
    tick = 0
    last = 0
    graph_history = []

    menu = True

    zoom = 1.0
    target_zoom = screen_height/(cell_size*grid_size)
    camera_x = (screen_width/2)-((cell_size*grid_size)/2)
    camera_y = (screen_height/2)-((cell_size*grid_size)/2)

    zoom_speed = 0.1
    lerp_speed = 0.1
    scroll_speed = 10
    scroll_y = 0
    new_val = ''
    dragging = False
    prev_mouse_pos = None

    # Create a surface for the info_box with rounded corners and a gradient background
    info_box_surface = pygame.Surface((200, 75), pygame.SRCALPHA)
    info_box_rect = info_box_surface.get_rect()
    corner_radius = 10
    bg_start_color = (100, 100, 255)
    bg_end_color = (255, 100, 100)
    draw_gradient_rect(info_box_surface, info_box_rect, bg_start_color, bg_end_color)
    draw_rounded_rect(info_box_surface, info_box_rect, (0, 0, 0), corner_radius)

    info_box_font = pygame.font.SysFont(None, 24)

    selected_menu_item = 0

    change_config_selected_item = None

    in_change_variables_menu = False

    in_load_variables_menu = False

    in_save_variables_menu = False

    menu_items = []
    main_menu_items = ['load config','change config', 'save config', 'load neural nets', 'save neural nets', 'quit']
    while running:
         
        mouse_x, mouse_y = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_i:
                    show_UI = not show_UI
                
                if event.key == pygame.K_ESCAPE and not in_change_variables_menu and not in_load_variables_menu:
                    menu = not menu
                    selected_menu_item = 0
                    in_change_variables_menu = False
                    in_load_variables_menu = False

                if menu:
                    if not in_change_variables_menu and not in_load_variables_menu and not in_save_variables_menu:
                        
                        menu_items = main_menu_items
                        if event.key == pygame.K_UP:
                            if selected_menu_item > 0:
                                selected_menu_item = selected_menu_item - 1
                            

                        elif event.key == pygame.K_RIGHT:
                            pass

                        elif event.key == pygame.K_DOWN:
                            selected_menu_item = selected_menu_item + 1
                            

                        elif event.key == pygame.K_LEFT:
                            pass
                        elif event.key == pygame.K_RETURN:
                            if selected_menu_item == 0:
                                
                                menu_items = list_txt_files()
                                selected_menu_item = 0
                                in_load_variables_menu = True
                                break

                            if selected_menu_item == 1:
                                
                                menu_items = list(make_dict_from_vars(grid_size,num_people,initial_lifespan,reproduction_timer,max_hunger,birth_hunger,food_value,replenish_rate,num_food_sources,food_production_rate,food_distribution_radius,repro_min_hunger_percent,move_cost))
                                selected_menu_item = 0
                                in_change_variables_menu = True
                                break

                            if selected_menu_item == 2:
                                
                                menu_items = ['enter text here', 'save']
                                selected_menu_item = 0
                                in_save_variables_menu = True
                                break
                            if selected_menu_item == 5:
                                running = False
                                break
                                
                    if in_change_variables_menu:
                        
                        
                        if event.key == pygame.K_UP:
                            if selected_menu_item > 0:
                                selected_menu_item = selected_menu_item - 1
                            
                            break
                        if event.key == pygame.K_RIGHT:
                            pass
                        if event.key == pygame.K_DOWN:
                            selected_menu_item = selected_menu_item + 1
                            
                            break
                        if event.key == pygame.K_LEFT:
                            pass
                        if change_config_selected_item == None:
                            if event.key == pygame.K_RETURN:
                                change_config_selected_item = menu_items[selected_menu_item]
                                change_config_selected_value = locals()[change_config_selected_item]
                                new_val = str(change_config_selected_value)
                                break
                        
                        if event.key == pygame.K_ESCAPE:
                            if change_config_selected_item == None:
                                
                                selected_menu_item = 0
                                in_change_variables_menu = False
                            change_config_selected_item = None
                            break
                        if not change_config_selected_item == None:
                            if event.key == pygame.K_RETURN:
                                
                                change_config_selected_item = menu_items[selected_menu_item]
                                locals()[change_config_selected_item] = type(locals()[change_config_selected_item])(new_val)
                                
                                
                                break

                            if event.key == pygame.K_BACKSPACE:
                                # get text input from 0 to -1 i.e. end.
                                new_val = new_val[:-1]
                                break
                            # Unicode standard is used for string
                            # formation
                            
                            new_val += event.unicode
                            

                    if in_save_variables_menu:
                        
                        if event.key == pygame.K_UP:
                            if selected_menu_item > 0:
                                selected_menu_item = selected_menu_item - 1
                            
                            break
                        if event.key == pygame.K_RIGHT:
                            pass
                        if event.key == pygame.K_DOWN:
                            selected_menu_item = selected_menu_item + 1
                            
                            break
                        if event.key == pygame.K_LEFT:
                            break
                        if event.key == pygame.K_RETURN:
                            if selected_menu_item == 0:
                                break
                            if selected_menu_item == 1:
                                
                                break
                        if event.key == pygame.K_ESCAPE:
                            
                            selected_menu_item = 0
                            in_save_variables_menu = False
                            break

                    if in_load_variables_menu:
                        
                        if event.key == pygame.K_UP:
                            if selected_menu_item > 0:
                                selected_menu_item = selected_menu_item - 1
                            break
                        if event.key == pygame.K_RIGHT:
                            pass
                        if event.key == pygame.K_DOWN:
                            selected_menu_item = selected_menu_item + 1
                            break
                        if event.key == pygame.K_LEFT:
                            pass
                        if event.key == pygame.K_RETURN:
                            if selected_menu_item == 0:
                                pass
                            if selected_menu_item == 1:
                                pass

                        if event.key == pygame.K_ESCAPE:
                            
                            selected_menu_item = 0
                            in_load_variables_menu = False
                            break
                elif event.key == pygame.K_PERIOD:
                    pass
                elif event.key == pygame.K_COMMA:
                    pass
                elif event.key == pygame.K_m:
                    pass
                elif event.key == pygame.K_f:
                    pass
            if menu:
                pass
            if not menu:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        dragging = True
                        prev_mouse_pos = event.pos
                        break
                    
                    if list_x <= mouse_x < list_x + list_width:
                        if event.button == 4:  # Scroll up
                            scroll_y = min(scroll_y + scroll_speed, 0)
                        elif event.button == 5:  # Scroll down
                            scroll_y -= scroll_speed
                        break
                    if list_x + list_width <= mouse_x:
                        if event.button == 4:  # Scroll up
                            target_zoom = np.clip(target_zoom * (1 + zoom_speed), 0.1, 10)

                        elif event.button == 5:  # Scroll down
                            target_zoom = np.clip(target_zoom * (1 - zoom_speed), 0.1, 10)
                        break
                        
                        
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left click
                        dragging = False
                    break
                elif event.type == pygame.MOUSEMOTION:
                    if dragging:
                        dx, dy = event.pos[0] - prev_mouse_pos[0], event.pos[1] - prev_mouse_pos[1]
                        camera_x += dx
                        camera_y += dy
                        prev_mouse_pos = event.pos
                    break
                    
        # Update zoom with lerp for smoothness
        prev_zoom = zoom
        zoom = zoom + lerp_speed * (target_zoom - zoom)

        # Update camera position to keep the center of the screen fixed during zooming
        if prev_zoom != zoom:
            camera_x += (1 - zoom / prev_zoom) * (screen_width / 2 - camera_x)
            camera_y += (1 - zoom / prev_zoom) * (screen_height / 2 - camera_y)
        
        if not menu:
            simulation.step(tick)

        #render game objects
        text = "tick = %d" % tick
        text_surface = font.render(text, True, (0, 255, 255))
        person_surface = pygame.Surface((cell_size, cell_size))
        person_surface.fill(person_colour)
        food_surface = pygame.Surface((cell_size, cell_size))
        food_surface.fill(food_colour)
        egg_surface = pygame.Surface((cell_size, cell_size))
        egg_surface.fill(egg_colour)
        empty_surface = pygame.Surface((cell_size, cell_size))
        empty_surface.fill(empty_colour)
        wall_surface = pygame.Surface((cell_size, cell_size))
        wall_surface.fill(wall_colour)
        
        person_surface = pygame.transform.scale(person_surface, (int(person_surface.get_width() * zoom) + 1, int(person_surface.get_height() * zoom) + 1))
        food_surface = pygame.transform.scale(food_surface, (int(food_surface.get_width() * zoom) + 1, int(food_surface.get_height() * zoom) + 1))
        egg_surface = pygame.transform.scale(egg_surface, (int(egg_surface.get_width() * zoom) + 1, int(egg_surface.get_height() * zoom) + 1))
        empty_surface = pygame.transform.scale(empty_surface, (int(empty_surface.get_width() * zoom) + 1, int(empty_surface.get_height() * zoom) + 1))
        wall_surface = pygame.transform.scale(wall_surface, (int(wall_surface.get_width() * zoom) + 1, int(wall_surface.get_height() * zoom) + 1))


        #fill background
        screen.fill(background_colour)
        info_box_displayed = False
        info_box_text = ""
            
        # display game objects
        for x in range(grid_size):
            for y in range(grid_size):
            
                object_size = cell_size * zoom
                if object_size < 1:

                    color = empty_colour

                    if isinstance(grid[x][y], Person):
                        color = person_colour # yellow

                    if isinstance(grid[x][y], Egg):
                        color = egg_colour

                    if isinstance(grid[x][y], Food):
                        color = food_colour

                    if isinstance(grid[x][y], Wall):
                        color = wall_colour

                    
                    screen.set_at((int((x * cell_size * zoom)+camera_x), int((y * cell_size * zoom)+camera_y)), color)

                else:

                    object_surface = empty_surface

                    if isinstance(grid[x][y], Person):
                        object_surface = person_surface

                    if isinstance(grid[x][y], Egg):
                        object_surface = egg_surface

                    if isinstance(grid[x][y], Food):
                        object_surface = food_surface

                    if isinstance(grid[x][y], Wall):
                        object_surface = wall_surface

                    screen.blit(object_surface, ((x * cell_size * zoom) + camera_x, (y * cell_size * zoom) + camera_y))
                
                # display information box
                if (x * cell_size * zoom + camera_x) <= mouse_x <= (x * cell_size * zoom + camera_x + object_size) and \
                (y * cell_size * zoom + camera_y) <= mouse_y <= (y * cell_size * zoom + camera_y + object_size):
                    draw = False
                    if isinstance(grid[x][y], Person):
                        info_box_text = "Lifespan: {} \nHunger: {} \nRepro Timer: {}".format(grid[x][y].lifespan, grid[x][y].hunger, grid[x][y].reproduction_timer)
                        draw = True
                    if isinstance(grid[x][y], Food):
                        info_box_text = "Food value: {}".format(grid[x][y].food_value)
                        draw = True
                    if isinstance(grid[x][y], Egg):
                        info_box_text = "Eggness: 1"
                        draw = True
                   
                
                    # Update the info_box position
                    info_box_surface.set_alpha(200)
                    info_box_rect.center = (int((x * cell_size * zoom) + camera_x), int((y * cell_size * zoom) + camera_y) - 60)
                    if draw:
                        info_box_displayed = True
                

                        
        if info_box_displayed:
            # Render and blit the info_box
            screen.blit(info_box_surface, info_box_rect)

            # Render and blit the text with newlines handled
            lines = info_box_text.split('\n')
            text_vertical_spacing = 5
            for i, line in enumerate(lines):
                info_box_text_surface = info_box_font.render(line, True, (255, 255, 255))
                text_y = info_box_rect.y + 10 + i * (info_box_font.get_height() + text_vertical_spacing)
                screen.blit(info_box_text_surface, (info_box_rect.x + 10, text_y))

        #ui blits
        if show_UI:
            simulation.people.sort(key=lambda x: x.lifespan)
            for i, person in enumerate(simulation.people):
                if i == 0:
                    text_pre = ', '.join(prop for prop in displayed_properties + [str(zoom)])
                    text = list_font.render(text_pre, True, text_colour)
                    screen.blit(text, (list_x, i + scroll_y))
                text = list_font.render(person.get_properties(displayed_properties), True, text_colour)
                screen.blit(text, (list_x, 20 + i * 20 + scroll_y))
                
            screen.blit(text_surface, (screen_width/2, 0))

            graph_scaling = 0.3

            for graph_line in graph_history:
                # pygame.draw.rect(screen, (255,255,255), (graph_line.last, screen_height - graph_line.pop, 1, 1), 1)
                pygame.draw.rect(screen, (50,50,70), (graph_line.last, screen_height - (graph_line.pop * graph_scaling), 1, (graph_line.pop * graph_scaling)), 1)
            
        if tick % 10 == 0:
            if last > screen_width:
                last = 0
            for gl in graph_history:
                if gl.last == last:
                    graph_history.remove(gl)
            new_gl = Graph_Line(last,len(simulation.people))
            graph_history.append(new_gl)
            
            last = last + 1

        if menu:
            if in_load_variables_menu:
                pass
            if in_change_variables_menu:
                pass
            if in_save_variables_menu:
                pass

            if not in_change_variables_menu and not in_load_variables_menu and not in_save_variables_menu:
                menu_items = main_menu_items
            
            if selected_menu_item >= len(menu_items):
                selected_menu_item = selected_menu_item - 1

            
            
            for count in range(len(menu_items)):
                
                menu_colour = menu_background_colour
                if selected_menu_item == count:
                    menu_colour = menu_select_colour
                
                menu_item_width = screen_width/5
                menu_item_height = screen_height/20
                padding = 10
                text = font.render(menu_items[count], True, text_colour)
                pygame.draw.rect(screen, menu_colour, ((screen_width/2)-(menu_item_width/2), (screen_height/5) + (count * (menu_item_height + padding)) - ((menu_item_height + padding) * selected_menu_item), menu_item_width, menu_item_height), 0)
                screen.blit(text, ((screen_width/2)-(menu_item_width/2), (screen_height/5)+ (count * (menu_item_height + padding)) - ((menu_item_height + padding) * selected_menu_item)))
                count = count + 1

            if not change_config_selected_item == None:
                text = font.render(str(change_config_selected_value), True, text_colour)
                pygame.draw.rect(screen, menu_colour, ((screen_width/2)-(menu_item_width/2), 50, menu_item_width, menu_item_height), 0)
                screen.blit(text, ((screen_width/2)-(menu_item_width/2), 50))
                text = font.render(new_val, True, text_colour)
                screen.blit(text, ((screen_width/2)-(menu_item_width/2)+50, 50))
                
        if len(simulation.people) < 11:
            
            for x in range(num_people):
                parent1 = simulation.people[random.randint(0, len(simulation.people) - 1)]
                parent2 = simulation.people[random.randint(0, len(simulation.people) - 1)]
                while parent1.uuid == parent2.uuid:
                    parent2 = simulation.people[random.randint(0, len(simulation.people) - 1)]
                x, y = random.randint(0, len(grid) - 1), random.randint(0, len(grid[0]) - 1)
                grid[x, y] = None
                grid.append(Person(x, y, parent1, parent2, parent1.initial_lifespan, parent1.reproduction_cooldown_initial, parent1.repro_min_hunger_percent * parent1.birth_hunger, parent1.hunger_max, parent1.repro_min_hunger_percent, parent1.move_cost))
        pygame.display.flip()

        tick = tick + 1
        clock.tick(framerate)
        

    pygame.quit()

if __name__ == "__main__":
    main()
