from concurrent.futures import ProcessPoolExecutor
import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import uuid
import copy
import asyncio



class Person:
    def __init__(self, x, y, parent1, parent2, lifespan, reproduction_timer, birth_hunger, hunger_max):
        self.x, self.y = x, y
        self.lifespan, self.reproduction_timer = lifespan, random.randint(1, reproduction_timer)
        self.initial_lifespan = self.lifespan
        self.reproduction_cooldown_initial = reproduction_timer
        self.uuid = uuid.uuid4()
        self.hunger = birth_hunger
        self.hunger_max = hunger_max
        self.birth_hunger = birth_hunger
        

        if parent1 is None or parent2 is None:
            self.nn = self.create_neural_net()
        else:
            self.nn = self.combine_neural_nets(parent1.nn, parent2.nn)

    def get_properties(self, properties):
        return ', '.join(str(int(getattr(self, prop))) for prop in properties)

    def create_neural_net(self):
        nn_model = nn.Sequential(
            nn.Linear(27, 32),
            nn.ReLU(),
            nn.Linear(32, 43),
            nn.ReLU(),
            nn.Linear(43, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Softmax(dim=1)
        )
        return nn_model

    def combine_neural_nets(self, nn1, nn2):
        new_nn = self.create_neural_net()
        total_weights = sum([param.numel() for param in new_nn.parameters()])
        num_randomized_weights = round(0.01 * total_weights)

        weights_to_randomize = set(random.sample(range(total_weights), num_randomized_weights))
        weight_index = 0

        for child_param, parent1_param, parent2_param in zip(new_nn.parameters(), nn1.parameters(), nn2.parameters()):
            for i in range(child_param.data.numel()):
                if weight_index in weights_to_randomize:
                    # Totally randomize the weight
                    child_param.data.view(-1)[i] = torch.randn_like(child_param.data.view(-1)[i])
                    weights_to_randomize.remove(weight_index)
                else:
                    # Select the weight from either parent randomly and apply randomization factor up to 1%
                    src_param = random.choice([parent1_param, parent2_param])
                    child_param.data.view(-1)[i] = src_param.data.view(-1)[i] * (1 + 0.01 * random.uniform(-1, 1))
                weight_index += 1

        return new_nn


    
    def move(self, grid):
        directions = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]

        surroundings = self.get_surroundings(grid)

        # Pad or truncate the surroundings list
        surroundings_data = pad_or_truncate(surroundings, 24)

        input_tensor = torch.tensor([self.lifespan, self.reproduction_timer, self.hunger] + surroundings_data, dtype=torch.float32).unsqueeze(0)
        output = self.nn(input_tensor)
        
        move_idx = torch.argmax(output).item()
        dx, dy = directions[move_idx]
        grid_size = len(grid)
        new_x, new_y = (self.x + dx) % grid_size, (self.y + dy) % grid_size

        if isinstance(grid[new_x][new_y], Food):
            
           
            if self.hunger < self.hunger_max:
                self.hunger += grid[new_x][new_y].food_value
                grid[new_x][new_y] = None
        elif grid[new_x][new_y] is None:
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
                if self.hunger < (self.hunger_max * 0.50) or grid[px][py].hunger < (grid[px][py].hunger_max * 0.50):
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
                        if self.hunger < (self.hunger_max * 0.50) or grid[px][py].hunger < (grid[px][py].hunger_max * 0.50):
                            continue

                        self.hunger = self.hunger / 2

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

    def add_people(self, num_people, initial_lifespan, reproduction_timer, birth_hunger, max_hunger):
        people = []
        for _ in range(num_people):
            while True:
                x, y = random.randint(0, len(self.grid) - 1), random.randint(0, len(self.grid[0]) - 1)
                if self.grid[x][y] is None:
                    person = Person(x, y, None, None, initial_lifespan, reproduction_timer, birth_hunger, max_hunger)
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
                new_x, new_y = x + dx, y + dy
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



def hatch_egg(egg):
    parent1, parent2 = egg.parent1, egg.parent2
    child = Person(egg.x, egg.y, parent1, parent2, (parent1.initial_lifespan + parent2.initial_lifespan) / 2, parent1.reproduction_cooldown_initial, parent1.birth_hunger, parent1.hunger_max)
    return child

class Simulation:
    def __init__(self, grid, people, food_sources, replenish_rate):
        self.grid = grid
        self.people = people
        
        self.food_sources = food_sources
        self.initial_replenish_rate = replenish_rate
        self.replenish_rate = replenish_rate
        
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
        return data + [0] * (target_length - len(data))

def main():
    
    grid_size = 100
    num_people = 300
    initial_lifespan = 5000
    reproduction_timer = 100
    max_hunger = 2000
    birth_hunger = 1000
    food_value = 200
    replenish_rate = 1
    num_food_sources = 3
    food_production_rate = 1
    food_distribution_radius = 10

    grid = Grid(grid_size)
    people = grid.add_people(num_people, initial_lifespan, reproduction_timer, birth_hunger, max_hunger)
    
    food_sources = grid.add_food_sources(num_food_sources, food_production_rate, food_distribution_radius, food_value)
    for _ in food_sources:
        for _ in range(10):
            grid.generate_food(food_sources)

    simulation = Simulation(grid, people, food_sources, replenish_rate)

    person_colour = (255,255,0)
    food_colour = (0,255,0)
    background_colour = (0,0,0)
    egg_colour = (255, 248, 220)

    show_UI = False
    steps_per_second = 60
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

    while running:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_i:
                    show_UI = not show_UI
                elif event.key == pygame.K_DOWN:
                    pass
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    pass
                elif event.key == pygame.K_RIGHT:
                    pass
                elif event.key == pygame.K_LEFT:
                    pass
                elif event.key == pygame.K_PERIOD:
                    pass
                elif event.key == pygame.K_COMMA:
                    pass
                elif event.key == pygame.K_m:
                    menu = not menu
                elif event.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()
            if menu:
                pass
            if not menu:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        dragging = True
                        prev_mouse_pos = event.pos
                    
                    if list_x <= mouse_x < list_x + list_width:
                        if event.button == 4:  # Scroll up
                            scroll_y = min(scroll_y + scroll_speed, 0)
                        elif event.button == 5:  # Scroll down
                            scroll_y -= scroll_speed
                    if list_x + list_width <= mouse_x:
                        if event.button == 4:  # Scroll up
                            target_zoom = np.clip(target_zoom * (1 + zoom_speed), 0.1, 10)
                        elif event.button == 5:  # Scroll down
                            target_zoom = np.clip(target_zoom * (1 - zoom_speed), 0.1, 10)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left click
                        dragging = False
                elif event.type == pygame.MOUSEMOTION:
                    if dragging:
                        dx, dy = event.pos[0] - prev_mouse_pos[0], event.pos[1] - prev_mouse_pos[1]
                        camera_x += dx
                        camera_y += dy
                        prev_mouse_pos = event.pos
                    
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
        
        person_surface = pygame.transform.scale(person_surface, (int(person_surface.get_width() * zoom), int(person_surface.get_height() * zoom)))
        food_surface = pygame.transform.scale(food_surface, (int(food_surface.get_width() * zoom), int(food_surface.get_height() * zoom)))
        egg_surface = pygame.transform.scale(egg_surface, (int(egg_surface.get_width() * zoom), int(egg_surface.get_height() * zoom)))


        #fill background
        screen.fill(background_colour)
        info_box_displayed = False
            
        # display game objects
        for x in range(grid_size):
            for y in range(grid_size):
                if isinstance(grid[x][y], Person) or isinstance(grid[x][y], Food) or isinstance(grid[x][y], Egg):
                    object_size = cell_size * zoom
                    if object_size < 1:

                        if isinstance(grid[x][y], Person):
                            color = person_colour # yellow

                        if isinstance(grid[x][y], Egg):
                            color = egg_colour

                        if isinstance(grid[x][y], Food):
                            color = food_colour

                        screen.set_at((int((x * cell_size * zoom)+camera_x), int((y * cell_size * zoom)+camera_y)), color)

                    else:

                        if isinstance(grid[x][y], Person):
                            object_surface = person_surface

                        if isinstance(grid[x][y], Egg):
                            object_surface = egg_surface

                        if isinstance(grid[x][y], Food):
                            object_surface = food_surface

                        screen.blit(object_surface, ((x * cell_size * zoom) + camera_x, (y * cell_size * zoom) + camera_y))
                    
                    # display information box
                    if (x * cell_size * zoom + camera_x) <= mouse_x <= (x * cell_size * zoom + camera_x + object_size) and \
                    (y * cell_size * zoom + camera_y) <= mouse_y <= (y * cell_size * zoom + camera_y + object_size):
                        if isinstance(grid[x][y], Person):
                            info_box_text = "Lifespan: {} \nHunger: {} \nRepro Timer: {}".format(grid[x][y].lifespan, grid[x][y].hunger, grid[x][y].reproduction_timer)
                        if isinstance(grid[x][y], Food):
                            info_box_text = "Food value: {}".format(grid[x][y].food_value)
                        if isinstance(grid[x][y], Egg):
                            info_box_text = "Eggness: 1"
                    
                        # Update the info_box position
                        info_box_surface.set_alpha(200)
                        info_box_rect.center = (int((x * cell_size * zoom) + camera_x), int((y * cell_size * zoom) + camera_y) - 60)
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
                    text = list_font.render(text_pre, True, (255,255,255))
                    screen.blit(text, (list_x, i + scroll_y))
                text = list_font.render(person.get_properties(displayed_properties), True, (255,255,255))
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

        pygame.display.flip()

        tick = tick + 1
        clock.tick(30)
        

    pygame.quit()

if __name__ == "__main__":
    main()
