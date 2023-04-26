import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import uuid


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
            nn.Linear(9, 32),
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
        input_tensor = torch.tensor([self.lifespan, self.reproduction_timer, self.hunger] + surroundings, dtype=torch.float32).unsqueeze(0)
    
        output = self.nn(input_tensor)
        
        move_idx = torch.argmax(output).item()
        dx, dy = directions[move_idx]
        grid_size = len(grid)
        new_x, new_y = (self.x + dx) % grid_size, (self.y + dy) % grid_size

        if isinstance(grid[new_x][new_y], Food):
            max_hunger = self.hunger_max
            food_value = 500
            if self.hunger < (max_hunger - food_value):
                self.hunger += food_value

            grid[new_x][new_y] = None
        elif grid[new_x][new_y] is None:
            self.update_position(grid, new_x, new_y)
            

    def get_surroundings(self, grid):
        surroundings = [0] * 6
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = self.x + dx, self.y + dy
            if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]):
                if isinstance(grid[new_x][new_y], Food):
                    surroundings[0] += 1
                elif isinstance(grid[new_x][new_y], Person):
                    surroundings[1] += 1
                elif grid[new_x][new_y] is None:
                    surroundings[2] += 1
            else:
                surroundings[3] += 1
        surroundings[4] = self.lifespan
        surroundings[5] = self.hunger
        return surroundings

    def update_position(self, grid, new_x, new_y):
        grid[self.x][self.y] = None
        self.x, self.y = new_x, new_y
        grid[new_x][new_y] = self

    def reproduce(self, grid, x, y):

        other_person = grid[x][y]
        if other_person is None:
            return None
        
        if self.reproduction_timer > 0:
            return None
        
        # no repro if your hunger is lower than minimum repro hunger
        if self.hunger < (self.hunger_max * 0.75) and other_person.hunger < (other_person.hunger_max * 0.75):
            return None
        for nx, ny in get_valid_neighbors(x, y, grid.size):
            if grid[nx][ny] == None:
                self.hunger = self.hunger / 2
                child = Person(nx, ny, self, grid[x][y], (self.initial_lifespan + grid[x][y].initial_lifespan) / 2, self.reproduction_cooldown_initial, self.birth_hunger, self.hunger_max)
                grid[nx][ny] = child
                self.reproduction_timer = self.reproduction_cooldown_initial
                return child
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
                    genetic_code = ''.join(random.choices('ATCG', k=8))
                    person = Person(x, y, None, None, initial_lifespan, reproduction_timer, birth_hunger, max_hunger)
                    self.grid[x][y] = person
                    people.append(person)
                    break
        return people

    def add_food_sources(self, num_food_sources, food_production_rate, food_distribution_radius):
        food_sources = []
        for _ in range(num_food_sources):
            while True:
                x, y = random.randint(0, len(self.grid) - 1), random.randint(0, len(self.grid[0]) - 1)
                if self.grid[x][y] is None:
                    food_source = FoodSource(x, y, food_production_rate, food_distribution_radius)
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
    def __init__(self, x, y, food_production_rate, food_distribution_radius):
        self.x = x
        self.y = y
        self.food_production_rate = food_production_rate
        self.food_distribution_radius = food_distribution_radius

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
                    food = Food(x, y)
                    food_generated.append(food)
                    break
        return food_generated

class Graph_Line:
    def __init__(self, last, pop):
        self.last = last
        self.pop = pop

class Simulation:
    def __init__(self, grid, people, food_sources, replenish_rate):
        self.grid = grid
        self.people = people
        
        self.food_sources = food_sources
        self.initial_replenish_rate = replenish_rate
        self.replenish_rate = replenish_rate
       
    def step(self, tick):
        new_children = []
        for person in self.people:
            child = None
            person.move(self.grid)
            for x, y in get_valid_neighbors(person.x, person.y, self.grid.size):
                if isinstance(self.grid[x][y], Person):
                    child = person.reproduce(self.grid, x, y)
                    if child is not None:
                        break
            if child is not None:
                new_children.append(child)
            if person.decrease_lifespan_and_cooldown_and_hunger(self.grid):
                self.people.remove(person)
        self.people.extend(new_children)

        if tick % 1 == 0:
            self.grid.move_sources(self.food_sources)

        if self.replenish_rate <= 0:
            self.food_list = self.grid.generate_food(self.food_sources)
            self.replenish_rate = self.initial_replenish_rate
        if self.replenish_rate > 0:
            self.replenish_rate -= 1



class Food:
    def __init__(self, x, y):
        self.x, self.y = x, y
       
       
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


def main():
    #food is weird, look at food replenish func for details, uses magic num
    grid_size = 100
    num_people = 200
    initial_lifespan = 5000
    reproduction_timer = 100
    max_hunger = 4000
    birth_hunger = 3000
    
    replenish_rate = 1

    grid = Grid(grid_size)
    people = grid.add_people(num_people, initial_lifespan, reproduction_timer, birth_hunger, max_hunger)
    
    food_sources = grid.add_food_sources(num_food_sources=2, food_production_rate=1, food_distribution_radius=10)
    for _ in food_sources:
        for _ in range(10):
            grid.generate_food(food_sources)

    simulation = Simulation(grid, people, food_sources, replenish_rate)

    person_colour = (255,255,0)
    food_colour = (0,255,0)
    background_colour = (0,0,0)

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

    while running:

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
                    mouse_x, mouse_y = pygame.mouse.get_pos()
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

        
        person_surface = pygame.transform.scale(person_surface, (int(person_surface.get_width() * zoom), int(person_surface.get_height() * zoom)))
        food_surface = pygame.transform.scale(food_surface, (int(food_surface.get_width() * zoom), int(food_surface.get_height() * zoom)))


        #fill background
        screen.fill(background_colour)

        # display game objects
        for x in range(grid_size):
            for y in range(grid_size):
                if isinstance(grid[x][y], Person):
                    object_size = cell_size * zoom
                    if object_size < 1:
                        if person_surface.get_size() != (0, 0):
                            screen.set_at((int((x * cell_size * zoom)+camera_x), int((y * cell_size * zoom)+camera_y)), person_surface.get_at((0, 0)))
                        else:
                            screen.set_at((int((x * cell_size * zoom)+camera_x), int((y * cell_size * zoom)+camera_y)), (255, 255, 0))  # yellow
                    else:
                        screen.blit(person_surface, ((x * cell_size * zoom) + camera_x, (y * cell_size * zoom) + camera_y))
                elif isinstance(grid[x][y], Food):
                    object_size = cell_size * zoom
                    if object_size < 1:
                        if food_surface.get_size() != (0, 0):
                            screen.set_at((int((x * cell_size * zoom)+camera_x), int((y * cell_size * zoom)+camera_y)), food_surface.get_at((0, 0)))
                        else:
                            screen.set_at((int((x * cell_size * zoom)+camera_x), int((y * cell_size * zoom)+camera_y)), (0, 255, 0))  # green
                    else:
                        screen.blit(food_surface, ((x * cell_size * zoom)+camera_x, (y * cell_size * zoom)+camera_y))

        for i, person in enumerate(simulation.people):
            if i == 0:
                text_pre = ', '.join(prop for prop in displayed_properties + [str(zoom)])
                text = list_font.render(text_pre, True, (255,255,255))
                screen.blit(text, (list_x, i + scroll_y))
            text = list_font.render(person.get_properties(displayed_properties), True, (255,255,255))
            screen.blit(text, (list_x, 20 + i * 20 + scroll_y))
        
        #ui blits
        if show_UI:
            screen.blit(text_surface, (0, 0))

            for graph_line in graph_history:
                pygame.draw.rect(screen, (255,255,255), (graph_line.last, (((grid_size * cell_size) - graph_line.pop  )/10)+(((grid_size * cell_size)/10)*9),1, grid_size * cell_size), 1)
            
        if tick % 10 == 0:
            if last > grid_size * cell_size:
                last = 0
            for gl in graph_history:
                if gl.last == last:
                    graph_history.remove(gl)
            new_gl = Graph_Line(last,len(simulation.people))
            graph_history.append(new_gl)
            
            last = last + 1

        pygame.display.flip()

        tick = tick + 1
        clock.tick(steps_per_second)
        

    pygame.quit()

if __name__ == "__main__":
    main()
