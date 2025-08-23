import pygame, json
from typing import Literal
from random import randint, randrange, gauss
from math import ceil
from lib.constants import *
from lib.network import Network


def change_weights(pesos: list) -> list:
    new_pesos = []
    for c in range(len(pesos)):
        camada = []
        for n in range(len(pesos[c])):
            neuronio = []
            for i in range(len(pesos[c][n])):
                neuronio.append(min(max(pesos[c][n][i] + gauss(0, SIGMA), -1), 1))
            camada.append(neuronio)
        new_pesos.append(camada)
    return new_pesos


def change_biases(biases: list) -> list:
    new_biases = []
    for c in range(len(biases)):
        camada = []
        for n in range(len(biases[c])):
            camada.append(min(max(biases[c][n] + gauss(0, SIGMA), -1), 1))
        new_biases.append(camada)
    return new_biases


def draw_text(screen: pygame.Surface, msg: str, tam: int, pos: tuple[int, int],  color: str, shadow: None | tuple[int, int, str] = None) -> None:
    font = pygame.font.SysFont('04b19', tam)
    if shadow:
        msg_formatada = font.render(msg, True, shadow[2])
        screen.blit(msg_formatada, msg_formatada.get_rect(topleft=(pos[0] + shadow[0], pos[1] + shadow[1])))
    msg_formatada = font.render(msg, True, color)
    screen.blit(msg_formatada, msg_formatada.get_rect(topleft=pos))


def draw_network(screen: pygame.Surface, network: Network, parameters: list, pos: tuple):
    def colorir(value: float) -> list:
        if value > 20 or value < -20:
            value = 20
        return [abs(value / 20 * 255), 0, 0]
    

    maior = max(len(item) for item in network.get_layers())
    if maior < network._Network__inputs['numbers_of_neurons']:
        maior = network._Network__inputs['numbers_of_neurons']

    # links
    for n in range(network._Network__inputs['numbers_of_neurons']):
        for c in range(len(network.get_layers()[0])):
            pygame.draw.line(screen, colorir(parameters[n] * network.get_weights()[0][n][c]), (-60 + pos[0], n * 40 + pos[1] + (maior - network._Network__inputs['numbers_of_neurons']) * 20), (pos[0], c * 40 + pos[1] + (maior - len(network.get_layers()[0])) * 20), 2)
            # pygame.draw.line(screen, 'green' if parameters[n] > 0 else 'red', (-60 + pos[0], n * 40 + pos[1] + (maior - network._Network__inputs) * 20), (pos[0], c * 40 + pos[1] + (maior - len(network.get_layers()[0])) * 20), 2)
    for l in range(len(network.get_layers()) - 1):
        for n in range(len(network.get_layers()[l])):
            for c in range(len(network.get_layers()[l + 1])):
                pygame.draw.line(screen, 'red' if network.get_layers()[l].get_neurons()[n].value * network.get_weights()[l][n][c] > 0 else 'black', (l * 60 + pos[0], n * 40 + pos[1] + (maior - len(network.get_layers()[l])) * 20), ((l + 1) * 60 + pos[0], c * 40 + pos[1] + (maior - len(network.get_layers()[l + 1])) * 20), 2)
                # pygame.draw.line(screen, 'green' if network.get_layers()[l].get_neurons()[n].value > 0 else 'red', (l * 60 + pos[0], n * 40 + pos[1] + (maior - len(network.get_layers()[l])) * 20), ((l + 1) * 60 + pos[0], c * 40 + pos[1] + (maior - len(network.get_layers()[l + 1])) * 20), 2)

    # Neurons
    for n in range(network._Network__inputs['numbers_of_neurons']):
        pygame.draw.circle(screen, colorir(parameters[n]), (-60 + pos[0], n * 40 + pos[1] + (maior - network._Network__inputs['numbers_of_neurons']) * 20), 10)
        # pygame.draw.circle(screen, 'green' if parameters[n] > 0 else 'red', (-60 + pos[0], n * 40 + pos[1] + (maior - network._Network__inputs['numbers_of_neurons']) * 20), 10)
        pygame.draw.circle(screen, 'black', (-60 + pos[0], n * 40 + pos[1] + (maior - network._Network__inputs['numbers_of_neurons']) * 20), 10, 2)
    for idx_l, layer in enumerate(network.get_layers()):
        for idx_c, neuron in enumerate(layer.get_neurons()):
            pygame.draw.circle(screen, 'red' if neuron.value > 0 else 'black', (idx_l * 60 + pos[0], idx_c * 40 + pos[1] + (maior - len(layer)) * 20), 10)
            # pygame.draw.circle(screen, 'green' if neuron.value > 0 else 'red', (idx_l * 60 + pos[0], idx_c * 40 + pos[1] + (maior - len(layer)) * 20), 10)
            pygame.draw.circle(screen, 'black', (idx_l * 60 + pos[0], idx_c * 40 + pos[1] + (maior - len(layer)) * 20), 10, 2)    


class SpriteSheet:
    def __init__(self, sprite_sheet: str, size: tuple):
        sprite_sheet = pygame.image.load(sprite_sheet).convert_alpha()
        width = sprite_sheet.get_width()
        height = sprite_sheet.get_height()
        self.sprite = []
        for l in range(0, height, size[1]):
            for c in range(0, width, size[0]):
                self.sprite.append(sprite_sheet.subsurface((c, l, size[0], size[1])))
    
    def get_sprites(self) -> list:
        return self.sprite


class Dino:
    def __init__(self, pos: tuple, weights: list):
        self.x, self.y = pos
        self.status = 'RUN'
        self.alive = True
        self.img_idx = 0
        self.speed_y = 0
        self.sprites_init()
        self.select_animation()
        self.display_config()
        self.jumped = False
        self.squated = False
        with open(os.path.join(PATH_DATA, 'network_structure.json'), encoding='UTF-8') as f:
            data = json.load(f)
            data['weights'] = weights
            self.network = Network(**data)
    
    def sprites_init(self):
        dino = randint(1, 4)
        sprite_run = SpriteSheet(os.path.join(PATH_SPRITES, f'Dino {dino}', 'Dino run.png'), (84, 64))
        sprite_jump = SpriteSheet(os.path.join(PATH_SPRITES, f'Dino {dino}', 'Dino jump.png'), (84, 64))
        sprite_squat = SpriteSheet(os.path.join(PATH_SPRITES, f'Dino {dino}', 'Dino squat.png'), (84, 64))
        self.sprites = {
            'RUN': (sprite_run, 0.15),
            'JUMP': (sprite_jump, 1),
            'SQUAT': (sprite_squat, 0.15),
        }
    
    def select_animation(self):
        self.sprites_used = self.sprites[self.status][0].get_sprites()
        self.animation_speed = self.sprites[self.status][1]
    
    def liven(self):
        self.img_idx += self.animation_speed
        if self.img_idx >= len(self.sprites_used):
            self.img_idx = 0
        
    def display_config(self):
        self.rect = self.sprites_used[int(self.img_idx)].get_rect(center=(self.x, self.y))
        if self.status in ['RUN', 'JUMP']:
            self.rect_colision = pygame.Rect(self.x - 27, self.y - 29, 54, 48)
        else:
            self.rect_colision = pygame.Rect(self.x - 37, self.y - 5, 74, 24)
        
    def think(self, speed: int, obstacles: 'ObstacleGroup') -> Literal['JUMP', 'SQUAT', 'RUN']:
        min_distance = WIDTH
        height = 0
        for obstacle in obstacles.obstacles:
            distance = obstacle.rect_colision.centerx - self.rect_colision.centerx
            if distance < min_distance and distance > -50:
                min_distance = distance
                height = HEIGHT - obstacle.rect_colision.centery
        
        actions = self.network.forward([speed, height, min_distance])
        if not all(actions):
            if actions[0]:
                return 'JUMP'
            if actions[1]:
                return 'SQUAT'
        return 'RUN'
    
    def update(self, speed: int, obstacles: 'ObstacleGroup'):
        status_before = self.status

        action = self.think(speed, obstacles)
        if action == 'JUMP' and not self.jumped and not self.squated:
            self.speed_y = -15
            self.jumped = True
            self.status = 'JUMP'
        if action == 'SQUAT':
            self.status = 'SQUAT'
            if not self.squated:
                self.speed_y = 5
            self.squated = True
        else:
            self.squated = False
        if not self.jumped and not self.squated:
            self.status = 'RUN'
        
        if self.status != status_before:
            self.img_idx = 0
        
        self.speed_y += GRAVITY
        self.dy = self.speed_y
        self.liven()
        self.colision(speed, obstacles)
        self.y += self.dy
    
    def colision(self, speed: int, obstacles: 'ObstacleGroup'):
        if self.rect_colision.bottom + ceil(self.speed_y) >= 510:
            self.dy = 510 - self.rect_colision.bottom
            self.speed_y = 0
            self.jumped = False
        
        for obstacle in obstacles.obstacles:
            if obstacle.rect_colision.colliderect(self.rect_colision.x + speed, self.rect_colision.y + ceil(self.speed_y), self.rect_colision.width, self.rect_colision.height):
                self.alive = False
                break
    
    def draw(self, screen: pygame.Surface):
        self.select_animation()
        self.display_config()
        screen.blit(self.sprites_used[int(self.img_idx)], self.rect)


class SingleCactus:
    def __init__(self, pos: tuple):
        self.x, self.y = pos
        self.alive = False
        self.display_config()
    
    def display_config(self):
        self.image = pygame.image.load(os.path.join(PATH_SPRITES, 'Obstacle/One Cactus.png')).convert_alpha()
        self.rect_colision = self.image.get_rect(bottomleft=(self.x, self.y))
    
    def update(self, speed: int):
        self.rect_colision.x -= speed
        if self.rect_colision.right < 0:
            self.rect_colision.left = WIDTH
            self.alive = False
    
    def draw(self, screen: pygame.Surface):
        screen.blit(self.image, self.rect_colision)


class DoubleCactus:
    def __init__(self, pos: tuple):
        self.x, self.y = pos
        self.alive = False
        self.display_config()
    
    def display_config(self):
        self.image = pygame.image.load(os.path.join(PATH_SPRITES, 'Obstacle/Two Cactus.png')).convert_alpha()
        self.rect_colision = self.image.get_rect(bottomleft=(self.x, self.y))
    
    def update(self, speed: int):
        self.rect_colision.x -= speed
        if self.rect_colision.right < 0:
            self.rect_colision.left = WIDTH
            self.alive = False
    
    def draw(self, screen: pygame.Surface):
        screen.blit(self.image, self.rect_colision)


class QuadrupleCactus:
    def __init__(self, pos: tuple):
        self.x, self.y = pos
        self.alive = False
        self.display_config()
    
    def display_config(self):
        self.image = pygame.image.load(os.path.join(PATH_SPRITES, 'Obstacle/Four Cactus.png')).convert_alpha()
        self.rect_colision = self.image.get_rect(bottomleft=(self.x, self.y))
    
    def update(self, speed: int):
        self.rect_colision.x -= speed
        if self.rect_colision.right < 0:
            self.rect_colision.left = WIDTH
            self.alive = False
    
    def draw(self, screen: pygame.Surface):
        screen.blit(self.image, self.rect_colision)


class FlyingDino:
    def __init__(self, x: int):
        self.x, self.y = x, randrange(325, 450, 25)
        self.alive = False
        self.img_idx = 0
        self.display_config()
    
    def display_config(self):
        self.sprites_used = SpriteSheet(os.path.join(PATH_SPRITES, 'Obstacle/Flying dino.png'), (64, 53)).get_sprites()
        self.rect = self.sprites_used[int(self.img_idx)].get_rect(topleft=(self.x, self.y))
        self.rect_colision = pygame.Rect(self.rect.x + 4, self.rect.y + 10, 56, 28)
        self.animation_speed = 0.15
    
    def liven(self):
        self.img_idx += self.animation_speed
        if self.img_idx >= len(self.sprites_used):
            self.img_idx = 0

    def update(self, speed: int):
        self.rect.x -= speed
        if self.rect_colision.right < 0:
            self.rect.left = WIDTH
            self.y = randrange(325, 450, 25)
            self.alive = False
        self.rect_colision = pygame.Rect(self.rect.x + 4, self.rect.y + 10, 56, 28)
    
    def draw(self, screen: pygame.Surface):
        self.liven()
        screen.blit(self.sprites_used[int(self.img_idx)], self.rect)


class ObstacleGroup:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.dx = 0
        self.obstacles = [
            SingleCactus((WIDTH, 518)),
            SingleCactus((WIDTH, 518)),
            DoubleCactus((WIDTH, 518)),
            DoubleCactus((WIDTH, 518)),
            QuadrupleCactus((WIDTH, 518)),
            QuadrupleCactus((WIDTH, 518)),
            FlyingDino(WIDTH),
            FlyingDino(WIDTH)
        ]
        self.send()
    
    def update(self, speed: int):
        self.dx += speed
        for obstacle in self.obstacles:
            if obstacle.alive:
                obstacle.update(speed)
        # if self.dx >= randrange(500 * ((speed - 3) / 2), 800 * ((speed - 3) / 2), 50):
        if self.dx >= randrange(500, 800, 50):
            self.dx = 0
            self.send()
    
    def send(self):
        sort = randint(0, 7)
        while self.obstacles[sort].alive:
            sort = randint(0, 7)
        self.obstacles[sort].alive = True
    
    def draw(self):
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)


class Floor:
    def __init__(self, pos: tuple):
        self.x, self.y = pos
        self.display_config()
    
    def display_config(self):
        self.image = SpriteSheet(os.path.join(PATH_SPRITES, 'ForeGround/Floor.png'), (2400, 26)).get_sprites()[0]
        self.rect = self.image.get_rect(topleft=(self.x, self.y))
    
    def update(self, speed: int):
        self.rect.x -= speed
        if self.rect.right < 0:
            self.rect.left = 2400 + self.rect.right
    
    def draw(self, screen: pygame.Surface):
        screen.blit(self.image, self.rect)


class FloorGroup:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.floors = [Floor((n * 2400, 500)) for n in range(2)]

    def update(self, speed: int):
        for floor in self.floors:
            floor.update(speed)
    
    def draw(self):
        for floor in self.floors:
            floor.draw(self.screen)


class Cloud:
    def __init__(self):
        self.x, self.y = randint(0, WIDTH), randint(50, 400)
        self.display_config()
    
    def display_config(self):
        self.image = pygame.image.load(os.path.join(PATH_SPRITES, 'Background/Cloud.png')).convert_alpha()
        self.rect = self.image.get_rect(center=(self.x, self.y))

    def update(self, speed: int):
        self.rect.x -= speed // 4
        if self.rect.right < 0:
            self.rect.left = WIDTH - self.rect.right
            self.rect.y = randint(100, 350)
    
    def draw(self, screen: pygame.Surface):
        screen.blit(self.image, self.rect)


class Montain:
    def __init__(self, start: int):
        self.x, self.y = start, 450
        self.display_config()
    
    def display_config(self):
        self.image = pygame.image.load(os.path.join(PATH_SPRITES, 'Background/Montain.png')).convert_alpha()
        self.rect = self.image.get_rect(topleft=(self.x, self.y))
    
    def update(self, speed: int):
        self.rect.x -= speed // 2
        if self.rect.right < 0:
            self.rect.left = self.rect.width + self.rect.right
    
    def draw(self, screen: pygame.Surface):
        screen.blit(self.image, self.rect)


class Background:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.clouds = [Cloud() for _ in range(10)]
        self.montains = [Montain(n * 2400) for n in range(2)]
    
    def update(self, speed: int):
        for cloud in self.clouds:
            cloud.update(speed)
        for montain in self.montains:
            montain.update(speed)
    
    def draw(self):
        for cloud in self.clouds:
            cloud.draw(self.screen)
        for montain in self.montains:
            montain.draw(self.screen)


class Game:
    def __init__(self, screen: pygame.Surface, amount: int):
        self.screen = screen
        self.amount = amount
        # with open(os.path.join(PATH_DATA, 'better.json')) as f:
        #     data = json.load(f)
        #     self.generation = data['generation']
        #     self.scroll_max = data['record']
        #     self.better_weights = data['weights']
        self.generation = 1
        self.evolution = 0
        self.scroll_max = 0
        self.better_weights = []
        self.init()
        self.start = False
    
    def init(self):
        self.dinos = [Dino((randint(WIDTH / 8, WIDTH / 3), 490), change_weights(self.better_weights)) for _ in range(self.amount)]
        self.obstacle_group = ObstacleGroup(self.screen)
        self.floor = FloorGroup(self.screen)
        self.background = Background(self.screen)
        self.distance = self.scrolled = 0
        self.better = self.better_weights[:]
        self.speed = 6
    
    def update(self):
        if any(pygame.key.get_pressed()) and not self.start:
            self.start = True
        if self.start:
            self.background.update(self.speed)
            self.obstacle_group.update(self.speed)
            self.floor.update(self.speed)
            idx = 0
            while idx < len(self.dinos):
                if self.dinos[idx].alive:
                    self.dinos[idx].update(self.speed, self.obstacle_group)
                    idx += 1
                else:
                    self.dinos.remove(self.dinos[idx])
            if len(self.dinos) == 1:
                self.better = self.dinos[0].network.get_weights()
            if not self.dinos:
                self.generation += 1
                if self.distance > self.scroll_max:
                    self.evolution += 1
                    self.scroll_max = self.distance
                    self.better_weights = self.better
                    # with open(os.path.join(PATH_DATA, 'better.json'), 'w') as f:
                    #     json.dump({'generation': self.generation, 'record': self.scroll_max, 'weights': change_weights(self.better_weights, [0, 0])}, f, indent=4)
                self.init()
            
            self.scrolled += self.speed
            self.distance += self.speed
            if self.scrolled >= 5000:
                self.scrolled = 0
                self.speed += 1
    
    def draw(self):
        self.background.draw()
        self.floor.draw()
        self.obstacle_group.draw()
        for dino in self.dinos:
            dino.draw(self.screen)
        draw_text(self.screen, f'Record: {self.scroll_max}', 30, (20, 20), '#535353')
        draw_text(self.screen, f'Distance: {self.distance}', 30, (20, 70), '#535353')
        draw_text(self.screen, f'Amount: {len([dino for dino in self.dinos if dino.alive])}', 30, (20, 120), '#535353')
        draw_text(self.screen, f'Generation: {self.generation}', 30, (20, 170), '#535353')
        draw_text(self.screen, f'Evolution: {self.evolution}', 30, (20, 220), '#535353')
        draw_text(self.screen, f'Speed: {self.speed - 5}', 30, (20, 270), '#535353')
        if not self.start:
            draw_text(self.screen, 'Press any key to start', 30, (WIDTH / 2, (HEIGHT - 100) / 2), '#535353')
        for dino in self.dinos:
            if dino.alive:
                min_distance = WIDTH
                height = 0
                for obstacle in self.obstacle_group.obstacles:
                    distance = obstacle.rect_colision.left - dino.rect_colision.right
                    if distance < min_distance and distance > 0:
                        min_distance = distance
                        height = HEIGHT - obstacle.rect_colision.centery
                draw_network(self.screen, dino.network, [self.speed, height, min_distance], (WIDTH - 100, 50))
                break
    
    def run(self):
        self.update()
        self.draw()
