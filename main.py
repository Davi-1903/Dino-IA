from lib.classes import *


class DinoIA:
    def __init__(self):
        pygame.init()
        self.screen_config()
        self.loop()
    
    def screen_config(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Dino - IA')
        pygame.display.set_icon(pygame.image.load(os.path.join(PATH_SPRITES, 'icon.png')))
        self.clock = pygame.time.Clock()
    
    def loop(self):
        game = Game(self.screen, 2000) # Quantidade de dinossauros
        while True:
            self.screen.fill('white')
            self.clock.tick(FPS)
            self.events()
            game.run()
            pygame.display.update()
    
    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()


if __name__ == '__main__':
    DinoIA()
