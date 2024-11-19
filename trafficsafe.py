
import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
GRID_ROWS = 3
GRID_COLS = 3
BOX_HEIGHT = SCREEN_HEIGHT // GRID_ROWS
BOX_WIDTH = SCREEN_WIDTH // GRID_COLS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
CAR_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

# Car dimensions
CAR_WIDTH = 50
CAR_HEIGHT = 100

# Screen setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Car Simulation with Traffic Rules")

# Frame rate
clock = pygame.time.Clock()
FPS = 60

class Car:
    def __init__(self, x, y, speed, color):
        self.x = x
        self.y = y
        self.speed = speed
        self.color = color
        self.rect = pygame.Rect(self.x, self.y, CAR_WIDTH, CAR_HEIGHT)

    def move(self, stop_at_bottom, cars):
        if stop_at_bottom and self.y >= SCREEN_HEIGHT - CAR_HEIGHT:
            return

        # Collision avoidance
        if self.is_too_close(cars):
            return  # Stop if too close to the car ahead

        self.y += self.speed
        self.rect.y = self.y

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)

    def is_too_close(self, cars):
        """Check if the car is too close to another car ahead in the same lane."""
        for other_car in cars:
            if other_car == self:
                continue
            if self.x == other_car.x and other_car.y > self.y:
                distance = other_car.y - (self.y + CAR_HEIGHT)
                if distance < 50:  # Safe gap of 10 pixels
                    return True
        return False

def generate_cars(cars, max_cars=10):
    if len(cars) < max_cars:
        attempts = 0
        while attempts < 20:
            x = random.choice([BOX_WIDTH // 2, 3 * BOX_WIDTH // 2, 5 * BOX_WIDTH // 2])
            y = random.randint(-CAR_HEIGHT * 3, -CAR_HEIGHT)
            speed = random.randint(3, 6)
            color = random.choice(CAR_COLORS)
            new_car = Car(x, y, speed, color)
            if not any(new_car.rect.colliderect(car.rect) for car in cars):
                cars.append(new_car)
                break
            attempts += 1

def get_car_count_in_bottom_boxes(cars):
    count = 0
    for car in cars:
        if SCREEN_HEIGHT // 3 <= car.y < SCREEN_HEIGHT:
            count += 1
    return count

def draw_grid():
    for i in range(1, GRID_ROWS):
        pygame.draw.line(screen, BLACK, (0, i * SCREEN_HEIGHT // GRID_ROWS), (SCREEN_WIDTH, i * SCREEN_HEIGHT // GRID_ROWS), 2)
    for j in range(1, GRID_COLS):
        pygame.draw.line(screen, BLACK, (j * SCREEN_WIDTH // GRID_COLS, 0), (j * SCREEN_WIDTH // GRID_COLS, SCREEN_HEIGHT), 2)

def draw_signal(signal_color):
    pygame.draw.circle(screen, signal_color, (SCREEN_WIDTH - 50, 50), 20)

def main():
    cars = []
    run = True
    signal_color = RED
    stop_at_bottom = True
    delay_timer = 0
    green_signal_active = False
    green_signal_timer = 0  # Timer for green signal duration

    while run:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        car_count_in_bottom_boxes = get_car_count_in_bottom_boxes(cars)

        # Traffic signal logic
        if signal_color == RED and car_count_in_bottom_boxes > 5:
            delay_timer += 1
            if delay_timer >= FPS * 3:  # 6-second delay for red signal
                signal_color = GREEN
                green_signal_active = True
                delay_timer = 0
        elif signal_color == GREEN:
            green_signal_timer += 1  # Increase the timer while green is active
            if green_signal_timer >= FPS * 5:  # 5 seconds for green signal
                signal_color = RED
                stop_at_bottom = True
                green_signal_active = False
                green_signal_timer = 0  # Reset the timer after switching to red

        if signal_color == GREEN:
            stop_at_bottom = False

        # Generate new cars
        if random.randint(1, 20) == 1:
            generate_cars(cars, max_cars=15)

        # Move and draw cars
        for car in cars[:]:
            car.move(stop_at_bottom, cars)
            car.draw(screen)
            if car.y > SCREEN_HEIGHT:
                cars.remove(car)

        draw_grid()
        draw_signal(signal_color)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
