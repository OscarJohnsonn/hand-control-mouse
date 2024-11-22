import pyautogui
import time
import numpy as np
from PIL import ImageGrab
import cv2
import random
from selenium import webdriver
from selenium.webdriver.common.by import By

# Define the screen region dynamically using template matching
def get_game_region(template_path='image.png'):
    # Capture the entire screen
    screen = np.array(ImageGrab.grab())
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    
    # Load the template image
    template = cv2.imread(template_path, 0)
    w, h = template.shape[::-1]
    
    # Perform template matching
    res = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    
    # Get the top-left corner of the matched region
    for pt in zip(*loc[::-1]):
        x1, y1 = pt
        break
    
    # Define the bottom-right corner
    x2, y2 = x1 + w, y1 + h
    
    width = x2 - x1
    height = y2 - y1

    return (x1, y1, width, height)

# Action keys
ACTIONS = ['left', 'right', 'up']

# Simulation Settings
NUM_GENERATIONS = 150
POPULATION_SIZE = 200  # Increased population size
MUTATION_RATE = 0.5
STEPS_PER_EPISODE = 150

# Simulate key presses for controlling the car
def press_key(key, duration=0.01):  # Reduced duration for faster key presses
    pyautogui.keyDown(key)
    time.sleep(duration)
    pyautogui.keyUp(key)

# Capture the game screen
def get_game_state(region):
    screenshot = ImageGrab.grab(bbox=region).resize((region[2]//2, region[3]//2))  # Reduce resolution
    return np.array(screenshot)

# Process the game state to detect road edges using edge detection
def process_game_state(game_state):
    gray = cv2.cvtColor(game_state, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Use smaller kernel for faster processing
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    return edges

# Define the reward function
def get_reward(driver):
    # Extract the speedometer value
    speedometer = driver.find_element(By.CSS_SELECTOR, '.stat-value.svelte-11y65kj').text
    speed = float(speedometer)
    
    # Extract the distance covered
    distance_covered = driver.find_elements(By.CSS_SELECTOR, '.stat-value.svelte-11y65kj')[1].text
    distance = float(distance_covered)
    
    # Reward based on speed and distance covered
    reward = speed * distance
    return reward

# Detect the warning on the screen
def detect_warning(region, template_path='warning.png'):
    screenshot = np.array(ImageGrab.grab(bbox=region))
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    
    template = cv2.imread(template_path, 0)
    w, h = template.shape[::-1]
    
    res = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    
    if len(loc[0]) > 0:
        return True
    return False

# Make a decision based on the game state
def decide_action(edges):
    height, width = edges.shape
    left = edges[:, :width // 2].sum()
    right = edges[:, width // 2:].sum()

    if left > right:
        return 'left'
    elif right > left:
        return 'right'
    return 'up'  # Go straight

# Play one episode of the game
def play_game(action, region, driver):
    for _ in range(STEPS_PER_EPISODE):
        screenshot = get_game_state(region=region)
        edges = process_game_state(screenshot)
        reward = get_reward(driver)
        
        if detect_warning(region):
            reward = -1  # Punish the AI
            press_key('r', duration=0.1)  # Press 'r' to place the AI back on the road
        
        press_key(action, duration=0.05)  # Reduced duration for faster key presses
        if reward == -1:  # If off the road, stop the game
            break
    return reward

# Evolution strategy to evolve actions
def random_action():
    return random.choice(ACTIONS)

def mutate(action):
    if random.random() < MUTATION_RATE:
        return random.choice(ACTIONS)
    return action

def select_best(scores, population):
    sorted_population = [x for _, x in sorted(zip(scores, population), reverse=True)]
    return sorted_population[:POPULATION_SIZE // 2]

# Main evolutionary loop
def main():
    # Initialize the Selenium WebDriver
    driver = webdriver.Chrome()
    driver.get('URL_OF_THE_GAME_PAGE')  # Replace with the actual URL of the game page
    
    game_region = get_game_region()
    population = [random_action() for _ in range(POPULATION_SIZE)]
    
    for generation in range(NUM_GENERATIONS):
        scores = []
        for action in population:
            reward = play_game(action, region=game_region, driver=driver)
            scores.append(reward)
        
        best_actions = select_best(scores, population)
        population = best_actions[:]
        
        # Mutate the best actions
        for i in range(len(best_actions), POPULATION_SIZE):
            action = mutate(random.choice(best_actions))
            population.append(action)
    
    driver.quit()

if __name__ == "__main__":
    main()