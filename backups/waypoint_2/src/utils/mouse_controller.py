import pyautogui
import numpy as np
from screeninfo import get_monitors

class MouseController:
    def __init__(self):
        self.is_controlling = False
        self.smoothing_factor = 0.15  # Decreased from 0.3 to 0.15 for smoother movement
        self.last_position = None
        self._initialize_screen_bounds()

    def _initialize_screen_bounds(self):
        """Initialize screen boundaries"""
        monitor = get_monitors()[0]
        self.screen_width = monitor.width
        self.screen_height = monitor.height
        
        # Set up PyAutoGUI safety
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1

    def move_mouse(self, x, y):
        """Move mouse to specified coordinates with smoothing"""
        if not self.is_controlling:
            return

        # Ensure coordinates are within screen bounds
        x = np.clip(x, 0, self.screen_width)
        y = np.clip(y, 0, self.screen_height)

        # Apply smoothing if there's a previous position
        if self.last_position is not None:
            x = self.last_position[0] + (x - self.last_position[0]) * self.smoothing_factor
            y = self.last_position[1] + (y - self.last_position[1]) * self.smoothing_factor

        try:
            pyautogui.moveTo(x, y)
            self.last_position = (x, y)
        except pyautogui.FailSafeException:
            # Mouse hit screen corner - failsafe triggered
            self.is_controlling = False

    def start_control(self):
        """Start mouse control"""
        self.is_controlling = True
        self.last_position = None

    def stop_control(self):
        """Stop mouse control"""
        self.is_controlling = False
        self.last_position = None

    def set_smoothing(self, factor):
        """Set smoothing factor (0-1)"""
        self.smoothing_factor = np.clip(factor, 0, 1)

    def click(self):
        """Perform mouse click"""
        if self.is_controlling:
            pyautogui.click()

    def double_click(self):
        """Perform double click"""
        if self.is_controlling:
            pyautogui.doubleClick()

    def right_click(self):
        """Perform right click"""
        if self.is_controlling:
            pyautogui.rightClick()