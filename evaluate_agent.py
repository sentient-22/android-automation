import os
import time
import torch
import logging
from typing import List, Dict, Any
from appium import webdriver
from appium.webdriver.common.appiumby import AppiumBy
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AndroidAgent:
    def __init__(self, model_path: str, device_name: str = 'emulator-5554'):
        """Initialize the Android agent with Qwen-VL model."""
        self.device_name = device_name
        self.driver = None
        self.model = None
        self.processor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Load model and processor
        self._load_model(model_path)
        
    def _load_model(self, model_path: str):
        """Load the Qwen-VL model and processor."""
        logger.info(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        logger.info("Model loaded successfully")
    
    def start_appium_session(self):
        """Start Appium session with Android emulator."""
        desired_caps = {
            'platformName': 'Android',
            'deviceName': self.device_name,
            'automationName': 'UiAutomator2',
            'newCommandTimeout': 3600,
            'adbExecTimeout': 50000,
            'autoGrantPermissions': True,
            'noReset': False,
            'fullReset': False,
            'appPackage': 'com.android.settings',
            'appActivity': '.Settings',
        }
        
        self.driver = webdriver.Remote('http://localhost:4723/wd/hub', desired_caps)
        logger.info("Appium session started")
    
    def capture_screenshot(self) -> Image.Image:
        """Capture screenshot from the device."""
        screenshot = self.driver.get_screenshot_as_png()
        return Image.frombytes('RGB', 
                            (self.driver.get_window_size()['width'], 
                             self.driver.get_window_size()['height']), 
                            screenshot, 'raw')
    
    def get_ui_elements(self) -> List[Dict[str, Any]]:
        """Extract UI elements from the current screen."""
        elements = []
        for element in self.driver.find_elements(AppiumBy.XPATH, "//*"):
            try:
                elements.append({
                    'text': element.text if element.text else "",
                    'class': element.get_attribute("class"),
                    'bounds': element.rect,
                    'clickable': element.get_attribute("clickable") == 'true',
                    'enabled': element.get_attribute("enabled") == 'true',
                    'selected': element.get_attribute("selected") == 'true',
                    'focusable': element.get_attribute("focusable") == 'true',
                })
            except Exception as e:
                logger.warning(f"Error extracting element: {e}")
        return elements
    
    def process_screen(self, task: str) -> str:
        """Process the current screen and generate action."""
        # Capture screenshot
        image = self.capture_screenshot()
        
        # Get UI elements
        ui_elements = self.get_ui_elements()
        
        # Format prompt
        prompt = f"""You are an AI assistant controlling an Android device. 
Current task: {task}

UI Elements (text, class, bounds, clickable):
"""
        for i, element in enumerate(ui_elements[:20]):  # Limit to first 20 elements for brevity
            prompt += f"{i+1}. {element['text']} | {element['class']} | {element['bounds']} | Clickable: {element['clickable']}\n"
        
        prompt += "\nWhat should be the next action? (click [number], type [text], or say [message]): "
        
        # Generate response
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
        )
        
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()
    
    def execute_action(self, action: str):
        """Execute the specified action on the device."""
        try:
            action = action.lower().strip()
            
            if action.startswith("click"):
                # Extract element number
                try:
                    element_num = int(action.split()[1]) - 1
                    elements = self.driver.find_elements(AppiumBy.XPATH, "//*")
                    if 0 <= element_num < len(elements):
                        elements[element_num].click()
                        logger.info(f"Clicked element {element_num + 1}")
                except (IndexError, ValueError):
                    logger.warning("Invalid click command")
            
            elif action.startswith("type"):
                # Extract text to type
                text = ' '.join(action.split()[1:])
                self.driver.press_keycode(84)  # Focus on input
                self.driver.hide_keyboard()
                self.driver.find_element(AppiumBy.CLASS_NAME, "android.widget.EditText").send_keys(text)
                logger.info(f"Typed: {text}")
            
            elif action.startswith("swipe"):
                # Simple swipe down
                size = self.driver.get_window_size()
                start_x = size['width'] // 2
                start_y = int(size['height'] * 0.8)
                end_y = int(size['height'] * 0.2)
                self.driver.swipe(start_x, start_y, start_x, end_y, 400)
                logger.info("Swiped down")
            
            elif action.startswith("home"):
                self.driver.press_keycode(3)  # Home button
                logger.info("Pressed home button")
            
            elif action.startswith("back"):
                self.driver.press_keycode(4)  # Back button
                logger.info("Pressed back button")
            
            time.sleep(1)  # Small delay between actions
            
        except Exception as e:
            logger.error(f"Error executing action: {e}")
    
    def run_task(self, task: str, max_steps: int = 10):
        """Run a task with the agent."""
        logger.info(f"Starting task: {task}")
        
        for step in range(max_steps):
            logger.info(f"\n--- Step {step + 1} ---")
            
            # Process screen and get action
            action = self.process_screen(task)
            logger.info(f"Action: {action}")
            
            # Execute the action
            self.execute_action(action)
            
            # Check if task is complete
            if "task complete" in action.lower():
                logger.info("Task completed successfully!")
                break
        else:
            logger.warning("Reached maximum number of steps")
    
    def close(self):
        """Close the Appium session."""
        if self.driver:
            self.driver.quit()
            logger.info("Appium session closed")


def main():
    # Initialize the agent
    model_path = "./output"  # Path to your trained model
    agent = AndroidAgent(model_path=model_path)
    
    try:
        # Start Appium session
        agent.start_appium_session()
        
        # Example tasks
        tasks = [
            "Open Wi-Fi settings and connect to a network",
            "Take a screenshot of the home screen",
            "Open the camera app and take a photo"
        ]
        
        # Run tasks
        for task in tasks:
            agent.run_task(task)
            time.sleep(2)  # Small delay between tasks
    
    except Exception as e:
        logger.error(f"Error during execution: {e}")
    
    finally:
        # Clean up
        agent.close()


if __name__ == "__main__":
    main()
