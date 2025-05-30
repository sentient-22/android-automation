import json
import yaml
import traceback
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
from appium import webdriver
from appium.webdriver.common.appiumby import AppiumBy

from .base_agent import BaseAgent
from ..utils.gemini_client import GeminiClient
from ..utils.helpers import capture_screenshot, get_ui_hierarchy
from ..data.dataset import VLMDataset
from ..utils.logger import logger

class LLMAgent(BaseAgent):
    def __init__(self, config_path: str = "config/gemini_config.yaml"):
        super().__init__()
        
        # Load Gemini config
        self.gemini_config = self._load_gemini_config(config_path)
        
        # Initialize Gemini client with API key from config
        self.gemini = GeminiClient(
            model_name=self.gemini_config.get('model', 'gemini-pro'),
            api_key=self.gemini_config.get('api_key')
        )
        
        # Setup directories
        self.data_dir = Path("data")
        self.screenshot_dir = self.data_dir / "screenshots"
        self.dataset_dir = self.data_dir / "vlm_dataset"
        
        # Create necessary directories
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset
        self.dataset = VLMDataset(self.dataset_dir)
        
        # Initialize Appium driver
        self.driver = None
        
        # Load Appium config
        self.appium_config = self._load_appium_config()
        
        logger.info(f"LLMAgent initialized with data directory: {self.data_dir.absolute()}")
    
    def _load_gemini_config(self, config_path: str) -> Dict[str, Any]:
        """Load Gemini configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Gemini config file not found at {config_path}, using defaults")
            return {}
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('gemini', {})
        except Exception as e:
            logger.error(f"Error loading Gemini config: {e}")
            return {}
    
    def _load_appium_config(self) -> Dict[str, Any]:
        """Load Appium configuration from YAML file."""
        config_path = Path("config/appium_config.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Appium config file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _get_desired_capabilities(self) -> Dict[str, Any]:
        """Get desired capabilities for Appium session."""
        # Get base capabilities from config
        device_config = self.appium_config.get('device', {})
        
        # Required capabilities
        caps = {
            'platformName': device_config.get('platform_name', 'Android'),
            'appium:automationName': device_config.get('automation_name', 'UiAutomator2'),
            'appium:deviceName': device_config.get('device_name', 'Android Emulator'),
            'appium:appPackage': device_config.get('app_package', 'com.android.settings'),
            'appium:appActivity': device_config.get('app_activity', '.Settings'),
            'appium:noReset': device_config.get('no_reset', True),
            'appium:fullReset': device_config.get('full_reset', False)
        }
        
        # Add platform version if specified
        if 'platform_version' in device_config:
            caps['appium:platformVersion'] = device_config['platform_version']
            
        # Add any additional capabilities
        if 'additional_capabilities' in device_config:
            for key, value in device_config['additional_capabilities'].items():
                caps[f'appium:{key}'] = value
                
        return caps
        
    def start(self) -> bool:
        """Initialize the agent and start an Appium session.
        
        Returns:
            bool: True if the session was started successfully, False otherwise
        """
        try:
            # Get the Appium server URL from config or use default
            server_url = self.appium_config.get('appium', {}).get('base_url', 'http://localhost:4723/wd/hub')
            logger.info(f"Connecting to Appium server at: {server_url}")
            
            # Get desired capabilities
            caps = self._get_desired_capabilities()
            logger.info(f"Using capabilities: {caps}")
            
            # Import Android options
            from appium.options.android import UiAutomator2Options
            
            # Initialize the Appium driver with Android options
            options = UiAutomator2Options()
            options.load_capabilities(caps)
            
            logger.info("Initializing WebDriver...")
            self.driver = webdriver.Remote(
                command_executor=server_url,
                options=options
            )
            
            if not self.driver:
                logger.error("Failed to initialize WebDriver - driver is None")
                return False
                
            # Verify session was created
            session_id = self.driver.session_id
            if not session_id:
                logger.error("Failed to create session - no session ID")
                return False
                
            logger.info(f"Appium session started successfully with ID: {session_id}")
            
            # Set implicit wait
            implicit_wait = self.appium_config.get('timeouts', {}).get('implicit', 10)
            self.driver.implicitly_wait(implicit_wait)
            logger.info(f"Set implicit wait to {implicit_wait} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Appium session: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            if hasattr(self, 'driver') and self.driver:
                try:
                    self.driver.quit()
                except Exception as quit_error:
                    logger.error(f"Error while quitting driver: {quit_error}")
            self.driver = None
            return False
    
    def stop(self) -> None:
        """Clean up resources and stop the Appium session."""
        try:
            if self.driver:
                self.driver.quit()
                logger.info("Appium session stopped")
        except Exception as e:
            logger.error(f"Error stopping Appium session: {e}")
        finally:
            self.driver = None
    
    def capture_state(self, save_screenshot: bool = True) -> Dict[str, Any]:
        """Capture the current state of the device.
        
        Args:
            save_screenshot: Whether to save the screenshot to disk
            
        Returns:
            Dict containing state information
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            screenshot_path = self.screenshot_dir / f"screenshot_{timestamp}.png"
            
            # Capture screenshot and UI hierarchy
            screenshot_base64 = capture_screenshot(self.driver, str(screenshot_path) if save_screenshot else None)
            ui_hierarchy = get_ui_hierarchy(self.driver)
            
            state = {
                "screenshot": str(screenshot_path) if save_screenshot else None,
                "screenshot_base64": screenshot_base64,
                "ui_hierarchy": ui_hierarchy,
                "timestamp": timestamp
            }
            
            logger.debug(f"Captured state at {timestamp}")
            return state
            
        except Exception as e:
            logger.error(f"Error capturing state: {e}")
            raise
    
    def generate_action(
        self,
        task: str,
        save_to_dataset: bool = True
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate the next action based on the current state and task.
        
        Args:
            task: The task to perform
            save_to_dataset: Whether to save this interaction to the dataset
                
        Returns:
            Tuple of (action_dict, state_dict)
        """
        state = self.capture_state(save_screenshot=save_to_dataset)
        
        # Prepare the prompt
        system_prompt = """You are an AI assistant helping to automate Android device interactions. 
        Your task is to analyze the current screen and determine the best action to complete the given task.
        
        Guidelines:
        1. Be precise and specific in identifying UI elements
        2. Prefer direct actions over multiple steps
        3. For system apps like Settings, look for direct access points
        4. If an action doesn't work, try alternative approaches
        5. Use the most reliable selectors available (resource-id > content-desc > text > class)
        
        Common package names:
        - Google Play Store: com.android.vending
        - Settings: com.android.settings
        - Chrome: com.android.chrome
        - Gmail: com.google.android.gm
        - Play Store: com.android.vending
        - Phone: com.android.dialer
        - Messages: com.google.android.apps.messaging
        - Camera: com.android.camera2
        - Files: com.android.documentsui
        - Clock: com.google.android.deskclock
        - Calendar: com.google.android.calendar
        """
        
        user_prompt = f"""
        TASK: {task}
        
        CURRENT SCREEN:
        {state['ui_hierarchy']}
        
        INSTRUCTIONS:
        1. Analyze the UI elements and identify the best action to complete the task
        2. Consider the element's properties (text, content-desc, resource-id) to ensure accuracy
        3. If the task is complete, return {{"action": "complete", "reasoning": "Task completed"}}
        4. If you need to open an app, use the 'open_app' action with the package name
        5. If unsure, provide your best guess with lower confidence
        
        RESPONSE FORMAT (JSON):
        {{
            "action": "click|long_click|type|swipe|back|home|open_app|etc",
            "element": {{
                "resource-id": "value",  # Preferred selector
                "content-desc": "value",  # Second choice
                "text": "value",          # Third choice
                "class": "value"           # Last resort
            }},
            "package_name": "com.example.app",  # Required for open_app action
            "text": "text to type",  # Only for type action
            "confidence": 0.0-1.0,
            "reasoning": "Your step-by-step reasoning"
        }}
        
        IMPORTANT: Only respond with valid JSON, no other text or markdown formatting.
        """
        
        # Get the screenshot data
        image_path = state['screenshot']
        
        try:
            # Get generation config from YAML or use defaults
            gen_config = self.gemini_config.get("generation_config", {
                "max_tokens": 1024,
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40
            })
            
            # Get the response from Gemini
            response = self.gemini.generate_text(
                prompt=user_prompt,
                **gen_config
            )
            
            # Try to extract JSON from markdown code block if present
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()
            
            action = json.loads(response)
            action['raw_response'] = response  # Keep original for debugging
            
            # Save to dataset if enabled
            if save_to_dataset and image_path:
                self._add_to_dataset(
                    task=task,
                    state=state,
                    action=action,
                    image_path=image_path
                )
            
            return action, state
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON response: {e}"
            logger.error(error_msg)
            logger.error(f"Response was: {response}")
            
            # Create an error action
            error_action = {
                "action": "error",
                "error": error_msg,
                "raw_response": response
            }
            
            # Save to dataset if enabled
            if save_to_dataset and image_path:
                self._add_to_dataset(
                    task=task,
                    state=state,
                    action=error_action,
                    image_path=image_path
                )
            
            return error_action, state
            
        except Exception as e:
            error_msg = f"Error generating action: {e}"
            logger.error(error_msg, exc_info=True)
            
            # Create an error action
            error_action = {
                "action": "error",
                "error": error_msg
            }
            
            # Save to dataset if enabled
            if save_to_dataset and image_path:
                self._add_to_dataset(
                    task=task,
                    state=state,
                    action=error_action,
                    image_path=image_path
                )
            
            return error_action, state
    
    def _add_to_dataset(
        self,
        task: str,
        state: Dict[str, Any],
        action: Dict[str, Any],
        image_path: str
    ) -> Optional[str]:
        """Add the current interaction to the dataset.
        
        Args:
            task: The original task
            state: Current device state
            action: Generated action
            image_path: Path to the screenshot
            
        Returns:
            Sample ID if successful, None otherwise
        """
        try:
            # Format the instruction and output for the dataset
            instruction = f"""Based on the current screen and the task "{task}", 
            what should be the next action?"""
            
            # Format the output as the action JSON
            output = json.dumps({
                "action": action.get('action'),
                "element": action.get('element', {}),
                "text": action.get('text'),
                "reasoning": action.get('reasoning', '')
            }, indent=2)
            
            # Add to dataset
            sample_id = self.dataset.add_sample(
                image_path=image_path,
                instruction=instruction,
                input_text=task,
                output_text=output,
                ui_hierarchy=state.get('ui_hierarchy'),
                action=action,
                metadata={
                    "timestamp": state.get('timestamp'),
                    "task": task,
                    "action_type": action.get('action')
                }
            )
            
            logger.debug(f"Added interaction to dataset with ID: {sample_id}")
            return sample_id
            
        except Exception as e:
            logger.error(f"Error adding to dataset: {e}", exc_info=True)
            return None
    
    def _is_app_installed(self, package_name: str) -> bool:
        """Check if an app is installed on the device.
        
        Args:
            package_name: The package name of the app to check
            
        Returns:
            bool: True if the app is installed, False otherwise
        """
        try:
            return self.driver.is_app_installed(package_name)
        except Exception as e:
            logger.warning(f"Error checking if app {package_name} is installed: {e}")
            return False
            
    def _find_element(self, element_info: Dict[str, str]):
        """Find a UI element based on the provided element information.
        
        Args:
            element_info: Dictionary containing element locators (resource-id, content-desc, text, class)
            
        Returns:
            WebElement if found, None otherwise
        """
        if not element_info or not isinstance(element_info, dict):
            logger.warning("Invalid element information provided")
            return None
            
        # Special handling for Play Store search field
        if element_info.get('class') == 'android.widget.EditText' and not element_info.get('resource-id'):
            try:
                # Try common Play Store search field resource IDs first
                play_store_search_ids = [
                    'com.android.vending:id/search_box_idle_text',
                    'com.android.vending:id/search_box_startup_text',
                    'com.android.vending:id/search_box_text_input',
                    'search_box_idle_text',
                    'search_box_startup_text',
                    'search_box_text_input'
                ]
                
                # Try resource IDs first
                for res_id in play_store_search_ids:
                    try:
                        element = self.driver.find_element(AppiumBy.ID, res_id)
                        if element and element.is_displayed():
                            logger.debug(f"Found Play Store search field with ID: {res_id}")
                            return element
                    except Exception:
                        continue
                        
                # Try to find any visible EditText (likely search field)
                elements = self.driver.find_elements(AppiumBy.CLASS_NAME, 'android.widget.EditText')
                for element in elements:
                    try:
                        if element.is_displayed() and element.is_enabled():
                            logger.debug("Found visible and enabled EditText (likely search field)")
                            return element
                    except Exception:
                        continue
                        
                # Try finding by content description containing 'search'
                try:
                    elements = self.driver.find_elements(AppiumBy.XPATH, "//*[contains(@content-desc, 'search') or contains(@content-desc, 'Search')]")
                    for element in elements:
                        if element.is_displayed() and element.is_enabled():
                            logger.debug("Found element with search in content description")
                            return element
                except Exception as e:
                    logger.debug(f"Error searching by content description: {e}")
                    
            except Exception as e:
                logger.debug(f"Error in special handling for search field: {e}")
        
        # Try different locator strategies in order of preference
        locators = [
            (AppiumBy.ID, element_info.get('resource-id')),  # resource-id
            (AppiumBy.ACCESSIBILITY_ID, element_info.get('content-desc')),  # content-desc
            (AppiumBy.ANDROID_UIAUTOMATOR, 
                f'new UiSelector().text(\"{element_info.get("text")}\")' if element_info.get('text') else None),  # text
            (AppiumBy.CLASS_NAME, element_info.get('class')),  # class
            (AppiumBy.XPATH, element_info.get('xpath')),  # xpath as fallback
            # Additional locators for common cases
            (AppiumBy.ANDROID_UIAUTOMATOR, 
                'new UiSelector().className(\"android.widget.EditText\").instance(0)' 
                if element_info.get('class') == 'android.widget.EditText' else None),
            # Try finding any clickable element that might be a search field
            (AppiumBy.ANDROID_UIAUTOMATOR, 
                'new UiSelector().clickable(true).className(\"android.widget.EditText\")'
                if element_info.get('class') == 'android.widget.EditText' else None)
        ]
        
        for by, value in locators:
            if not value:
                continue
                
            try:
                # Handle resource-id format (might include package name)
                if by == AppiumBy.ID and ':' in value:
                    value = value.split(':')[-1]
                
                if by == AppiumBy.ANDROID_UIAUTOMATOR and isinstance(value, str):
                    elements = self.driver.find_elements(by, value)
                    for element in elements:
                        try:
                            if element.is_displayed() and element.is_enabled():
                                logger.debug(f"Found element using {by}: {value}")
                                return element
                        except Exception:
                            continue
                else:
                    element = self.driver.find_element(by, value)
                    if element and element.is_displayed() and element.is_enabled():
                        logger.debug(f"Found element using {by}: {value}")
                        return element
                        
            except Exception as e:
                logger.debug(f"Element not found with {by}={value}: {e}")
                continue
        
        # Last resort: Try to find any clickable element that might be a search field
        if element_info.get('class') == 'android.widget.EditText':
            try:
                elements = self.driver.find_elements(AppiumBy.CLASS_NAME, 'android.widget.EditText')
                for element in elements:
                    try:
                        if element.is_displayed() and element.is_enabled():
                            logger.debug("Found any visible and enabled EditText as last resort")
                            return element
                    except Exception:
                        continue
            except Exception as e:
                logger.debug(f"Last resort search failed: {e}")
                
        logger.warning(f"Could not find element with any locator: {element_info}")
        return None
    
    def execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute the specified action on the device.
        
        Args:
            action: Action dictionary containing action type and parameters
            
        Returns:
            bool: True if the action was executed successfully, False otherwise
        """
        if not action or 'action' not in action:
            logger.error("No action specified")
            return False
            
        action_type = action.get('action')
        logger.info(f"Executing action: {action_type}")
        
        try:
            if action_type == 'click':
                element = self._find_element(action.get('element', {}))
                if element:
                    element.click()
                    logger.info(f"Clicked on element: {action.get('element')}")
                    return True
                logger.warning(f"Element not found for click: {action.get('element')}")
                return False
                
            elif action_type == 'long_click':
                element = self._find_element(action.get('element', {}))
                if element:
                    self.driver.execute_script('mobile: longClickGesture', {
                        'elementId': element.id,
                        'duration': 1000  # 1 second
                    })
                    logger.info(f"Long clicked on element: {action.get('element')}")
                    return True
                logger.warning(f"Element not found for long click: {action.get('element')}")
                return False
                
            elif action_type == 'type':
                element = self._find_element(action.get('element', {}))
                if element and 'text' in action:
                    element.clear()
                    element.send_keys(action['text'])
                    logger.info(f"Typed '{action['text']}' into element: {action.get('element')}")
                    return True
                logger.warning(f"Element not found or no text provided for type action: {action}")
                return False
                
            elif action_type == 'swipe':
                # Implement swipe logic
                logger.warning("Swipe action not yet implemented")
                return False
                
            elif action_type == 'back':
                self.driver.back()
                logger.info("Pressed back button")
                return True
                
            elif action_type == 'home':
                self.driver.press_keycode(3)  # KEYCODE_HOME
                logger.info("Pressed home button")
                return True
                
            elif action_type == 'complete':
                logger.info("Task marked as complete")
                return True
                
            elif action_type == 'open_app':
                package_name = action.get('package_name')
                if not package_name:
                    logger.error("No package name provided for open_app action")
                    return False
                
                try:
                    # First check if the app is installed
                    if not self._is_app_installed(package_name):
                        logger.warning(f"App {package_name} is not installed")
                        
                        # If it's not the Play Store, try to open the Play Store page for the app
                        if package_name != 'com.android.vending':
                            try:
                                # First ensure Play Store is installed
                                if not self._is_app_installed('com.android.vending'):
                                    logger.error("Play Store is not installed on the device")
                                    return False
                                
                                # Method 1: Try using ADB command first (most reliable)
                                try:
                                    play_store_url = f"market://details?id={package_name}"
                                    self.driver.execute_script('mobile: shell', {
                                        'command': 'am',
                                        'args': ['start', '-a', 'android.intent.action.VIEW', '-d', play_store_url]
                                    })
                                    logger.info(f"Opened Play Store page for {package_name} via ADB")
                                    return True
                                except Exception as adb_error:
                                    logger.debug(f"ADB method failed, trying deep link: {adb_error}")
                                    
                                    # Method 2: Try deep link
                                    try:
                                        self.driver.execute_script('mobile: deepLink', {
                                            'url': play_store_url,
                                            'package': 'com.android.vending'
                                        })
                                        logger.info(f"Opened Play Store page for {package_name} via deep link")
                                        return True
                                    except Exception as deeplink_error:
                                        logger.debug(f"Deep link failed, trying direct launch: {deeplink_error}")
                                        
                                        # Method 3: Just open Play Store
                                        try:
                                            self.driver.activate_app('com.android.vending')
                                            logger.info("Opened Play Store")
                                            return True
                                        except Exception as e2:
                                            logger.error(f"All methods to open Play Store failed: {e2}")
                                            return False
                                    
                            except Exception as e:
                                logger.error(f"Error opening Play Store: {e}")
                                return False
                        return False
                    
                    # If we get here, the app is installed, so try to open it
                    self.driver.activate_app(package_name)
                    logger.info(f"Opened app with package: {package_name}")
                    return True
                    
                except Exception as e:
                    logger.error(f"Failed to open app {package_name}: {e}")
                    return False
            
            else:
                logger.warning(f"Unsupported action type: {action_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing action {action_type}: {str(e)}", exc_info=True)
            return False
