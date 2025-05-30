from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
import yaml
import os
from typing import Dict, Any

class AppiumClient:
    def __init__(self, config_path: str = 'config/appium_config.yaml'):
        self.config = self._load_config(config_path)
        self.driver = None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def start_session(self):
        """Initialize the Appium driver with the loaded configuration"""
        options = UiAutomator2Options()
        
        # Set basic capabilities
        options.set_capability('platformName', self.config['device']['platform_name'])
        options.set_capability('platformVersion', self.config['device']['platform_version'])
        options.set_capability('deviceName', self.config['device']['device_name'])
        options.set_capability('automationName', self.config['device']['automation_name'])
        options.set_capability('noReset', self.config['device']['no_reset'])
        options.set_capability('fullReset', self.config['device']['full_reset'])
        
        # Set app package and activity if provided
        if self.config['device'].get('app_package'):
            options.set_capability('appPackage', self.config['device']['app_package'])
        if self.config['device'].get('app_activity'):
            options.set_capability('appActivity', self.config['device']['app_activity'])
        
        # Initialize the driver
        url = f"http://{self.config['appium']['host']}:{self.config['appium']['port']}"
        self.driver = webdriver.Remote(command_executor=url, options=options)
        
        # Set timeouts
        self.driver.implicitly_wait(self.config['timeouts']['implicit'])
        return self.driver
    
    def stop_session(self):
        """Close the Appium session"""
        if self.driver:
            self.driver.quit()
    
    def get_ui_hierarchy(self) -> str:
        """Get the current UI hierarchy as XML"""
        if not self.driver:
            raise RuntimeError("Driver not initialized. Call start_session() first.")
        return self.driver.page_source
    
    def take_screenshot(self, save_path: str = None) -> str:
        """Take a screenshot and optionally save it"""
        if not self.driver:
            raise RuntimeError("Driver not initialized. Call start_session() first.")
        
        screenshot = self.driver.get_screenshot_as_png()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(screenshot)
        return screenshot
