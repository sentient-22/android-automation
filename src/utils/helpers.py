"""Helper functions for Android automation."""
import base64
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from appium.webdriver import Remote
from PIL import Image
import io


def capture_screenshot(driver: Remote, save_path: Optional[str] = None) -> str:
    """Capture a screenshot and optionally save it to disk.
    
    Args:
        driver: Appium WebDriver instance
        save_path: Path to save the screenshot (optional)
        
    Returns:
        Base64 encoded screenshot
    """
    try:
        # Take screenshot
        screenshot = driver.get_screenshot_as_base64()
        
        # Save to file if path is provided
        if save_path:
            img_data = base64.b64decode(screenshot)
            with open(save_path, 'wb') as f:
                f.write(img_data)
                
        return screenshot
        
    except Exception as e:
        raise Exception(f"Failed to capture screenshot: {e}")


def get_ui_hierarchy(driver: Remote) -> str:
    """Get the UI hierarchy from the current screen.
    
    Args:
        driver: Appium WebDriver instance
        
    Returns:
        XML string of the UI hierarchy
    """
    try:
        return driver.page_source
    except Exception as e:
        raise Exception(f"Failed to get UI hierarchy: {e}")


def find_element_by_attributes(driver: Remote, attributes: Dict[str, str]):
    """Find an element using multiple attributes.
    
    Args:
        driver: Appium WebDriver instance
        attributes: Dictionary of attributes to search for
        
    Returns:
        WebElement if found, None otherwise
    """
    from appium.webdriver.common.appiumby import AppiumBy
    
    # Try different locator strategies in order of reliability
    locators = [
        (AppiumBy.ID, attributes.get('resource-id')),  # Most reliable
        (AppiumBy.ACCESSIBILITY_ID, attributes.get('content-desc')),
        (AppiumBy.XPATH, f"//*[contains(@text, '{attributes.get('text')}')]") if attributes.get('text') else None,
        (AppiumBy.CLASS_NAME, attributes.get('class')),
    ]
    
    for by, value in locators:
        if not value:
            continue
            
        try:
            if by == AppiumBy.XPATH:
                return driver.find_element(by, value)
            else:
                return driver.find_element(by, value)
        except:
            continue
            
    return None


def resize_image(image_data: bytes, max_size: Tuple[int, int] = (1024, 1024)) -> bytes:
    """Resize an image while maintaining aspect ratio.
    
    Args:
        image_data: Binary image data
        max_size: Maximum (width, height)
        
    Returns:
        Resized image as bytes
    """
    try:
        img = Image.open(io.BytesIO(image_data))
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed (for JPEG compatibility)
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
            
        # Save to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85)
        return img_byte_arr.getvalue()
        
    except Exception as e:
        raise Exception(f"Failed to resize image: {e}")


def ensure_directory(path: str) -> Path:
    """Ensure a directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
