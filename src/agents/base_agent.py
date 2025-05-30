"""Base agent class for Android automation."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseAgent(ABC):
    """Base class for all automation agents."""
    
    def __init__(self):
        """Initialize the base agent."""
        self.driver = None
        self.session_started = False
    
    @abstractmethod
    def start(self) -> bool:
        """Start the agent session.
        
        Returns:
            bool: True if the session started successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the agent session."""
        pass
    
    def is_running(self) -> bool:
        """Check if the agent session is active.
        
        Returns:
            bool: True if the session is active, False otherwise
        """
        return self.session_started and self.driver is not None
