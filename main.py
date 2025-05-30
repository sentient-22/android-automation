import argparse
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.absolute()))

from src.agents.llm_agent import LLMAgent
from src.utils.logger import setup_logger, logger
from src.data.dataset import VLMDataset


def print_action(action: dict):
    """Log action details in a readable format"""
    logger.info("\n" + "="*50)
    logger.info(f"Action: {action.get('action', 'unknown').upper()}")
    if 'element' in action and action['element']:
        logger.info("Target Element:")
        for k, v in action['element'].items():
            logger.info(f"  {k}: {v}")
    if 'text' in action and action['text']:
        logger.info(f"Text: {action['text']}")
    if 'reasoning' in action:
        logger.info(f"Reasoning: {action['reasoning']}")
    logger.info("="*50 + "\n")

def setup_argparse():
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(description='LLM Android Automation POC')
    parser.add_argument('--task', type=str, 
                      help='Task to perform', 
                      default="Open the settings app")
    parser.add_argument('--max-steps', type=int, 
                      default=10, 
                      help='Maximum number of steps to perform')
    parser.add_argument('--log-level', type=str,
                      default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    parser.add_argument('--output-dir', type=str,
                      default='output',
                      help='Output directory for logs and data')
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = setup_argparse()
    
    # Setup logging
    log_dir = Path(args.output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    setup_logger(
        name='android_automation',
        log_level=args.log_level,
        log_file=log_dir / 'automation.log'
    )
    
    logger.info("\n" + "="*50)
    logger.info("LLM Android Automation POC")
    logger.info("="*50)
    logger.info(f"Task: {args.task}")
    logger.info(f"Max Steps: {args.max_steps}")
    logger.info(f"Log Level: {args.log_level}")
    
    # Initialize the agent
    try:
        agent = LLMAgent()
        
        if not agent.start():
            logger.error("❌ Failed to start session")
            return 1
            
        logger.info("✅ Session started successfully")
        
        # Main interaction loop
        for step in range(args.max_steps):
            logger.info(f"\n🔍 Step {step + 1}/{args.max_steps}")
            
            # Generate next action
            action, state = agent.generate_action(args.task)
            print_action(action)
            
            # Check for completion or errors
            if action.get('action') == 'complete':
                logger.info("✅ Task completed successfully!")
                break
                
            if action.get('action') == 'error':
                logger.error(f"❌ Error: {action.get('error', 'Unknown error')}")
                if 'raw_response' in action:
                    logger.debug(f"Raw response: {action['raw_response'][:200]}...")
                break
            
            # Execute the action
            logger.info("Executing action...")
            success = agent.execute_action(action)
            
            if not success:
                logger.warning("⚠️ Action failed to execute")
                # Optionally, we could ask the LLM for a different approach
                continue
                
            # Small delay to allow the UI to update
            import time
            time.sleep(2)
            
        else:
            logger.warning(f"⚠️ Reached maximum number of steps ({args.max_steps}) without completing the task")
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Operation cancelled by user")
    except Exception as e:
        logger.critical(f"\n❌ An error occurred: {str(e)}", exc_info=True)
        return 1
    finally:
        try:
            agent.stop()
            logger.info("\nSession ended")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    return 0

if __name__ == "__main__":
    main()
