from typing import Optional, Iterator, Dict
from .debug_agents import SingleAgent
from .utils import split_into_chunks, get_default_prompts
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChainOfAgents:
    """Main class for the Chain of Agents implementation."""
    
    def __init__(
        self,
        model: str = "gpt-4.1-mini",  
        chunk_size: int = 5000,
        prompt: Optional[str] = None,
        max_new_tokens: int = 2000,
    ):
        """
        Initialize the Chain of Agents.
        
        Args:
            worker_model: Model to use for worker agents
            manager_model: Model to use for manager agent
            chunk_size: Maximum tokens per chunk
            worker_prompt: Custom system prompt for workers
            manager_prompt: Custom system prompt for manager
        """
        default_prompt = get_default_prompts()
        
        
        self.prompt = prompt or default_prompt  # Reusing worker prompt for react agent
        self.chunk_size = chunk_size
        self.model = model
        self.max_new_tokens = max_new_tokens
        
        logger.info(f"Initialized Chain of Agents with {worker_model} workers and {manager_model} manager")
    
    # def process(self, input_text: str, query: str) -> str:
        """
        Process a long text input using the Chain of Agents.
        
        Args:
            input_text: The long input text to process
            query: The user's query about the text
            
        Returns:
            str: The final response from the manager agent
        """
        # Split text into chunks
        chunks = split_into_chunks(input_text, self.chunk_size, self.model)
        
        # Process chunks with worker agents
        worker_outputs = []
        previous_cu = None
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")

            # Worker step
            worker = WorkerAgent(self.worker_model, self.worker_prompt)
            output = worker.process_chunk(chunk, query, previous_cu)

            # Critic step
            critic = CriticAgent(self.worker_model, "You are a critic agent who verifies and improves the worker output.")
            critic_output = critic.critique([worker_output], query)
            worker_outputs.append(critic_output)
            previous_cu = critic_output
        
        # Synthesize results with manager agent
        manager = ManagerAgent(self.manager_model, self.manager_prompt)
        final_output = manager.synthesize(worker_outputs, query)
        
        return final_output 
    
    # def process_stream(self, input_text: str, query: str) -> Iterator[Dict[str, str]]:
        """Process text with streaming - yields worker and manager messages."""
        worker_outputs = []
        previous_cu = None
        
        chunks = split_into_chunks(input_text, self.chunk_size, self.worker_model)
        total_chunks = len(chunks)
        
        # Debug logging for metadata
        metadata_message = {
            "type": "metadata",
            "content": json.dumps({
                "total_chunks": total_chunks,
                "total_pages": getattr(input_text, 'total_pages', 0)
            })
        }
        logger.info(f"Sending metadata: {metadata_message}")  # Debug log
        yield metadata_message
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{total_chunks}")
            # 🟢 Worker step
            worker = WorkerAgent(self.worker_model, self.worker_prompt)
            worker_output = worker.process_chunk(chunk, query, previous_cu)

            # 🟢 Critic step
            critic = CriticAgent(self.worker_model, "You are a critic agent who verifies and improves the worker output.")
            critic_output = critic.critique([worker_output], query)

            worker_outputs.append(critic_output)
            previous_cu = critic_output

            # Send combined Worker–Critic message
            yield {
                "type": "worker",
                "content": worker_output,
                "critic_content": critic_output,
                "progress": {
                    "current": i + 1,
                    "total": total_chunks
                }
            }
        
        logger.info("Processing manager synthesis")
        manager = ManagerAgent(self.manager_model, self.manager_prompt)
        final_output = manager.synthesize(worker_outputs, query)
        
        yield {
            "type": "manager",
            "content": final_output
        } 