import os
import json
import anthropic
import base64
import time
from PIL import Image
import io
from datetime import datetime
from typing import Any
import uuid

class ClaudeChat:
    """Custom node for ComfyUI that provides chat interface with Claude models with optional image support"""
    
    # Kolory dla logów
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    
    # Ścieżka do pliku licznika
    COUNTER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'usage_counter.json')
    
    # Definicje modeli i ich parametrów
    MODELS = {
        "claude-3-5-sonnet-latest": {
            "name": "claude-3-5-sonnet-20241022",
            "description": "Latest Sonnet - Balanced speed and intelligence",
            "cost_input": 3.00,
            "cost_output": 15.00,
            "max_output_tokens": 8192,
            "context_window": 200000,
            "has_vision": True,
            "comparative_latency": "Fast"
        },
        "claude-3-opus": {
            "name": "claude-3-opus-20240229",
            "description": "Opus - Most powerful for complex tasks",
            "cost_input": 15.00,
            "cost_output": 75.00,
            "max_output_tokens": 4096,
            "context_window": 200000,
            "has_vision": True,
            "comparative_latency": "Moderately fast"
        },
        "claude-3-sonnet": {
            "name": "claude-3-sonnet-20240229",
            "description": "Sonnet - Balanced performance",
            "cost_input": 3.00,
            "cost_output": 15.00,
            "max_output_tokens": 4096,
            "context_window": 200000,
            "has_vision": True,
            "comparative_latency": "Fast"
        },
        "claude-3-haiku": {
            "name": "claude-3-haiku-20240307",
            "description": "Haiku - Fastest and most compact",
            "cost_input": 0.25,
            "cost_output": 1.25,
            "max_output_tokens": 4096,
            "context_window": 200000,
            "has_vision": True,
            "comparative_latency": "Fastest"
        }
    }
    
    # API Rate Limits dla Tier 1 (domyślne)
    API_LIMITS = {
        "requests_per_minute": 50,  # RPM
        "tokens_per_minute": 40000,  # TPM
        "tokens_per_day": 1000000,  # TPD
    }

    # Tracking wykorzystania API
    api_usage = {
        "last_request_time": 0,
        "requests_this_minute": 0,
        "tokens_this_minute": 0,
        "tokens_today": 0,
        "last_minute_reset": 0,
        "last_day_reset": 0
    }
    
    # Stałe dla summaryzacji
    SUMMARY_THRESHOLD = 0.75  # 75% maksymalnej liczby tokenów
    MESSAGES_TO_SUMMARIZE = 3  # Liczba ostatnich wiadomości do streszczenia
    SUMMARY_SYSTEM_PROMPT = """You are a precise summarization assistant. Provide the essence of the conversation in one concise sentence. Focus only on the most important points and maintain the context. Be direct and brief."""
    
    DEFAULT_MODEL = "claude-3-5-sonnet-latest"
    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant in ComfyUI environment. Provide clear, accurate, and concise responses. When discussing code or technical concepts, use specific examples where helpful. If you're not sure about something, say so."""
    
    # Stałe dla zarządzania historią
    MAX_HISTORY_LENGTH = 20  # Maksymalna liczba wiadomości w historii
    MAX_MESSAGE_TOKENS = 100000  # Maksymalna liczba tokenów w całej konwersacji
    
    def _load_counter_static():
        """Static method to load counter value"""
        try:
            counter_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'usage_counter.json')
            if os.path.exists(counter_file):
                with open(counter_file, 'r') as f:
                    data = json.load(f)
                    return data.get('usage_count', 0)
        except Exception as e:
            print(f"Error loading counter: {e}")
        return 0
    
    _current_usage_count = _load_counter_static()
    
    def __init__(self):
        self.usage_count = self._load_counter()
        self.__class__._current_usage_count = self.usage_count
        self.client = None
        self.conversation_history = []
        self.session_id = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Your Anthropic API key. Can be set as ANTHROPIC_API_KEY environment variable."
                }),
                "model": (list(cls.MODELS.keys()), {
                    "default": cls.DEFAULT_MODEL,
                    "tooltip": "Select Claude model. Each has different capabilities and costs."
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Hello, Claude!",
                    "tooltip": "The message to send to Claude"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": cls.DEFAULT_SYSTEM_PROMPT,
                    "tooltip": "System prompt to set context and behavior for Claude"
                }),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Maximum number of tokens in response"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Controls randomness (0 = deterministic, 1 = creative)"
                }),
                "new_session": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Start a new conversation session"
                }),
                "clear_history": (["no", "yes", "all"], {
                    "default": "no",
                    "tooltip": "Clear history options:\nno: Keep history\nyes: Clear current session\nall: Clear all saved conversations"
                }),
                "summarize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically summarize conversation history when approaching token limits"
                }),
                "show_model_info": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Display detailed information about the selected model"
                }),
                "buy_me_a_coffee": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Support the development! ☕"
                })
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional image to analyze. Only supported by Claude 3 models (except Haiku)."
                }),
                "image_prompt": ("STRING", {
                    "multiline": True,
                    "default": "What can you see in this image?",
                    "tooltip": "Specific question about the image. Used only when image is provided."
                }),
                "conversation_id": ("STRING", {
                    "default": "",
                    "tooltip": "Optional conversation ID to continue specific conversation"
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("response_text", "usage_info", "input_tokens", "stop_reason", 
                   "conversation_id", "history_messages", "cost_usd", "model_info")
    FUNCTION = "generate_response"
    CATEGORY = "AI/Claude"
    
    def _load_counter(self):
        """Load usage counter with error handling"""
        try:
            if os.path.exists(self.COUNTER_FILE):
                with open(self.COUNTER_FILE, 'r') as f:
                    data = json.load(f)
                    return data.get('usage_count', 0)
        except Exception as e:
            print(f"{self.RED}Error loading counter: {e}{self.ENDC}")
        return 0

    def _save_counter(self):
        """Save usage counter with error handling"""
        try:
            with open(self.COUNTER_FILE, 'w') as f:
                json.dump({'usage_count': self.usage_count}, f)
        except Exception as e:
            print(f"{self.RED}Error saving counter: {e}{self.ENDC}")

    def _increment_counter(self):
        """Increment and save usage counter"""
        self.usage_count += 1
        self._save_counter()
        self.__class__._current_usage_count = self.usage_count
        print(f"{self.BLUE}ClaudeChat: Usage count updated: {self.usage_count}{self.ENDC}")

    def _generate_session_id(self):
        """Generate a unique session ID"""
        return str(uuid.uuid4())

    def _check_api_limits(self, estimated_tokens: int) -> bool:
        """Check if request would exceed API limits"""
        current_time = time.time()
        
        # Reset counters if needed
        if current_time - self.api_usage["last_minute_reset"] >= 60:
            self.api_usage["requests_this_minute"] = 0
            self.api_usage["tokens_this_minute"] = 0
            self.api_usage["last_minute_reset"] = current_time
        
        if current_time - self.api_usage["last_day_reset"] >= 86400:
            self.api_usage["tokens_today"] = 0
            self.api_usage["last_day_reset"] = current_time

        # Check limits
        if self.api_usage["requests_this_minute"] >= self.API_LIMITS["requests_per_minute"]:
            raise Exception(f"Rate limit exceeded: {self.API_LIMITS['requests_per_minute']} requests per minute")
        
        if (self.api_usage["tokens_this_minute"] + estimated_tokens) > self.API_LIMITS["tokens_per_minute"]:
            raise Exception(f"Token rate limit exceeded: {self.API_LIMITS['tokens_per_minute']} tokens per minute")
        
        if (self.api_usage["tokens_today"] + estimated_tokens) > self.API_LIMITS["tokens_per_day"]:
            raise Exception(f"Daily token limit exceeded: {self.API_LIMITS['tokens_per_day']} tokens per day")
        
        return True

    def _update_api_usage(self, tokens_used: int):
        """Update API usage tracking"""
        current_time = time.time()
        self.api_usage["last_request_time"] = current_time
        self.api_usage["requests_this_minute"] += 1
        self.api_usage["tokens_this_minute"] += tokens_used
        self.api_usage["tokens_today"] += tokens_used

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD based on token usage and model"""
        try:
            model_config = self.MODELS[model]
            input_cost = (input_tokens / 1_000_000) * model_config["cost_input"]
            output_cost = (output_tokens / 1_000_000) * model_config["cost_output"]
            total_cost = input_cost + output_cost
            return round(total_cost, 4)
        except KeyError:
            print(f"{self.RED}Error: Model {model} not found in cost configuration{self.ENDC}")
            return 0.0
        except Exception as e:
            print(f"{self.RED}Error calculating cost: {str(e)}{self.ENDC}")
            return 0.0

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of tokens in text"""
        return len(text) // 4

    def _estimate_conversation_tokens(self) -> int:
        """Estimate total tokens in conversation history"""
        total_tokens = 0
        for msg in self.conversation_history:
            if isinstance(msg["content"], str):
                total_tokens += self._estimate_tokens(msg["content"])
            elif isinstance(msg["content"], list):
                # For messages with images, estimate text tokens only
                for content in msg["content"]:
                    if content["type"] == "text":
                        total_tokens += self._estimate_tokens(content["text"])
        return total_tokens

    def _check_token_limit(self) -> bool:
        """Check if conversation is within token limits"""
        total_tokens = sum(
            self._estimate_tokens(str(msg["content"])) 
            for msg in self.conversation_history
        )
        return total_tokens < self.MAX_MESSAGE_TOKENS

    def _encode_image_to_base64(self, image):
        """Convert image to base64"""
        try:
            # Konwersja tensora PyTorch na obraz PIL
            if len(image.shape) == 4:  # batch of images
                image = image[0]
            
            # Przygotowanie obrazu
            image = image.cpu().numpy()
            image = (image * 255).astype('uint8')
            if image.shape[0] == 3:  # CHW -> HWC
                image = image.transpose(1, 2, 0)
            
            # Konwersja na PIL Image
            pil_image = Image.fromarray(image)
            
            # Konwersja do JPEG i base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return img_str
            
        except Exception as e:
            print(f"{self.RED}ClaudeChat: Error encoding image: {str(e)}{self.ENDC}")
            return None
            
    def _update_conversation_history(self, role: str, content: Any):
        """Add a message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})
        self._truncate_history_if_needed()

    def _truncate_history_if_needed(self):
        """Truncate history if it exceeds limits"""
        if len(self.conversation_history) > self.MAX_HISTORY_LENGTH:
            keep_start = 2
            keep_end = self.MAX_HISTORY_LENGTH - keep_start
            self.conversation_history = (
                self.conversation_history[:keep_start] + 
                self.conversation_history[-keep_end:]
            )
            print(f"{self.YELLOW}ClaudeChat: Truncated conversation history to {len(self.conversation_history)} messages{self.ENDC}")

    def _load_conversation(self, conversation_id: str) -> bool:
        """Load specific conversation from file"""
        if not conversation_id:
            return False
            
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conversations')
        save_path = os.path.join(save_dir, f'conversation_{conversation_id}.json')
        
        try:
            if os.path.exists(save_path):
                with open(save_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.conversation_history = data['history']
                    self.session_id = conversation_id
                    print(f"{self.BLUE}Loaded conversation: {conversation_id}{self.ENDC}")
                    return True
            else:
                print(f"{self.YELLOW}Conversation {conversation_id} not found{self.ENDC}")
        except Exception as e:
            print(f"{self.RED}Error loading conversation: {str(e)}{self.ENDC}")
        return False

    def _save_conversation(self):
        """Save conversation history to file"""
        if self.session_id and self.conversation_history:
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conversations')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'conversation_{self.session_id}.json')
            
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'session_id': self.session_id,
                        'history': self.conversation_history,
                        'timestamp': str(datetime.now())
                    }, f, indent=2)
                print(f"{self.BLUE}ClaudeChat: Saved conversation to {save_path}{self.ENDC}")
            except Exception as e:
                print(f"{self.YELLOW}ClaudeChat: Could not save conversation: {str(e)}{self.ENDC}")

    def _clear_all_conversations(self):
        """Clear all saved conversations"""
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conversations')
        try:
            if os.path.exists(save_dir):
                for file in os.listdir(save_dir):
                    if file.startswith('conversation_') and file.endswith('.json'):
                        os.remove(os.path.join(save_dir, file))
                print(f"{self.BLUE}Cleared all saved conversations{self.ENDC}")
        except Exception as e:
            print(f"{self.RED}Error clearing conversations: {str(e)}{self.ENDC}")

    def _summarize_recent_messages(self, client: anthropic.Anthropic, model_name: str) -> str:
        """Summarize last few messages of conversation"""
        if len(self.conversation_history) < self.MESSAGES_TO_SUMMARIZE:
            return None

        # Get last messages to summarize
        messages_to_summarize = self.conversation_history[-self.MESSAGES_TO_SUMMARIZE:]
        
        # Prepare messages for summarization
        summarization_prompt = "Summarize this conversation fragment:\n\n"
        for msg in messages_to_summarize:
            role = "User" if msg["role"] == "user" else "Assistant"
            if isinstance(msg["content"], str):
                summarization_prompt += f"{role}: {msg['content']}\n"
            elif isinstance(msg["content"], list):
                # Handle messages with images
                text_content = "\n".join(
                    content["text"] for content in msg["content"] 
                    if content["type"] == "text"
                )
                summarization_prompt += f"{role}: {text_content}\n"

        try:
            # Make API call for summarization
            summary_response = client.messages.create(
                model=model_name,
                max_tokens=100,  # Limit summary length
                temperature=0.3,  # More deterministic for summaries
                system=self.SUMMARY_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": summarization_prompt}]
            )
            
            summary = summary_response.content[0].text if summary_response.content else ""
            print(f"{self.BLUE}Generated summary of recent messages{self.ENDC}")
            return summary

        except Exception as e:
            print(f"{self.YELLOW}Failed to generate summary: {str(e)}{self.ENDC}")
            return None

    def _manage_conversation_length(self, client: anthropic.Anthropic, model_name: str, 
                                  model_config: dict, summarize: bool = True):
        """Manage conversation length, optionally using summarization"""
        current_tokens = self._estimate_conversation_tokens()
        max_tokens = model_config["context_window"]
        
        if current_tokens > (max_tokens * self.SUMMARY_THRESHOLD):
            print(f"{self.YELLOW}Conversation approaching token limit ({current_tokens}/{max_tokens}){self.ENDC}")
            
            if summarize:
                # Try to summarize recent messages
                summary = self._summarize_recent_messages(client, model_name)
                if summary:
                    # Replace last few messages with summary
                    self.conversation_history = (
                        self.conversation_history[:-self.MESSAGES_TO_SUMMARIZE] +
                        [{"role": "assistant", "content": f"Summary of recent conversation: {summary}"}]
                    )
                    print(f"{self.GREEN}Replaced last {self.MESSAGES_TO_SUMMARIZE} messages with summary{self.ENDC}")
                    return
            
            # If summarization failed or was disabled, use standard truncation
            self._truncate_history_if_needed()
    def generate_response(self, api_key: str, model: str, prompt: str, system_prompt: str = "", 
                         max_tokens: int = 1024, temperature: float = 0.7,
                         new_session: bool = False, clear_history: str = "no",
                         show_model_info: bool = False, buy_me_a_coffee: bool = False,
                         summarize: bool = True, image: Any = None, 
                         image_prompt: str = None, conversation_id: str = None):
        """Main processing function with conversation summarization"""
        try:
            self._increment_counter()
            
            # Get model configuration and validate image support
            model_config = self.MODELS[model]
            model_name = model_config["name"]
            
            if image is not None:
                if not model_config["has_vision"]:
                    print(f"{self.YELLOW}Warning: Selected model {model} does not support vision. Ignoring image.{self.ENDC}")
                    image = None
                elif image_prompt is None:
                    image_prompt = "What can you see in this image?"
                    print(f"{self.BLUE}Using default image prompt: '{image_prompt}'{self.ENDC}")

            # Handle conversation loading and clearing
            if conversation_id and not new_session:
                if not self._load_conversation(conversation_id):
                    print(f"{self.YELLOW}Starting new conversation as {conversation_id} not found{self.ENDC}")
                    self.session_id = self._generate_session_id()
                    self.conversation_history = []
            
            if clear_history == "yes":
                self.conversation_history = []
                self.session_id = self._generate_session_id()
                print(f"{self.BLUE}Cleared current conversation history{self.ENDC}")
            elif clear_history == "all":
                self._clear_all_conversations()
                self.conversation_history = []
                self.session_id = self._generate_session_id()

            # Prepare message content
            if image is not None:
                try:
                    img_base64 = self._encode_image_to_base64(image)
                    message_content = [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": image_prompt
                        }
                    ]
                    print(f"{self.BLUE}Image successfully encoded and added to message{self.ENDC}")
                except Exception as e:
                    print(f"{self.YELLOW}Warning: Failed to process image: {str(e)}. Continuing with text only.{self.ENDC}")
                    message_content = prompt
            else:
                message_content = prompt

            # Initialize Anthropic client
            if not api_key and 'ANTHROPIC_API_KEY' in os.environ:
                api_key = os.environ['ANTHROPIC_API_KEY']
            
            if not api_key:
                raise ValueError(f"{self.RED}No API key provided. Set it in the node or as ANTHROPIC_API_KEY environment variable.{self.ENDC}")

            client = anthropic.Anthropic(api_key=api_key)
            
            # Estimate token usage and check limits
            estimated_tokens = self._estimate_tokens(str(message_content)) + max_tokens
            self._check_api_limits(estimated_tokens)
            
            # Manage conversation length before adding new message
            self._manage_conversation_length(client, model_name, model_config, summarize)
            
            # Add new message to history
            self._update_conversation_history("user", message_content)
            
            # Make API call
            response = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt if system_prompt else None,
                messages=self.conversation_history
            )
            
            # Process response
            response_text = response.content[0].text if response.content else ""
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = self._calculate_cost(model, input_tokens, output_tokens)
            
            # Update API usage
            total_tokens = input_tokens + output_tokens
            self._update_api_usage(total_tokens)
            
            # Add response to history
            self._update_conversation_history("assistant", response_text)
            
            # Save conversation
            self._save_conversation()
            
            # Prepare usage info
            usage_info = (
                f"Model: {model_name}\n"
                f"Input tokens: {input_tokens}, Output tokens: {output_tokens}\n"
                f"Cost: ${cost:.4f} USD\n"
                f"Image included: {'Yes' if image is not None else 'No'}\n"
                f"Conversation tokens: {self._estimate_conversation_tokens()}/{model_config['context_window']}\n"
                f"Summarization: {'Enabled' if summarize else 'Disabled'}\n"
                f"Requests this minute: {self.api_usage['requests_this_minute']}/{self.API_LIMITS['requests_per_minute']}\n"
                f"Tokens this minute: {self.api_usage['tokens_this_minute']}/{self.API_LIMITS['tokens_per_minute']}\n"
                f"Tokens today: {self.api_usage['tokens_today']}/{self.API_LIMITS['tokens_per_day']}"
            )
            
            print(f"{self.GREEN}Successfully generated response{self.ENDC}")
            print(f"{self.BLUE}{usage_info}{self.ENDC}")
            
            return (
                response_text,
                usage_info,
                input_tokens,
                response.stop_reason or "unknown",
                self.session_id,
                len(self.conversation_history),
                cost,
                self._get_model_info(model) if show_model_info else ""
            )

        except Exception as e:
            print(f"{self.RED}Error generating response: {str(e)}{self.ENDC}")
            return ("Error: " + str(e), "Error", 0, "error", 
                   self.session_id if self.session_id else "error", 0, 0.0, "")

# Node registration
NODE_CLASS_MAPPINGS = {
    "AnthropicChat": ClaudeChat
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnthropicChat": "Claude Chat"
}