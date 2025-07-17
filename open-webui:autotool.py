"""
title: AutoTool Filter
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 2.1.0
required_open_webui_version: 0.6.0
"""

from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Any, Optional, Dict, List
import json
import re
import logging

# Updated imports for OpenWebUI 0.6.16
from open_webui.models.users import Users
from open_webui.models.tools import Tools
from open_webui.models.models import Models
from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.misc import get_last_user_message

# Set up logging
logger = logging.getLogger(__name__)


class Filter:
    class Valves(BaseModel):
        template: str = Field(
            default="""Tools: {{TOOLS}}

Analyze the user's request and return ONLY a JSON list of tool IDs that match the query.
- If no tools match, return: []
- If one tool matches, return: ["tool_id"]
- If multiple tools match, return: ["tool_id1", "tool_id2"]
- Consider the entire conversation context
- Be conservative - only select tools that clearly match the request
- Return ONLY the JSON list, no other text"""
        )
        status: bool = Field(default=True)
        max_message_history: int = Field(default=10, description="Maximum number of messages to include in context")
        debug: bool = Field(default=False, description="Enable debug logging")

    def __init__(self):
        self.valves = self.Valves()
        self._tool_cache: Dict[str, Dict] = {}

    def _get_available_tools(self, model_info: Dict) -> List[Dict]:
        """Get available tools with caching"""
        try:
            # Try multiple methods to get tools due to API changes
            all_tools = []
            
            # Method 1: Try get_tools() (older API)
            try:
                all_tools = Tools.get_tools()
                if self.valves.debug:
                    logger.info(f"Found {len(all_tools)} tools using get_tools()")
            except (AttributeError, TypeError):
                # Method 2: Try get_all() (newer API)
                try:
                    all_tools = Tools.get_all()
                    if self.valves.debug:
                        logger.info(f"Found {len(all_tools)} tools using get_all()")
                except (AttributeError, TypeError):
                    # Method 3: Try accessing via query() or similar
                    try:
                        all_tools = Tools.query.all()
                        if self.valves.debug:
                            logger.info(f"Found {len(all_tools)} tools using query.all()")
                    except (AttributeError, TypeError):
                        logger.error("Could not access tools - API may have changed")
                        return []
            
            # Get tool IDs from model info
            available_tool_ids = (
                model_info.get("info", {}).get("meta", {}).get("toolIds", [])
            )
            
            if self.valves.debug:
                logger.info(f"Model has {len(available_tool_ids)} tool IDs: {available_tool_ids}")
            
            available_tools = []
            for tool in all_tools:
                # Handle different tool object structures
                tool_id = getattr(tool, 'id', None) or getattr(tool, 'tool_id', None)
                if not tool_id:
                    continue
                    
                if tool_id in available_tool_ids:
                    # Try different ways to get description
                    description = (
                        getattr(tool, 'description', None) or 
                        getattr(getattr(tool, 'meta', None), 'description', None) or
                        getattr(tool, 'name', None) or
                        tool_id
                    )
                    
                    tool_info = {
                        "id": tool_id,
                        "description": description
                    }
                    available_tools.append(tool_info)
                    self._tool_cache[tool_id] = tool_info
            
            if self.valves.debug:
                logger.info(f"Available tools: {[t['id'] for t in available_tools]}")
            
            return available_tools
        except Exception as e:
            logger.error(f"Error getting available tools: {e}")
            return []

    def _parse_tool_response(self, content: str) -> List[str]:
        """Safely parse tool selection response"""
        if not content:
            return []
        
        try:
            # Clean the content
            content = content.strip()
            
            # Find JSON list pattern
            json_pattern = r'\[(?:[^\[\]]*(?:"[^"]*"[^\[\]]*)*)*\]'
            match = re.search(json_pattern, content)
            
            if match:
                json_str = match.group(0)
                # Parse JSON safely
                result = json.loads(json_str)
                
                # Validate result
                if isinstance(result, list) and all(isinstance(item, str) for item in result):
                    return result
            
            # Fallback: try to parse the entire content as JSON
            result = json.loads(content)
            if isinstance(result, list) and all(isinstance(item, str) for item in result):
                return result
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool response as JSON: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing tool response: {e}")
        
        return []

    def _build_context_prompt(self, messages: List[Dict], user_message: str) -> str:
        """Build context prompt with limited message history"""
        try:
            # Limit message history to prevent context overflow
            recent_messages = messages[-self.valves.max_message_history:]
            
            history_lines = []
            for msg in reversed(recent_messages):
                role = msg.get('role', 'unknown').upper()
                content = msg.get('content', '')
                # Truncate very long messages
                if len(content) > 500:
                    content = content[:500] + "..."
                history_lines.append(f"{role}: \"\"\"{content}\"\"\"")
            
            return (
                "Recent conversation history:\n" + 
                "\n".join(history_lines) + 
                f"\n\nCurrent query: {user_message}"
            )
        except Exception as e:
            logger.error(f"Error building context prompt: {e}")
            return f"Current query: {user_message}"

    async def _emit_status(self, event_emitter: Callable, description: str, done: bool = False):
        """Emit status with error handling"""
        try:
            if self.valves.status:
                await event_emitter({
                    "type": "status",
                    "data": {
                        "description": description,
                        "done": done,
                    },
                })
        except Exception as e:
            logger.error(f"Error emitting status: {e}")

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __request__: Any,
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        try:
            if self.valves.debug:
                logger.info(f"Filter called with model: {__model__}")
                logger.info(f"Body keys: {list(body.keys())}")
            
            messages = body.get("messages", [])
            user_message = get_last_user_message(messages)
            
            if not user_message:
                if self.valves.debug:
                    logger.info("No user message found")
                return body

            # Process existing tool outputs
            if "tool_outputs" in body and body["tool_outputs"]:
                await self._process_tool_outputs(body, messages)

            await self._emit_status(__event_emitter__, "Finding the right tools...")

            # Get available tools
            available_tools = self._get_available_tools(__model__ or {})
            
            if not available_tools:
                await self._emit_status(__event_emitter__, "No tools available for this model", True)
                if self.valves.debug:
                    logger.info("No available tools found")
                return body

            # Generate tool selection prompt
            system_prompt = self.valves.template.replace("{{TOOLS}}", json.dumps(available_tools, indent=2))
            context_prompt = self._build_context_prompt(messages, user_message)

            # Call LLM for tool selection
            selected_tools = await self._select_tools(
                body, __request__, __user__, system_prompt, context_prompt
            )

            if selected_tools:
                await self._configure_tools(body, selected_tools, available_tools)
                tool_names = [
                    next((t["description"] for t in available_tools if t["id"] == tool_id), tool_id)
                    for tool_id in selected_tools
                ]
                await self._emit_status(
                    __event_emitter__, 
                    f"Selected tools: {', '.join(tool_names)}", 
                    True
                )
                if self.valves.debug:
                    logger.info(f"Selected tools: {selected_tools}")
            else:
                await self._emit_status(__event_emitter__, "No matching tools found", True)
                if self.valves.debug:
                    logger.info("No tools selected")

        except Exception as e:
            logger.error(f"Error in AutoTool Filter: {e}")
            await self._emit_status(__event_emitter__, f"Error: {str(e)}", True)

        return body

    async def _process_tool_outputs(self, body: dict, messages: List[Dict]):
        """Process tool outputs and add to context"""
        try:
            for tool_output in body["tool_outputs"]:
                tool_id = tool_output.get("tool_id", "unknown_tool")
                output = tool_output.get("output", {})
                
                # Format output safely
                if isinstance(output, dict):
                    output_str = json.dumps(output, indent=2)
                else:
                    output_str = str(output)
                
                # Limit output length
                if len(output_str) > 1000:
                    output_str = output_str[:1000] + "... (truncated)"
                
                tool_message = {
                    "role": "system",
                    "content": f"Tool `{tool_id}` executed with output: {output_str}",
                }
                messages.append(tool_message)
                
        except Exception as e:
            logger.error(f"Error processing tool outputs: {e}")

    async def _select_tools(
        self, 
        body: dict, 
        request: Any, 
        user: dict, 
        system_prompt: str, 
        context_prompt: str
    ) -> List[str]:
        """Select tools using LLM"""
        try:
            payload = {
                "model": body["model"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_prompt},
                ],
                "stream": False,
                "temperature": 0.1,  # Low temperature for consistent tool selection
            }

            user_obj = Users.get_user_by_id(user["id"])
            response = await generate_chat_completion(
                request=request, 
                form_data=payload, 
                user=user_obj
            )
            
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if self.valves.debug:
                logger.info(f"Tool selection response: {content}")
            
            return self._parse_tool_response(content)
            
        except Exception as e:
            logger.error(f"Error selecting tools: {e}")
            return []

    async def _configure_tools(self, body: dict, selected_tools: List[str], available_tools: List[Dict]):
        """Configure selected tools in the body"""
        try:
            body["tool_ids"] = selected_tools
            
            # Add tool availability message
            tool_descriptions = [
                next((t["description"] for t in available_tools if t["id"] == tool_id), tool_id)
                for tool_id in selected_tools
            ]
            
            tools_message = {
                "role": "system",
                "content": f"Available tools for this request: {', '.join(tool_descriptions)}. Use these tools as needed to complete the user's request.",
            }
            body["messages"].append(tools_message)
            
            # Initialize tools array if needed
            if "tools" not in body:
                body["tools"] = []
            
            # Add tool configurations
            existing_tool_ids = {tool.get("id") for tool in body["tools"]}
            
            for tool_id in selected_tools:
                if tool_id not in existing_tool_ids:
                    tool_info = next((t for t in available_tools if t["id"] == tool_id), None)
                    if tool_info:
                        body["tools"].append({
                            "id": tool_id,
                            "description": tool_info["description"],
                            "enabled": True,
                        })
                        
        except Exception as e:
            logger.error(f"Error configuring tools: {e}")
