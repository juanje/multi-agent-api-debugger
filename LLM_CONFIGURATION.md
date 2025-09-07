# LLM Configuration Guide

This document describes how to configure and use the LLM models in the multi-agent system.

## Overview

The multi-agent system uses different LLM models optimized for specific agent roles. Each agent has its own model configuration with tailored parameters and system prompts.

## Model Configuration

### Agent-Specific Models

| Agent | Model | Purpose | Temperature | Max Tokens |
|-------|-------|---------|-------------|------------|
| Supervisor | `qwen3:8b` | Routing and orchestration | 0.0 | 512 |
| API Operator | `llama3.2:3b` | Tool execution | 0.0 | 256 |
| Debugger | `qwen2.5-coder:7b` | Error analysis | 0.3 | 1024 |
| Knowledge Assistant | `llama3-chatqa:8b` | Q&A and information | 0.7 | 512 |
| Response Synthesizer | `gemma3:4b` | Response formatting | 0.6 | 768 |

### Model Parameters

Each model is configured with the following parameters:

- **Temperature**: Controls randomness (0.0 = deterministic, 1.0 = creative)
- **Top-p**: Nucleus sampling parameter (0.0-1.0)
- **Top-k**: Top-k sampling parameter
- **Repeat Penalty**: Penalty for repeated tokens
- **Timeout**: Request timeout in seconds
- **Max Tokens**: Maximum tokens to generate

## Prerequisites

### 1. Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve
```

### 2. Pull Required Models

```bash
# Pull all required models
ollama pull qwen3:8b
ollama pull llama3.2:3b
ollama pull qwen2.5-coder:7b
ollama pull llama3-chatqa:8b
ollama pull gemma3:4b
```

### 3. Install Python Dependencies

```bash
# Install the updated dependencies
uv sync
```

## Usage

### Basic Usage

```python
from multi_agent import AgentType, generate_agent_response

# Generate a response using the Supervisor agent
response = await generate_agent_response(
    AgentType.SUPERVISOR,
    "I need to run a data processing job"
)
```

### Advanced Usage

```python
from multi_agent.llm_service import get_llm_service
from multi_agent.llm_config import AgentType

# Get the LLM service
service = await get_llm_service()

# Generate with custom system prompt
response = await service.generate_with_system_prompt(
    AgentType.DEBUGGER,
    "Analyze this error: Connection timeout",
    system_prompt="You are an expert debugger..."
)

# Generate with template
response = await service.generate_with_template(
    AgentType.KNOWLEDGE_ASSISTANT,
    "What is {concept}?",
    {"concept": "machine learning"},
    system_prompt="Answer questions about technology..."
)
```

### Streaming Responses

```python
# Stream responses for real-time output
async for chunk in service.stream_response(
    AgentType.RESPONSE_SYNTHESIZER,
    [HumanMessage(content="Format this data...")]
):
    print(chunk, end="", flush=True)
```

## Configuration

### Customizing Model Parameters

```python
from multi_agent.llm_config import LLMConfig, AgentLLMConfig, update_agent_config, AgentType

# Create custom configuration
custom_config = LLMConfig(
    model_name="custom-model:8b",
    temperature=0.5,
    max_tokens=1024,
    top_p=0.9,
    top_k=40,
    repeat_penalty=1.1,
    timeout=60
)

# Create agent configuration
agent_config = AgentLLMConfig(
    agent_type=AgentType.SUPERVISOR,
    llm_config=custom_config,
    system_prompt="Custom system prompt...",
    description="Custom supervisor agent"
)

# Update the configuration
update_agent_config(AgentType.SUPERVISOR, agent_config)
```

### Environment Variables

You can configure the Ollama host using environment variables:

```bash
export OLLAMA_HOST="http://localhost:11434"
```

Or in your code:

```python
from multi_agent.llm_service import get_llm_service

service = await get_llm_service(ollama_host="http://your-ollama-host:11434")
```

## Testing

### Run the Test Suite

```bash
# Test LLM configuration and connectivity
python test_llm_setup.py
```

### Manual Testing

```python
import asyncio
from multi_agent import AgentType, generate_agent_response

async def test_agents():
    # Test each agent
    agents = [
        AgentType.SUPERVISOR,
        AgentType.API_OPERATOR,
        AgentType.DEBUGGER,
        AgentType.KNOWLEDGE_ASSISTANT,
        AgentType.RESPONSE_SYNTHESIZER
    ]
    
    for agent in agents:
        try:
            response = await generate_agent_response(
                agent,
                "Hello, can you help me?"
            )
            print(f"{agent.value}: {response[:100]}...")
        except Exception as e:
            print(f"{agent.value}: Error - {e}")

asyncio.run(test_agents())
```

## Troubleshooting

### Common Issues

1. **Model Not Found**
   - Ensure Ollama is running: `ollama serve`
   - Check if model is pulled: `ollama list`
   - Pull the model: `ollama pull <model-name>`

2. **Connection Refused**
   - Verify Ollama is running on the correct port
   - Check firewall settings
   - Ensure the host URL is correct

3. **Timeout Errors**
   - Increase timeout in model configuration
   - Check system resources (CPU/GPU)
   - Consider using smaller models for testing

4. **Memory Issues**
   - Use smaller models for development
   - Close unused model instances
   - Monitor system memory usage

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

### Model Selection

- **Development**: Use smaller models (3B-7B parameters)
- **Production**: Use larger models (8B+ parameters) for better quality
- **Testing**: Use the smallest available models

### Resource Management

- **CPU**: Models run on CPU by default
- **GPU**: Set `OLLAMA_GPU=1` for GPU acceleration
- **Memory**: Monitor RAM usage, especially with multiple models

### Caching

The LLM service automatically caches model instances. For production use, consider implementing response caching for frequently asked questions.

## Security Considerations

- **Local Models**: All models run locally, no data sent to external services
- **Network**: Ensure Ollama is not exposed to external networks
- **Authentication**: Consider implementing authentication for production use
- **Data Privacy**: All conversations remain on your local system

## Monitoring

### Health Checks

```python
# Check model availability
service = await get_llm_service()
validation_results = await service.validate_models()
print(validation_results)
```

### Model Information

```python
# Get detailed model information
model_info = service.get_all_model_info()
for agent, info in model_info.items():
    print(f"{agent}: {info['model_name']} - {info['description']}")
```

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review Ollama documentation: https://ollama.com/docs
3. Check LangChain Ollama documentation: https://python.langchain.com/docs/integrations/chat/ollama
4. Open an issue in the project repository
