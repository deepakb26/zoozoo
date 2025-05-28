import boto3
import json
import os

class SupervisorAgentClient:
    """
    Client for interacting with a pre-configured Bedrock supervisor agent.
    """
    
    def __init__(self, agent_id=None, agent_alias_id=None, region_name="us-east-1"):
        """
        Initialize the supervisor agent client.
        
        Args:
            agent_id (str): The ID of the Bedrock agent
            agent_alias_id (str): The alias ID of the Bedrock agent
            region_name (str): AWS region name
        """
        self.agent_id = agent_id or os.environ.get('BEDROCK_AGENT_ID')
        self.agent_alias_id = agent_alias_id or os.environ.get('BEDROCK_AGENT_ALIAS_ID')
        self.bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=region_name)
    
    def invoke_agent(self, message, session_id=None):
        """
        Invoke the supervisor agent with a message.
        
        Args:
            message (str): The message to send to the agent
            session_id (str, optional): Session ID for continuing a conversation
            
        Returns:
            dict: The agent's response
        """
        if not self.agent_id or not self.agent_alias_id:
            raise ValueError("Agent ID and Agent Alias ID must be provided")
        
        request = {
            "inputText": message,
            "agentId": self.agent_id,
            "agentAliasId": self.agent_alias_id,
        }
        
        if session_id:
            request["sessionId"] = session_id
        
        try:
            response = self.bedrock_agent_runtime.invoke_agent(
                **request
            )
            
            # Process the response chunks
            result = {"messages": []}
            for event in response.get("completion"):
                chunk = event.get("chunk")
                if chunk and "bytes" in chunk:
                    # Decode the bytes to a string
                    message_chunk = json.loads(chunk["bytes"].decode('utf-8'))
                    result["messages"].append(message_chunk)
            
            # If a session ID was created, include it in the result
            if "sessionId" in response:
                result["sessionId"] = response["sessionId"]
                
            return result
            
        except Exception as e:
            print(f"Error invoking agent: {e}")
            return {"error": str(e)}